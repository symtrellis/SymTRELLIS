"""Extract pose-aligned TRELLIS1 and SAM3D sparse latents."""

import os

os.environ["ATTN_BACKEND"] = "flash_attn"
os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"

import argparse
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
import trellis.models as trellis1_models
import trellis.modules.sparse as trellis_sp
import utils3d
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sam3d_sp
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor as Sam3DSparseTensor

from dataset.base import format_entry_name
from preprocess.dataset.base import DatasetFiles, DatasetWorkspace
from preprocess.utils import Pipeline, Stage, mesh_to_voxel_coords, sample_mesh_srt

VOXEL_RESOLUTION = 64


@torch.no_grad()
def srt_sampler(
    sha256: str,
    shape_files: DatasetFiles,
    device: torch.device,
    num_scale: int,
    min_scale: float,
    num_rots: int,
    num_perts: int,
    perturbation_rad_std: float,
    seed: int,
) -> Dict[str, object]:
    """Load one normalized mesh and sample all of its SRT poses."""
    mesh = o3d.io.read_triangle_mesh(str(shape_files.path(sha256)))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"empty normalized mesh: {sha256}")

    vertices_tensor = torch.from_numpy(vertices.copy()).to(
        device=device,
        dtype=torch.float32,
    )
    if not torch.isfinite(vertices_tensor).all():
        raise ValueError(f"non-finite vertices: {sha256}")
    faces_tensor = torch.from_numpy(faces.copy()).to(dtype=torch.int64).contiguous()

    shape_seed = int.from_bytes(hashlib.sha256(f"{sha256}:{seed}".encode("ascii")).digest()[:8], "big")
    sampled_vertices, transforms, center, base_scale = sample_mesh_srt(
        vertices_tensor,
        num_scale,
        min_scale,
        num_rots,
        num_perts,
        perturbation_rad_std,
        VOXEL_RESOLUTION,
        shape_seed,
    )

    return {
        "sampled_vertices": sampled_vertices.cpu(),
        "faces": faces_tensor.cpu(),
        "transforms": transforms.cpu(),
        "center": center.cpu(),
        "base_scale": base_scale,
    }


@torch.no_grad()
def loader(
    sha256: str,
    center: torch.Tensor,
    base_scale: float,
    render_files: DatasetFiles,
    camera_files: DatasetFiles,
    device: torch.device,
    view_chunk_size: int = 8,
) -> Dict[str, object]:
    """Load shared renders and express their cameras in normalized coordinates."""
    if view_chunk_size <= 0:
        raise ValueError("view_chunk_size must be positive")

    with camera_files.path(sha256).open("r") as file:
        camera_metadata = json.load(file)
    frames = camera_metadata["frames"]
    if not frames:
        raise ValueError(f"camera metadata has no frames: {sha256}")

    # Convert Blender right/up/backward cameras to right/down/forward cameras.
    render_c2w = torch.tensor([frame["transform_matrix"] for frame in frames], device=device, dtype=torch.float32)
    render_c2w[:, :3, 1:3] *= -1
    render_c2w[:, :3, 3] = base_scale * (render_c2w[:, :3, 3] - center.to(device=device, dtype=torch.float32))
    render_extrinsics = torch.linalg.inv(render_c2w)

    # Intrinsics use centered NDC coordinates and are unchanged by image resizing.
    render_fov = torch.tensor([frame["camera_angle_x"] for frame in frames], device=device, dtype=torch.float32)
    render_focal = 1 / torch.tan(render_fov / 2)
    render_intrinsics = torch.diag_embed(torch.stack((render_focal, render_focal, torch.ones_like(render_focal)), dim=-1))

    # Premultiply alpha and low-pass the 1472-pixel renders once for every pose.
    image_chunks = []
    with zipfile.ZipFile(render_files.path(sha256)) as archive:
        for start in range(0, len(frames), view_chunk_size):
            source_chunk = []
            for frame in frames[start : start + view_chunk_size]:
                with Image.open(io.BytesIO(archive.read(frame["file_path"]))) as image:
                    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
                source_chunk.append(torch.from_numpy(rgba).permute(2, 0, 1))

            rgba = torch.stack(source_chunk).to(device=device, dtype=torch.float32) / 255
            rgb = rgba[:, :3] * rgba[:, 3:4]
            rgb = F.interpolate(rgb, size=(736, 736), mode="bilinear", align_corners=False, antialias=True)
            image_chunks.append(rgb.to(dtype=torch.float16).cpu())

    return {
        "source_images": torch.cat(image_chunks),
        "render_extrinsics": render_extrinsics.cpu(),
        "render_intrinsics": render_intrinsics.cpu(),
    }


@torch.no_grad()
def affine_worker(
    sha256: str,
    sampled_vertices: torch.Tensor,
    faces: torch.Tensor,
    transforms: torch.Tensor,
    source_images: torch.Tensor,
    render_extrinsics: torch.Tensor,
    render_intrinsics: torch.Tensor,
    device: torch.device,
) -> List[Dict[str, object]]:
    """Compute all pose-view image affines and emit one task per pose."""
    num_scale, num_rots, num_perts = transforms.shape[:3]
    num_poses = num_scale * num_rots * num_perts
    num_views = len(render_extrinsics)

    transforms_flat = transforms.reshape(num_poses, 4, 4).to(
        device=device,
        dtype=torch.float32,
    )
    render_extrinsics = render_extrinsics.to(device=device, dtype=torch.float32)
    render_intrinsics = render_intrinsics.to(device=device, dtype=torch.float32)
    render_c2w = torch.linalg.inv(render_extrinsics)

    # E_pose = E_render T^-1 maps posed points back to their stored render pixels.
    pose_render_extrinsics = render_extrinsics[None] @ torch.linalg.inv(transforms_flat)[:, None]
    pose_linear = transforms_flat[:, :3, :3]
    pose_scale = torch.linalg.vector_norm(pose_linear, dim=(1, 2)) / math.sqrt(3)
    pose_orientation = pose_linear / pose_scale[:, None, None]

    target_centers = torch.einsum("pij,vj->pvi", pose_linear, render_c2w[:, :3, 3]) + transforms_flat[:, None, :3, 3]
    target_forward = F.normalize(torch.einsum("pij,vj->pvi", pose_orientation, render_c2w[:, :3, 2]), dim=-1)
    target_z_up = pose_orientation[:, :, 2]

    # Project transformed z-up into each image plane to build proper cameras.
    target_up = target_z_up[:, None] - (target_forward * target_z_up[:, None]).sum(dim=-1, keepdim=True) * target_forward
    if (torch.linalg.vector_norm(target_up, dim=-1) <= 1e-6).any():
        raise ValueError(f"z-up is undefined for a camera view: {sha256}")
    target_up = F.normalize(target_up, dim=-1)
    target_right = F.normalize(torch.cross(target_forward, target_up, dim=-1), dim=-1)
    target_up = torch.cross(target_right, target_forward, dim=-1)
    target_rotation = torch.stack((target_right, -target_up, target_forward), dim=-1)

    target_c2w = torch.eye(4, device=device)[None, None].expand(num_poses, num_views, 4, 4).clone()
    target_c2w[:, :, :3, :3] = target_rotation
    target_c2w[:, :, :3, 3] = target_centers
    target_extrinsics = torch.linalg.inv(target_c2w)

    # The projected sphere includes half a voxel diagonal around surface centers.
    sphere_center = transforms_flat[:, :3, 3]
    sphere_radius = 0.5 * pose_scale[:, None]
    sphere_radius = sphere_radius + math.sqrt(3) / (2 * VOXEL_RESOLUTION) + 1e-6
    sphere_camera = torch.einsum("pvij,pvj->pvi", target_extrinsics[:, :, :3, :3], sphere_center[:, None] - target_centers)
    sphere_x, sphere_y, sphere_z = sphere_camera.unbind(dim=-1)
    if (sphere_z <= sphere_radius).any():
        raise ValueError(f"normalized sphere intersects or passes a camera: {sha256}")

    denominator = sphere_z.square() - sphere_radius.square()
    horizontal_root = sphere_radius * torch.sqrt(sphere_x.square() + sphere_z.square() - sphere_radius.square())
    vertical_root = sphere_radius * torch.sqrt(sphere_y.square() + sphere_z.square() - sphere_radius.square())
    u_min = (sphere_x * sphere_z - horizontal_root) / denominator
    u_max = (sphere_x * sphere_z + horizontal_root) / denominator
    v_min = (sphere_y * sphere_z - vertical_root) / denominator
    v_max = (sphere_y * sphere_z + vertical_root) / denominator
    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2
    half_extent = torch.maximum((u_max - u_min) / 2, (v_max - v_min) / 2)

    # Centered target intrinsics crop the smallest square containing the sphere.
    target_focal = 1 / half_extent
    target_intrinsics = torch.zeros(num_poses, num_views, 3, 3, device=device)
    target_intrinsics[:, :, 0, 0] = target_focal
    target_intrinsics[:, :, 1, 1] = target_focal
    target_intrinsics[:, :, 0, 2] = -u_center * target_focal
    target_intrinsics[:, :, 1, 2] = -v_center * target_focal
    target_intrinsics[:, :, 2, 2] = 1

    # H maps target centered-NDC coordinates directly to source centered NDC.
    relative_camera = pose_render_extrinsics @ target_c2w
    target_to_source = render_intrinsics[None] @ relative_camera[:, :, :3, :3] @ torch.linalg.inv(target_intrinsics)
    target_to_source = target_to_source / target_to_source[:, :, 2:3, 2:3]

    vertices_flat = sampled_vertices.reshape(num_poses, *sampled_vertices.shape[3:])
    transforms_cpu = transforms.reshape(num_poses, 4, 4)
    affines = target_to_source[:, :, :2].cpu()
    target_extrinsics = target_extrinsics.cpu()
    target_intrinsics = target_intrinsics.cpu()

    results = []
    for pose_index in range(num_poses):
        scale_idx, remainder = divmod(pose_index, num_rots * num_perts)
        rot_idx, pert_idx = divmod(remainder, num_perts)
        results.append(
            {
                "vertices": vertices_flat[pose_index],
                "faces": faces,
                "transform": transforms_cpu[pose_index],
                "source_images": source_images,
                "affines": affines[pose_index],
                "target_extrinsics": target_extrinsics[pose_index],
                "target_intrinsics": target_intrinsics[pose_index],
                "entry_name": format_entry_name(scale_idx, rot_idx, pert_idx),
                "num_poses": num_poses,
                "pose_index": pose_index,
            }
        )
    return results


@torch.no_grad()
def dino_aggregate(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    source_images: torch.Tensor,
    affines: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    dino_model,
    view_chunk_size: int = 45,
    voxel_chunk_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Voxelize one pose and aggregate DINO features for both encoders."""
    if view_chunk_size <= 0 or voxel_chunk_size <= 0:
        raise ValueError("DINO chunk sizes must be positive")
    num_views = len(source_images)
    if not (len(affines) == len(target_extrinsics) == len(target_intrinsics) == num_views):
        raise ValueError("image and camera view counts do not match")

    device = next(dino_model.parameters()).device
    vertices = vertices.to(device=device, dtype=torch.float32)
    faces = faces.to(device=device, dtype=torch.int64)

    voxels = mesh_to_voxel_coords(vertices, faces, VOXEL_RESOLUTION)

    voxel_positions = (voxels.to(torch.float32) + 0.5) / VOXEL_RESOLUTION - 0.5
    feature_dim = int(dino_model.embed_dim)
    trellis1_feature_sum = torch.zeros(len(voxels), feature_dim, device=device, dtype=torch.float32)
    sam3d_feature_sum = torch.zeros_like(trellis1_feature_sum)
    sam3d_visible_count = torch.zeros(len(voxels), device=device, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    depth_tolerance = math.sqrt(3) / (2 * VOXEL_RESOLUTION)

    del vertices, faces

    for start in range(0, num_views, view_chunk_size):
        end = min(start + view_chunk_size, num_views)
        source_chunk = source_images[start:end].to(device=device, dtype=torch.float32)
        affine_chunk = affines[start:end].to(device=device, dtype=torch.float32)
        extrinsic_chunk = target_extrinsics[start:end].to(device=device, dtype=torch.float32)
        intrinsic_chunk = target_intrinsics[start:end].to(device=device, dtype=torch.float32)

        grid = F.affine_grid(affine_chunk, [end - start, 3, 518, 518], align_corners=False)
        images = F.grid_sample(source_chunk, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        images = (images - mean) / std

        tokens = dino_model(images, is_training=True)["x_prenorm"]
        tokens = tokens[:, dino_model.num_register_tokens + 1 :]
        feature_size = math.isqrt(tokens.shape[1])
        if feature_size * feature_size != tokens.shape[1]:
            raise ValueError("DINO patch tokens do not form a square feature map")
        feature_chunk = tokens.permute(0, 2, 1).reshape(end - start, feature_dim, feature_size, feature_size)

        uv_ndc, linear_depth = utils3d.torch.project_cv(voxel_positions, extrinsic_chunk, intrinsic_chunk)
        valid_projection = torch.isfinite(uv_ndc).all(dim=-1) & torch.isfinite(linear_depth) & (linear_depth > 0) & (uv_ndc.abs() <= 1).all(dim=-1)
        pixel_x = (((uv_ndc[..., 0] + 1) * feature_size - 1) / 2).round()
        pixel_y = (((uv_ndc[..., 1] + 1) * feature_size - 1) / 2).round()
        pixel_x = pixel_x.long().clamp(0, feature_size - 1)
        pixel_y = pixel_y.long().clamp(0, feature_size - 1)
        pixel_index = pixel_y * feature_size + pixel_x

        surface_depth = torch.full((end - start, feature_size * feature_size), torch.inf, device=device, dtype=linear_depth.dtype)
        surface_depth.scatter_reduce_(1, pixel_index, torch.where(valid_projection, linear_depth, torch.inf), reduce="amin", include_self=True)
        surface_depth = surface_depth.reshape(end - start, 1, feature_size, feature_size)
        reference_depth = F.grid_sample(surface_depth, uv_ndc.unsqueeze(1), mode="nearest", padding_mode="zeros", align_corners=False).squeeze(1).squeeze(1)
        visibility = valid_projection & (linear_depth <= reference_depth + depth_tolerance)

        for voxel_start in range(0, len(voxels), voxel_chunk_size):
            voxel_end = min(voxel_start + voxel_chunk_size, len(voxels))
            sampled = F.grid_sample(feature_chunk, uv_ndc[:, voxel_start:voxel_end].unsqueeze(1), mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(2)
            visible = visibility[:, voxel_start:voxel_end]
            trellis1_feature_sum[voxel_start:voxel_end] += sampled.sum(dim=0).transpose(0, 1)
            sam3d_feature_sum[voxel_start:voxel_end] += (sampled * visible.unsqueeze(1)).sum(dim=0).transpose(0, 1)
            sam3d_visible_count[voxel_start:voxel_end] += visible.sum(dim=0)
            del sampled, visible

        del source_chunk, affine_chunk, extrinsic_chunk, intrinsic_chunk
        del grid, images, tokens, feature_chunk, uv_ndc, linear_depth
        del valid_projection, pixel_x, pixel_y, pixel_index
        del surface_depth, reference_depth, visibility

    trellis1_features = trellis1_feature_sum.div_(num_views).cpu()
    sam3d_features = sam3d_feature_sum.div_(sam3d_visible_count.clamp_min_(1).unsqueeze(1)).cpu()
    return voxels.cpu(), trellis1_features, sam3d_features


@torch.no_grad()
def trellis1_slat_encode(
    voxels: torch.Tensor,
    features: torch.Tensor,
    encoder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode TRELLIS1 sparse features and return its latent coordinates."""
    device = next(encoder.parameters()).device
    voxels = voxels.to(device=device, dtype=torch.int32)
    coords = torch.cat((torch.zeros_like(voxels[:, :1]), voxels), dim=1)
    sparse_input = trellis_sp.SparseTensor(
        feats=features.to(device=device, dtype=torch.float32),
        coords=coords,
    )
    latent = encoder(sparse_input, sample_posterior=False)
    if not torch.isfinite(latent.feats).all():
        raise ValueError("non-finite TRELLIS1 latent features")
    return (
        latent.coords[:, 1:].to(dtype=torch.uint8).cpu(),
        latent.feats.to(dtype=torch.float32).cpu(),
    )


@torch.no_grad()
def sam3d_slat_encode(
    voxels: torch.Tensor,
    features: torch.Tensor,
    encoder,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode SAM3D sparse features and return its latent coordinates."""
    device = next(encoder.parameters()).device
    voxels = voxels.to(device=device, dtype=torch.int32)
    coords = torch.cat((torch.zeros_like(voxels[:, :1]), voxels), dim=1)
    sparse_input = Sam3DSparseTensor(
        feats=features.to(device=device, dtype=torch.float32),
        coords=coords,
    )
    latent = encoder(sparse_input, sample_posterior=False)
    if not torch.isfinite(latent.feats).all():
        raise ValueError("non-finite SAM3D latent features")
    return (
        latent.coords[:, 1:].to(dtype=torch.uint8).cpu(),
        latent.feats.to(dtype=torch.float32).cpu(),
    )


@torch.no_grad()
def encoder(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    transform: torch.Tensor,
    source_images: torch.Tensor,
    affines: torch.Tensor,
    target_extrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    dino_model,
    trellis1_encoder,
    sam3d_encoder,
    device: torch.device,
) -> Dict[str, object]:
    """Aggregate DINO features and encode both sparse latent formats."""
    try:
        dino_model.to(device)
        voxels, trellis1_features, sam3d_features = dino_aggregate(
            vertices,
            faces,
            source_images,
            affines,
            target_extrinsics,
            target_intrinsics,
            dino_model,
        )
    finally:
        dino_model.cpu()

    try:
        trellis1_encoder.to(device)
        trellis1_coords, trellis1_latent = trellis1_slat_encode(
            voxels,
            trellis1_features,
            trellis1_encoder,
        )
    finally:
        trellis1_encoder.cpu()

    try:
        sam3d_encoder.to(device)
        sam3d_coords, sam3d_latent = sam3d_slat_encode(
            voxels,
            sam3d_features,
            sam3d_encoder,
        )
    finally:
        sam3d_encoder.cpu()

    return {
        "transform": transform.to(dtype=torch.float32).cpu(),
        "trellis1_coords": trellis1_coords,
        "trellis1_latent": trellis1_latent,
        "sam3d_coords": sam3d_coords,
        "sam3d_latent": sam3d_latent,
    }


def saver(
    sha256: str,
    entry_name: List[str],
    transform: List[torch.Tensor],
    trellis1_coords: List[torch.Tensor],
    trellis1_latent: List[torch.Tensor],
    sam3d_coords: List[torch.Tensor],
    sam3d_latent: List[torch.Tensor],
    trellis1_latent_files: DatasetFiles,
    sam3d_latent_files: DatasetFiles,
) -> Dict[str, object]:
    """Publish one shape as separate TRELLIS1 and SAM3D latent archives."""
    trellis1_destination = trellis1_latent_files.path(sha256)
    sam3d_destination = sam3d_latent_files.path(sha256)
    trellis1_temporary = trellis1_destination.parent / (f".{trellis1_destination.stem}.tmpfile")
    sam3d_temporary = sam3d_destination.parent / (f".{sam3d_destination.stem}.tmpfile")

    try:
        with (
            zipfile.ZipFile(trellis1_temporary, "w", compression=zipfile.ZIP_DEFLATED) as trellis1_archive,
            zipfile.ZipFile(sam3d_temporary, "w", compression=zipfile.ZIP_DEFLATED) as sam3d_archive,
        ):
            for index, name in enumerate(entry_name):
                trellis1_buffer = io.BytesIO()
                np.savez_compressed(
                    trellis1_buffer,
                    coords=trellis1_coords[index].numpy(),
                    feats=trellis1_latent[index].numpy(),
                    transform=transform[index].numpy(),
                )
                trellis1_archive.writestr(name, trellis1_buffer.getbuffer())

                sam3d_buffer = io.BytesIO()
                np.savez_compressed(
                    sam3d_buffer,
                    coords=sam3d_coords[index].numpy(),
                    feats=sam3d_latent[index].numpy(),
                    transform=transform[index].numpy(),
                )
                sam3d_archive.writestr(name, sam3d_buffer.getbuffer())

        trellis1_temporary.replace(trellis1_destination)
        sam3d_temporary.replace(sam3d_destination)
    finally:
        if trellis1_temporary.exists():
            trellis1_temporary.unlink()
        if sam3d_temporary.exists():
            sam3d_temporary.unlink()

    return {
        "sha256": sha256,
        trellis1_latent_files.rel_path: True,
        sam3d_latent_files.rel_path: True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract multi-pose TRELLIS1 and SAM3D sparse latents")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-scale", type=int, default=1)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--num-rots", type=int, default=16)
    parser.add_argument("--num-perts", type=int, default=4)
    parser.add_argument("--perturbation-rad-std", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    parser.add_argument("--trellis1-encoder", default="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16")
    parser.add_argument("--sam3d-encoder", default="facebook/sam-3d-objects/checkpoints/slat_encoder")
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    if args.num_scale <= 0 or args.num_rots <= 0 or args.num_perts <= 0:
        parser.error("pose sample counts must be positive")
    if not 0 < args.min_scale <= 1:
        parser.error("--min-scale must be in (0, 1]")
    if args.perturbation_rad_std < 0:
        parser.error("--perturbation-rad-std must be non-negative")
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--rank must be in [0, --world-size)")

    sam3d_parts = args.sam3d_encoder.split("/", 2)
    if len(sam3d_parts) != 3:
        parser.error("--sam3d-encoder must contain a Hugging Face repository and model path")
    sam3d_repo = "/".join(sam3d_parts[:2])
    sam3d_model = sam3d_parts[2]

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    if trellis_sp.ATTN != "flash_attn" or sam3d_sp.ATTN != "flash_attn":
        raise RuntimeError("TRELLIS1 and SAM3D sparse modules must be imported with flash_attn")

    dino_model = torch.hub.load("facebookresearch/dinov2", args.dino_model).eval().cpu()
    trellis1_encoder = trellis1_models.from_pretrained(args.trellis1_encoder).eval().cpu()

    sam3d_config_path = hf_hub_download(sam3d_repo, f"{sam3d_model}.yaml")
    sam3d_checkpoint_path = hf_hub_download(sam3d_repo, f"{sam3d_model}.ckpt")
    sam3d_encoder = instantiate(OmegaConf.load(sam3d_config_path))
    sam3d_encoder.load_state_dict(torch.load(sam3d_checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    sam3d_encoder.eval().cpu()

    sampling_name = "_".join(
        [
            f"s{args.num_scale}",
            f"min{args.min_scale}",
            f"r{args.num_rots}",
            f"p{args.num_perts}",
            f"std{args.perturbation_rad_std}",
            f"seed{args.seed}",
        ]
    )
    trellis1_name = "_".join(
        [
            args.dino_model,
            args.trellis1_encoder.split("/")[-1],
            sampling_name,
        ]
    )
    sam3d_name = "_".join(
        [
            args.dino_model,
            sam3d_model.split("/")[-1],
            sampling_name,
        ]
    )
    trellis1_rel_path = f"trellis1/multi_slats/{trellis1_name}"
    sam3d_rel_path = f"sam3d/multi_slats/{sam3d_name}"

    workspace = DatasetWorkspace(args.dataset_dir)
    shape_files = workspace.files("trellis1/normalized_shape", ".ply")
    render_files = workspace.files("trellis1/renders", ".zip")
    camera_files = workspace.files("trellis1/cameras", ".json")
    trellis1_latent_files = workspace.files(trellis1_rel_path, ".zip")
    sam3d_latent_files = workspace.files(sam3d_rel_path, ".zip")
    trellis1_latent_files.mkdir()
    sam3d_latent_files.mkdir()

    metadata = workspace.read_metadata()
    output_columns = [
        trellis1_latent_files.rel_path,
        sam3d_latent_files.rel_path,
    ]
    for column in output_columns:
        if column not in metadata.columns:
            metadata[column] = False

    ready = metadata[shape_files.rel_path].eq(True) & metadata[render_files.rel_path].eq(True) & metadata[camera_files.rel_path].eq(True)
    if not args.recompute_finished:
        completed = metadata[output_columns].eq(True).all(axis=1)
        ready &= ~completed
    sha256s = metadata.loc[ready, "sha256"].astype(str).drop_duplicates()

    start = len(sha256s) * args.rank // args.world_size
    end = len(sha256s) * (args.rank + 1) // args.world_size
    selected = sha256s.iloc[start:end]
    if selected.empty:
        return

    inputs = ({"sha256": str(value)} for value in selected)
    stages = [
        Stage(
            "sample mesh SRT",
            srt_sampler,
            queue_size=1,
            work_queue_size=1,
            params={
                "num_scale": args.num_scale,
                "min_scale": args.min_scale,
                "num_rots": args.num_rots,
                "num_perts": args.num_perts,
                "perturbation_rad_std": args.perturbation_rad_std,
                "seed": args.seed,
            },
            resources={"shape_files": shape_files, "device": device},
        ),
        Stage(
            "load shared renders",
            loader,
            queue_size=1,
            work_queue_size=1,
            resources={
                "render_files": render_files,
                "camera_files": camera_files,
                "device": device,
            },
        ),
        Stage(
            "prepare pose images",
            affine_worker,
            queue_size=1,
            work_queue_size=1,
            resources={"device": device},
        ),
        Stage(
            "encode pose latents",
            encoder,
            queue_size=1,
            work_queue_size=1,
            resources={
                "dino_model": dino_model,
                "trellis1_encoder": trellis1_encoder,
                "sam3d_encoder": sam3d_encoder,
                "device": device,
            },
        ),
        Stage(
            "save pose latents",
            saver,
            mode="group",
            queue_size=0,
            work_queue_size=1,
            group_key="sha256",
            group_size="num_poses",
            group_position="pose_index",
            group_timeout_s=3600.0,
            resources={
                "trellis1_latent_files": trellis1_latent_files,
                "sam3d_latent_files": sam3d_latent_files,
            },
        ),
    ]
    results = Pipeline(
        stages,
        max_active_roots=1,
        total=len(selected),
    ).run(inputs)

    records = pd.DataFrame(
        results,
        columns=["sha256", trellis1_rel_path, sam3d_rel_path],
    ).drop_duplicates(subset="sha256", keep="first")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/trellis1_multi_pose_feature_extract_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
