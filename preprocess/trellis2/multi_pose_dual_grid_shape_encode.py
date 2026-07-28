"""Encode transformed meshes as TRELLIS2 flexible-dual-grid latent archives."""

import argparse
import hashlib
import io
import pickle
import zipfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import trellis2.models as trellis2_models
from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder
from trellis2.modules.sparse.basic import SparseTensor

from dataset.base import format_entry_name
from preprocess.dataset.base import DatasetFiles, DatasetWorkspace
from preprocess.utils import Pipeline, Stage, sample_mesh_srt

MESH_REL_PATH = "trellis2/dumped_mesh"


def loader_srt_sampler(
    sha256: str,
    mesh_files: DatasetFiles,
    device: torch.device,
    num_scale: int,
    min_scale: float,
    num_rots: int,
    num_perts: int,
    perturbation_rad_std: float,
    shape_latent_resolution: int,
    seed: int,
) -> List[Dict[str, object]]:
    """Load one mesh and emit its sampled scale-rotation-translation copies."""
    with mesh_files.path(sha256).open("rb") as file:
        dump = pickle.load(file)

    start = 0
    vertices = []
    faces = []
    for obj in dump["objects"]:
        if obj["vertices"].size == 0 or obj["faces"].size == 0:
            continue
        vertices.append(obj["vertices"])
        faces.append(obj["faces"] + start)
        start += len(obj["vertices"])

    vertices_tensor = torch.from_numpy(np.concatenate(vertices, axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    faces_tensor = torch.from_numpy(np.concatenate(faces, axis=0)).to(
        dtype=torch.int64,
    )
    faces_tensor = faces_tensor.contiguous().cpu()
    if not torch.isfinite(vertices_tensor).all():
        raise ValueError(f"non-finite verts: {sha256}")

    shape_seed = int.from_bytes(hashlib.sha256(f"{sha256}:{seed}".encode("ascii")).digest()[:8], "big")
    transformed_vertices, transforms, _, _ = sample_mesh_srt(
        vertices_tensor,
        num_scale,
        min_scale,
        num_rots,
        num_perts,
        perturbation_rad_std,
        shape_latent_resolution,
        shape_seed,
    )
    group_size = num_scale * num_rots * num_perts

    return [
        {
            "sha256": sha256,
            "verts": transformed_vertices[scale_idx, rot_idx, pert_idx].clone().contiguous().cpu(),
            "faces": faces_tensor,
            "trans": transforms[scale_idx, rot_idx, pert_idx].detach().clone().cpu().numpy(),
            "group_size": group_size,
            "group_position": scale_idx * num_rots * num_perts + rot_idx * num_perts + pert_idx,
            "file_name": format_entry_name(scale_idx, rot_idx, pert_idx),
        }
        for scale_idx in range(num_scale)
        for rot_idx in range(num_rots)
        for pert_idx in range(num_perts)
    ]


@torch.no_grad()
def dual_grid_converter(
    verts: torch.Tensor,
    faces: torch.Tensor,
    shape_latent_resolution: int,
    device: torch.device,
) -> Dict[str, object]:
    """Convert one transformed mesh into a flexible dual grid."""
    resolution = 16 * shape_latent_resolution
    voxel_coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        vertices=verts.to(device),
        faces=faces.to(device),
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False,
    )

    dual_vertices = dual_vertices * resolution - voxel_coords
    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), "dual_vertices out of range"
    dual_vertices = torch.clamp(dual_vertices, 0, 1)

    return {
        "verts": None,
        "faces": None,
        "voxel_coords": voxel_coords.cpu(),
        "dual_vertices": dual_vertices.cpu(),
        "intersected": intersected.cpu(),
    }


@torch.no_grad()
def shape_latent_encode_worker(
    voxel_coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected: torch.Tensor,
    device: torch.device,
    encoder: FlexiDualGridVaeEncoder,
) -> Dict[str, object]:
    """Encode one flexible dual grid as a TRELLIS2 sparse latent."""
    voxel_coords = voxel_coords.to(device)
    dual_vertices = dual_vertices.to(device)
    intersected = intersected.to(device)

    # Construct the corresponding o-voxel vertices and intersection features.
    coords = torch.cat(
        [
            torch.zeros_like(
                voxel_coords[:, :1],
                dtype=torch.int32,
                device=device,
            ),
            voxel_coords.to(dtype=torch.int32),
        ],
        dim=1,
    )
    vertices_sp = SparseTensor(
        feats=dual_vertices.to(dtype=torch.float32),
        coords=coords,
    )
    intersected_sp = SparseTensor(
        feats=intersected.to(dtype=torch.bool),
        coords=coords,
    )

    shape_latent = encoder(vertices_sp, intersected_sp)
    torch.cuda.empty_cache()

    return {
        "voxel_coords": None,
        "dual_vertices": None,
        "intersected": None,
        "shape_latent_coords": shape_latent.coords[:, 1:].detach().cpu(),
        "shape_latent_feats": shape_latent.feats.detach().cpu(),
    }


def saver(
    sha256: str,
    file_name: List[str],
    shape_latent_feats: List[torch.Tensor],
    shape_latent_coords: List[torch.Tensor],
    trans: List[np.ndarray],
    remove_pickle: bool,
    mesh_files: DatasetFiles,
    shape_latent_files: DatasetFiles,
) -> Dict[str, object]:
    """Save one shape's ordered latent samples in the legacy ZIP format."""
    names = []
    buffers = []
    for name, feats, coords, transform in zip(file_name, shape_latent_feats, shape_latent_coords, trans):
        names.append(name)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            feats=feats.cpu().numpy(),
            coords=coords.cpu().numpy(),
            transform=transform,
        )
        buffers.append(buffer)

    destination = shape_latent_files.path(sha256)
    temporary = destination.parent / f".{destination.stem}.tmpfile"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, buffer in zip(names, buffers):
            archive.writestr(name, buffer.getbuffer())
    temporary.replace(destination)

    if remove_pickle:
        mesh_files.path(sha256).unlink(missing_ok=True)

    return {
        "sha256": sha256,
        MESH_REL_PATH: mesh_files.exists(sha256),
        shape_latent_files.rel_path: True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode TRELLIS2 flexible-dual-grid shape latents")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-scale", type=int, default=3)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--num-rots", type=int, default=6)
    parser.add_argument("--num-perts", type=int, default=3)
    parser.add_argument("--shape-latent-resolution", type=int, default=32)
    parser.add_argument("--perturbation-rad-std", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--shape-encoder", default="microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16")
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--remove-pickle", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    encoder = trellis2_models.from_pretrained(args.shape_encoder).eval().to(device)

    shape_latent_resolution = int(args.shape_latent_resolution)
    ovoxel_resolution = 16 * shape_latent_resolution
    encoder_name = args.shape_encoder.split("/")[-1]
    feature_name = "_".join(
        [
            encoder_name,
            f"shapelatentres_{shape_latent_resolution}",
            f"ovoxres_{ovoxel_resolution}",
            f"s{args.num_scale}",
            f"min{args.min_scale}",
            f"r{args.num_rots}",
            f"p{args.num_perts}",
            f"std{args.perturbation_rad_std}",
            f"seed{args.seed}",
        ]
    )
    shape_latent_rel_path = f"trellis2/multi_slats/{feature_name}"

    workspace = DatasetWorkspace(args.dataset_dir)
    mesh_files = workspace.files(MESH_REL_PATH, ".pickle")
    shape_latent_files = workspace.files(shape_latent_rel_path, ".zip")
    shape_latent_files.mkdir()

    metadata = workspace.read_metadata()
    if shape_latent_files.rel_path not in metadata.columns:
        metadata[shape_latent_files.rel_path] = False

    ready = metadata[mesh_files.rel_path].eq(True)
    if not args.recompute_finished:
        ready &= ~metadata[shape_latent_files.rel_path].eq(True)
    sha256s = metadata.loc[ready, "sha256"].astype(str)

    start = len(sha256s) * args.rank // args.world_size
    end = len(sha256s) * (args.rank + 1) // args.world_size
    inputs = [{"sha256": sha256} for sha256 in sha256s.iloc[start:end]]

    stages = [
        Stage(
            "sample mesh SRT",
            loader_srt_sampler,
            queue_size=1,
            work_queue_size=1,
            params={
                "num_scale": args.num_scale,
                "min_scale": args.min_scale,
                "num_rots": args.num_rots,
                "num_perts": args.num_perts,
                "perturbation_rad_std": args.perturbation_rad_std,
                "shape_latent_resolution": shape_latent_resolution,
                "seed": args.seed,
            },
            resources={"mesh_files": mesh_files, "device": device},
        ),
        Stage(
            "build flexible dual grid",
            dual_grid_converter,
            queue_size=1,
            work_queue_size=1,
            params={"shape_latent_resolution": shape_latent_resolution},
            resources={"device": device},
        ),
        Stage(
            "encode shape latent",
            shape_latent_encode_worker,
            queue_size=1,
            work_queue_size=1,
            resources={"device": device, "encoder": encoder},
        ),
        Stage(
            "save shape latent",
            saver,
            mode="group",
            queue_size=0,
            work_queue_size=1,
            group_key="sha256",
            group_size="group_size",
            group_position="group_position",
            group_timeout_s=3600.0,
            params={"remove_pickle": args.remove_pickle},
            resources={
                "mesh_files": mesh_files,
                "shape_latent_files": shape_latent_files,
            },
        ),
    ]
    results = Pipeline(stages).run(inputs)

    records = pd.DataFrame(
        results,
        columns=["sha256", MESH_REL_PATH, shape_latent_rel_path],
    ).drop_duplicates(subset="sha256", keep="first")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(
        f"unmerged_records/trellis2_multi_pose_dual_grid_shape_encode_{timestamp}_rank{args.rank}.csv",
    )
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
