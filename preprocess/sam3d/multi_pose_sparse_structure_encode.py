"""Encode transformed raw meshes as SAM3D sparse-structure latent archives."""

import argparse
import hashlib
import io
import zipfile
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import trimesh
from huggingface_hub import hf_hub_download
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dataset.base import format_entry_name
from preprocess.dataset.base import DatasetFiles, DatasetWorkspace
from preprocess.utils import Pipeline, Stage, mesh_to_voxel_coords, sample_mesh_srt

SPARSE_STRUCTURE_RESOLUTION = 64
GLTF_TO_BLENDER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


@torch.no_grad()
def loader_srt_sampler(
    sha256: str,
    raw_files: DatasetFiles,
    device: torch.device,
    num_scale: int,
    min_scale: float,
    num_rots: int,
    num_perts: int,
    perturbation_rad_std: float,
    seed: int,
) -> List[Dict[str, object]]:
    """Load one raw mesh and emit its sampled scale-rotation-translation copies."""
    raw_path = raw_files.find(sha256)
    if raw_path is None:
        raise FileNotFoundError(f"raw mesh not found: {sha256}")

    mesh = trimesh.load_scene(raw_path, process=False).to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"raw file does not contain a triangle mesh: {raw_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"empty raw mesh: {sha256}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"non-finite vertices: {sha256}")
    if raw_path.suffix.lower() in {".glb", ".gltf"}:
        vertices = vertices @ GLTF_TO_BLENDER.T

    vertices_tensor = torch.from_numpy(vertices).to(
        device=device,
        dtype=torch.float32,
    )
    faces_tensor = torch.from_numpy(faces).to(dtype=torch.int64).contiguous()
    shape_seed = int.from_bytes(
        hashlib.sha256(f"{sha256}:{seed}".encode("ascii")).digest()[:8],
        "big",
    )
    transformed_vertices, transforms, _, _ = sample_mesh_srt(
        vertices_tensor,
        num_scale,
        min_scale,
        num_rots,
        num_perts,
        perturbation_rad_std,
        SPARSE_STRUCTURE_RESOLUTION,
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
def voxelize_worker(
    verts: torch.Tensor,
    faces: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    """Convert one transformed mesh into sparse occupancy coordinates."""
    coords = mesh_to_voxel_coords(
        verts.to(device=device, dtype=torch.float32),
        faces.to(device=device, dtype=torch.int64),
        SPARSE_STRUCTURE_RESOLUTION,
    )
    return {
        "verts": None,
        "faces": None,
        "shape_latent_coords": coords.cpu(),
    }


@torch.no_grad()
def ss_latent_encode_worker(
    shape_latent_coords: torch.Tensor,
    device: torch.device,
    encoder,
) -> Dict[str, object]:
    """Encode sparse occupancy into one deterministic SAM3D latent."""
    coords = shape_latent_coords.to(device=device, dtype=torch.int64)
    sparse_structure = torch.zeros(
        1,
        SPARSE_STRUCTURE_RESOLUTION,
        SPARSE_STRUCTURE_RESOLUTION,
        SPARSE_STRUCTURE_RESOLUTION,
        dtype=torch.float32,
        device=device,
    )
    sparse_structure[0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
    ss_latent = encoder(sparse_structure[None])["z"][0]
    if tuple(ss_latent.shape) != (8, 16, 16, 16):
        raise ValueError(f"unexpected SAM3D sparse-structure latent shape: {tuple(ss_latent.shape)}")

    return {
        "shape_latent_coords": None,
        "ss_latent": ss_latent.detach().to(dtype=torch.float32).cpu(),
    }


def saver(
    sha256: str,
    file_name: List[str],
    ss_latent: List[torch.Tensor],
    trans: List[np.ndarray],
    ss_latent_files: DatasetFiles,
) -> Dict[str, object]:
    """Save one shape's ordered latent samples in the shared ZIP format."""
    destination = ss_latent_files.path(sha256)
    temporary = destination.parent / f".{destination.stem}.tmpfile"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, latent, transform in zip(file_name, ss_latent, trans):
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                feats=latent.numpy(),
                transform=transform,
            )
            archive.writestr(name, buffer.getbuffer())
    temporary.replace(destination)

    return {
        "sha256": sha256,
        ss_latent_files.rel_path: True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode SAM3D sparse-structure latents")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-scale", type=int, default=1)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--num-rots", type=int, default=16)
    parser.add_argument("--num-perts", type=int, default=4)
    parser.add_argument("--perturbation-rad-std", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument("--ss-encoder", default="facebook/sam-3d-objects/checkpoints/ss_encoder")
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

    encoder_parts = args.ss_encoder.split("/", 2)
    if len(encoder_parts) != 3:
        parser.error("--ss-encoder must contain a Hugging Face repository and model path")
    encoder_repo = "/".join(encoder_parts[:2])
    encoder_model = encoder_parts[2]

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    encoder_config_path = hf_hub_download(encoder_repo, f"{encoder_model}.yaml")
    encoder_checkpoint_path = hf_hub_download(encoder_repo, f"{encoder_model}.ckpt")
    encoder_config = OmegaConf.load(encoder_config_path)
    encoder_config.sample_posterior = False
    encoder_config.return_raw = True
    encoder = instantiate(encoder_config)
    encoder.load_state_dict(
        torch.load(encoder_checkpoint_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    encoder.eval().to(device)

    feature_name = "_".join(
        [
            encoder_model.split("/")[-1],
            "sslatentres_16",
            f"occres_{SPARSE_STRUCTURE_RESOLUTION}",
            f"s{args.num_scale}",
            f"min{args.min_scale}",
            f"r{args.num_rots}",
            f"p{args.num_perts}",
            f"std{args.perturbation_rad_std}",
            f"seed{args.seed}",
        ]
    )
    ss_rel_path = f"sam3d/multi_ss_latents/{feature_name}"

    workspace = DatasetWorkspace(args.dataset_dir)
    raw_files = workspace.files("raw", "")
    ss_latent_files = workspace.files(ss_rel_path, ".zip")
    ss_latent_files.mkdir()

    metadata = workspace.read_metadata()
    if "raw" not in metadata:
        raise ValueError("metadata does not contain the required 'raw' column")
    ready = metadata["raw"].eq(True)
    if not args.recompute_finished and ss_rel_path in metadata:
        ready &= ~metadata[ss_rel_path].eq(True)
    sha256s = metadata.loc[ready, "sha256"].astype(str).drop_duplicates()

    start = len(sha256s) * args.rank // args.world_size
    end = len(sha256s) * (args.rank + 1) // args.world_size
    selected = sha256s.iloc[start:end]
    if selected.empty:
        return

    inputs = ({"sha256": value} for value in selected)
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
                "seed": args.seed,
            },
            resources={"raw_files": raw_files, "device": device},
        ),
        Stage(
            "voxelize sparse structure",
            voxelize_worker,
            queue_size=1,
            work_queue_size=1,
            resources={"device": device},
        ),
        Stage(
            "encode sparse structure",
            ss_latent_encode_worker,
            queue_size=1,
            work_queue_size=1,
            resources={"device": device, "encoder": encoder},
        ),
        Stage(
            "save sparse structure",
            saver,
            mode="group",
            queue_size=0,
            work_queue_size=1,
            group_key="sha256",
            group_size="group_size",
            group_position="group_position",
            group_timeout_s=3600.0,
            resources={"ss_latent_files": ss_latent_files},
        ),
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(
        results,
        columns=["sha256", ss_rel_path],
    ).drop_duplicates(subset="sha256", keep="first")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(
        f"unmerged_records/sam3d_multi_pose_sparse_structure_encode_{timestamp}_rank{args.rank}.csv",
    )
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
