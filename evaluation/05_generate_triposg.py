import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import pymeshlab
import torch
import trimesh
from briarmbg import BriaRMBG
from image_process import prepare_image
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

from preprocess.utils import Pipeline, Stage

from .base import Workspace

TRIPOSG_MODEL_PATH = "VAST-AI/TripoSG"
RMBG_MODEL_PATH = "briaai/RMBG-1.4"

TRIPOSG_NUM_INFERENCE_STEPS = 50
TRIPOSG_GUIDANCE_SCALE = 7.0
MESH_DECIMATION_TARGET = 100000
RENDER_FORMAT = "{shape_id:06d}/{view_id:03d}"


@torch.no_grad()
def generate_triposg(
    idx,
    shape_id,
    view_id,
    seed,
    render_files,
    output_files,
    pipeline,
    background_model,
):
    render_path = render_files.path(".png", shape_id=shape_id, view_id=view_id)
    output_path = output_files.path(".glb", shape_id=shape_id, view_id=view_id)
    fail_path = output_files.path(".fail", shape_id=shape_id, view_id=view_id)

    if not render_path.is_file():
        if output_path.exists():
            output_path.unlink()
        fail_path.write_text(f"Image not found: {render_path}\n", encoding="utf-8")
        return {"idx": idx, output_files.rel_path: True}

    try:
        image = prepare_image(
            str(render_path),
            bg_color=np.array([1.0, 1.0, 1.0]),
            rmbg_net=background_model,
        )
        outputs = pipeline(
            image=image,
            generator=torch.Generator(device="cuda:0").manual_seed(seed),
            num_inference_steps=TRIPOSG_NUM_INFERENCE_STEPS,
            guidance_scale=TRIPOSG_GUIDANCE_SCALE,
        ).samples[0]

        vertices = outputs[0].astype(np.float32)
        faces = np.ascontiguousarray(outputs[1])
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        scale = np.max(bounds_max - bounds_min)
        vertices = vertices - center
        if scale > 0:
            vertices = vertices / scale

        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
        mesh_set.meshing_merge_close_vertices()
        mesh_set.meshing_decimation_quadric_edge_collapse(targetfacenum=MESH_DECIMATION_TARGET)
        simplified_mesh = mesh_set.current_mesh()
        mesh = trimesh.Trimesh(
            vertices=simplified_mesh.vertex_matrix(),
            faces=simplified_mesh.face_matrix(),
            process=False,
        )
        mesh.export(output_path)
    except Exception as error:
        if output_path.exists():
            output_path.unlink()
        fail_path.write_text(f"TripoSG generation failed: {error}\n", encoding="utf-8")
        return {"idx": idx, output_files.rel_path: True}

    if fail_path.exists():
        fail_path.unlink()
    return {"idx": idx, output_files.rel_path: True}


def main():
    parser = argparse.ArgumentParser(description="Generate TripoSG baseline shapes")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    render_files = workspace.files("renders", format=RENDER_FORMAT)
    output_files = workspace.files(f"experiments/triposg_seed_{args.seed}")
    output_files.mkdir()

    metadata = workspace.read_metadata().sort_values("idx")
    if output_files.rel_path not in metadata.columns:
        metadata[output_files.rel_path] = False

    if not args.recompute_finished:
        metadata = metadata.loc[~metadata[output_files.rel_path].eq(True)]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    selected = metadata.iloc[start:end]
    if selected.empty:
        return

    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    background_model = BriaRMBG.from_pretrained(RMBG_MODEL_PATH).to(device).eval()
    pipeline = TripoSGPipeline.from_pretrained(TRIPOSG_MODEL_PATH).to(device, torch.float16)

    inputs = (
        {
            "idx": row["idx"],
            "shape_id": row["shape_id"],
            "view_id": row["view_id"],
        }
        for _, row in selected.iterrows()
    )
    stages = [
        Stage(
            "generate TripoSG shapes",
            generate_triposg,
            params={"seed": args.seed},
            resources={
                "render_files": render_files,
                "output_files": output_files,
                "pipeline": pipeline,
                "background_model": background_model,
            },
        )
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(results, columns=["idx", output_files.rel_path])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/05_generate_triposg_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
