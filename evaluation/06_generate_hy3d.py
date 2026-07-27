import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from PIL import Image

from preprocess.utils import Pipeline, Stage

from .base import Workspace

HY3D_MODEL_PATH = "tencent/Hunyuan3D-2.1"
RENDER_FORMAT = "{shape_id:06d}/{view_id:03d}"


@torch.no_grad()
def generate_hy3d(
    idx,
    shape_id,
    view_id,
    render_files,
    output_files,
    pipeline,
    background_remover,
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
        with Image.open(render_path) as image_file:
            image = image_file.copy()

        if image.mode != "RGBA":
            image = background_remover(image.convert("RGB")).convert("RGBA")
        else:
            image = image.convert("RGBA")

        mesh = pipeline(image=image)[0]
        vertices = np.asarray(mesh.vertices).astype(np.float32).copy()
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        scale = np.max(bounds_max - bounds_min)
        vertices = vertices - center
        if scale > 0:
            vertices = vertices / scale

        mesh.vertices = vertices
        mesh.export(output_path)
    except Exception as error:
        if output_path.exists():
            output_path.unlink()
        fail_path.write_text(f"Hunyuan3D generation failed: {error}\n", encoding="utf-8")
        return {"idx": idx, output_files.rel_path: True}

    if fail_path.exists():
        fail_path.unlink()
    return {"idx": idx, output_files.rel_path: True}


def main():
    parser = argparse.ArgumentParser(description="Generate Hunyuan3D baseline shapes")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    render_files = workspace.files("renders", format=RENDER_FORMAT)
    output_files = workspace.files("experiments/hy3d")
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
    torch.cuda.reset_peak_memory_stats(device)
    background_remover = BackgroundRemover()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HY3D_MODEL_PATH,
        device=device,
        dtype=torch.float16,
    )

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
            "generate Hunyuan3D shapes",
            generate_hy3d,
            resources={
                "render_files": render_files,
                "output_files": output_files,
                "pipeline": pipeline,
                "background_remover": background_remover,
            },
        )
    ]
    results = Pipeline(stages, total=len(selected)).run(inputs)

    records = pd.DataFrame(results, columns=["idx", output_files.rel_path])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/06_generate_hy3d_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
