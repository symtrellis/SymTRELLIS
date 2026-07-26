import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

from inference.trellis2 import preprocess_image
from preprocess.utils import Pipeline, Stage

from .base import Files, Workspace

IMAGE_CONDITION_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
IMAGE_SIZE = 512
RENDER_FORMAT = "{shape_id:06d}/{view_id:03d}"


def load_image(
    shape_id: int,
    view_id: int,
    render_files: Files,
) -> dict[str, object]:
    render_path = render_files.path(".png", shape_id=shape_id, view_id=view_id)
    with Image.open(render_path) as image_file:
        source_image = image_file.copy()

    processed_image = preprocess_image(source_image, rembg_model=None, target_size=IMAGE_SIZE)
    return {"processed_image": processed_image}


@torch.no_grad()
def extract_condition(
    processed_image: list[Image.Image],
    image_condition_model: DinoV3FeatureExtractor,
) -> list[dict[str, object]]:
    conditions = image_condition_model(processed_image).to(device="cpu", dtype=torch.float32)
    return [{"processed_image": None, "condition": condition.unsqueeze(0)} for condition in conditions]


def save_condition(
    shape_id: int,
    view_id: int,
    condition: torch.Tensor,
    condition_files: Files,
) -> dict[str, object]:
    destination = condition_files.path(".npz", shape_id=shape_id, view_id=view_id)
    np.savez_compressed(destination, cond=condition.numpy())

    return {"condition": None, condition_files.rel_path: condition_files.exists(shape_id=shape_id, view_id=view_id)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRELLIS.2 image conditions")
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--output-dir", default="conditions")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--load-workers", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=4)
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    render_files = workspace.files("renders", format=RENDER_FORMAT)
    condition_files = workspace.files(args.output_dir)
    condition_files.mkdir()

    metadata = workspace.read_metadata().sort_values("idx")
    if not args.recompute_finished:
        finished = [condition_files.exists(shape_id=row["shape_id"], view_id=row["view_id"]) for _, row in metadata.iterrows()]
        metadata = metadata.loc[[not value for value in finished]]

    start = len(metadata) * args.rank // args.world_size
    end = len(metadata) * (args.rank + 1) // args.world_size
    selected = metadata.iloc[start:end]
    if selected.empty:
        return

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    image_condition_model = DinoV3FeatureExtractor(
        IMAGE_CONDITION_MODEL,
        image_size=IMAGE_SIZE,
    )
    image_condition_model.to(device)

    inputs = ({"idx": row["idx"], "shape_id": row["shape_id"], "view_id": row["view_id"]} for _, row in selected.iterrows())
    stages = [
        Stage(
            "load images",
            load_image,
            workers=args.load_workers,
            queue_size=2 * args.load_workers,
            work_queue_size=args.load_workers,
            resources={"render_files": render_files},
        ),
        Stage(
            "extract conditions",
            extract_condition,
            mode="batch",
            workers=1,
            queue_size=2 * args.batch_size,
            work_queue_size=1,
            batch_size=args.batch_size,
            resources={"image_condition_model": image_condition_model},
        ),
        Stage(
            "save conditions",
            save_condition,
            workers=args.save_workers,
            queue_size=2 * args.save_workers,
            work_queue_size=args.save_workers,
            resources={"condition_files": condition_files},
        ),
    ]
    results = Pipeline(
        stages,
        total=len(selected),
    ).run(inputs)

    records = pd.DataFrame(
        results,
        columns=["idx", condition_files.rel_path],
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/01_generate_image_condition_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)

    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(f"peak allocated: {peak_allocated:.1f} MiB")
    print(f"peak reserved:  {peak_reserved:.1f} MiB")


if __name__ == "__main__":
    main()
