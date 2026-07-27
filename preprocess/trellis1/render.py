"""Render shared TRELLIS.1 and SAM3D image-conditioning assets."""

import argparse
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path
from shutil import copy2
from subprocess import DEVNULL, PIPE, STDOUT, CalledProcessError, run
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import pandas as pd

from ..dataset.base import DatasetFiles, DatasetWorkspace
from ..utils import Pipeline, Stage, fibonacci_sphere_samples, install_blender

RENDER_REL_PATH = "trellis1/renders"
CAMERA_REL_PATH = "trellis1/cameras"
SHAPE_REL_PATH = "trellis1/normalized_shape"
BLENDER_SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "trellis1_sam3d_render.py"


def render(
    sha256: str,
    raw_path: Optional[Path],
    blender_path: Path,
    blender_script: Path,
    views_json: str,
    resolution: int,
    verbose: bool,
) -> Dict[str, object]:
    if raw_path is None:
        raise FileNotFoundError(f"raw artifact not found for sha256 {sha256!r}")

    temporary = TemporaryDirectory(dir="/dev/shm")
    command = [str(blender_path)]
    if str(raw_path).endswith(".blend"):
        command.append(str(raw_path))
    command.extend(
        [
            "-b",
            "--python-exit-code",
            "1",
            "-P",
            str(blender_script),
            "--",
            "--views",
            views_json,
            "--object",
            str(raw_path),
            "--resolution",
            str(resolution),
            "--output-folder",
            temporary.name,
        ]
    )

    try:
        run(
            command,
            check=True,
            stdin=DEVNULL,
            stdout=None if verbose else PIPE,
            stderr=None if verbose else STDOUT,
            text=True,
        )
    except (CalledProcessError, OSError) as error:
        temporary.cleanup()
        output = getattr(error, "stdout", None)
        message = f"Blender render failed for sha256 {sha256!r}: {error}"
        if output:
            message = f"{message}\n{output.rstrip()}"
        raise RuntimeError(message) from error

    return {"temporary": temporary}


def save(
    sha256: str,
    temporary: TemporaryDirectory,
    num_views: int,
    render_files: DatasetFiles,
    camera_files: DatasetFiles,
    shape_files: DatasetFiles,
) -> Dict[str, object]:
    work_dir = Path(temporary.name)
    try:
        images = sorted(work_dir.glob("*.png"))
        expected_names = [f"{index:03d}.png" for index in range(num_views)]
        image_names = [image.name for image in images]
        if image_names != expected_names:
            raise RuntimeError(f"Blender produced unexpected images for sha256 {sha256!r}: " f"expected {expected_names!r}, got {image_names!r}")

        camera_source = work_dir / "transforms.json"
        shape_source = work_dir / "mesh.ply"
        if not camera_source.is_file():
            raise FileNotFoundError(f"Blender did not produce transforms.json for sha256 {sha256!r}")
        if not shape_source.is_file():
            raise FileNotFoundError(f"Blender did not produce mesh.ply for sha256 {sha256!r}")

        render_destination = render_files.path(sha256)
        camera_destination = camera_files.path(sha256)
        shape_destination = shape_files.path(sha256)
        with (
            TemporaryDirectory(dir=render_destination.parent) as render_publish_dir,
            TemporaryDirectory(dir=camera_destination.parent) as camera_publish_dir,
            TemporaryDirectory(dir=shape_destination.parent) as shape_publish_dir,
        ):
            staged_render = Path(render_publish_dir) / render_destination.name
            staged_camera = Path(camera_publish_dir) / camera_destination.name
            staged_shape = Path(shape_publish_dir) / shape_destination.name

            with zipfile.ZipFile(
                staged_render,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as archive:
                for image in images:
                    archive.write(image, arcname=image.name)
            copy2(camera_source, staged_camera)
            copy2(shape_source, staged_shape)

            if not staged_render.is_file() or not staged_camera.is_file() or not staged_shape.is_file():
                raise RuntimeError(f"failed to stage rendered artifacts for sha256 {sha256!r}")

            staged_render.replace(render_destination)
            staged_camera.replace(camera_destination)
            staged_shape.replace(shape_destination)

        return {
            "sha256": sha256,
            RENDER_REL_PATH: True,
            CAMERA_REL_PATH: True,
            SHAPE_REL_PATH: True,
        }
    finally:
        temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TRELLIS.1 and SAM3D conditioning assets")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--num-views", type=int, default=225)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    if args.num_views <= 0:
        parser.error("--num-views must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.world_size <= 0:
        parser.error("--world-size must be positive")
    if args.rank < 0 or args.rank >= args.world_size:
        parser.error("--rank must be in [0, --world-size)")

    resolution = 1472
    radius = 2.0
    render_fov = 2 * math.asin(0.5 * math.sqrt(3) / radius)

    directions = fibonacci_sphere_samples(args.num_views)
    views = []
    for direction in directions:
        views.append(
            {
                "yaw": math.atan2(float(direction[1]), float(direction[0])),
                "pitch": math.asin(float(direction[2])),
                "radius": radius,
                "fov": render_fov,
            }
        )
    views_json = json.dumps(views)

    workspace = DatasetWorkspace(args.dataset_dir)
    raw_files = workspace.files("raw", "")
    render_files = workspace.files(RENDER_REL_PATH, ".zip")
    camera_files = workspace.files(CAMERA_REL_PATH, ".json")
    shape_files = workspace.files(SHAPE_REL_PATH, ".ply")
    render_files.mkdir()
    camera_files.mkdir()
    shape_files.mkdir()

    metadata = workspace.read_metadata()
    output_columns = [
        render_files.rel_path,
        camera_files.rel_path,
        shape_files.rel_path,
    ]
    for column in output_columns:
        if column not in metadata.columns:
            metadata[column] = False

    ready = metadata[raw_files.rel_path].eq(True)
    if not args.recompute_finished:
        completed = metadata[output_columns].eq(True).all(axis=1)
        ready &= ~completed
    sha256s = metadata.loc[ready, "sha256"].astype(str).drop_duplicates()

    start = len(sha256s) * args.rank // args.world_size
    end = len(sha256s) * (args.rank + 1) // args.world_size
    selected = sha256s.iloc[start:end]
    if selected.empty:
        return
    blender_path = install_blender()

    inputs = (
        {
            "sha256": str(value),
            "raw_path": raw_files.find(str(value)),
        }
        for value in selected
    )
    stages = [
        Stage(
            "render",
            render,
            workers=args.num_workers,
            queue_size=args.num_workers,
            work_queue_size=args.num_workers,
            params={
                "blender_path": blender_path,
                "blender_script": BLENDER_SCRIPT,
                "views_json": views_json,
                "resolution": resolution,
                "verbose": args.verbose,
            },
        ),
        Stage(
            "save",
            save,
            workers=1,
            queue_size=1,
            work_queue_size=1,
            params={"num_views": args.num_views},
            resources={
                "render_files": render_files,
                "camera_files": camera_files,
                "shape_files": shape_files,
            },
        ),
    ]
    results = Pipeline(
        stages,
        max_active_roots=args.num_workers + 1,
        total=len(selected),
    ).run(inputs)

    records = pd.DataFrame(
        results,
        columns=[
            "sha256",
            RENDER_REL_PATH,
            CAMERA_REL_PATH,
            SHAPE_REL_PATH,
        ],
    ).drop_duplicates(subset="sha256", keep="first")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/trellis1_render_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)


if __name__ == "__main__":
    main()
