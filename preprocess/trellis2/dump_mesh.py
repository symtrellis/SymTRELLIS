import argparse
from datetime import datetime
from pathlib import Path
from shutil import copy2
from subprocess import DEVNULL, run
from tempfile import TemporaryDirectory
from typing import Dict

import pandas as pd

from ..dataset.base import DatasetFiles, DatasetWorkspace
from ..utils import Pipeline, Stage, install_blender

MESH_REL_PATH = "trellis2/dumped_mesh"
BLENDER_SCRIPT = Path(__file__).resolve().parents[1] / "blender" / "trellis2_dump_mesh.py"


def dump_mesh(
    sha256: str,
    raw_path: Path,
    blender_path: Path,
    blender_script: Path,
    mesh_files: DatasetFiles,
    verbose: bool,
) -> Dict[str, object]:
    destination = mesh_files.path(sha256)
    with TemporaryDirectory(dir="/dev/shm") as work_dir, TemporaryDirectory(dir=destination.parent) as publish_dir:
        temporary = Path(work_dir) / "tmp_shape.pickle"
        staged = Path(publish_dir) / destination.name
        command = [
            str(blender_path),
            "-b",
            "-P",
            str(blender_script),
            "--",
            "--object",
            str(raw_path),
            "--output_path",
            str(temporary),
        ]
        if str(raw_path).endswith(".blend"):
            command.insert(1, str(raw_path))

        output = None if verbose else DEVNULL
        run(command, check=True, stdin=DEVNULL, stdout=output, stderr=output)
        copy2(temporary, staged)
        staged.replace(destination)

    return {"sha256": sha256, MESH_REL_PATH: True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump normalized TRELLIS2 meshes")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--recompute-finished", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    workspace = DatasetWorkspace(args.dataset_dir)
    raw_files = workspace.files("raw", "")
    mesh_files = workspace.files(MESH_REL_PATH, ".pickle")
    mesh_files.mkdir()
    blender_path = install_blender()

    metadata = workspace.read_metadata()
    if mesh_files.rel_path not in metadata.columns:
        metadata[mesh_files.rel_path] = False

    ready = metadata[raw_files.rel_path].eq(True)
    if not args.recompute_finished:
        ready &= ~metadata[mesh_files.rel_path].eq(True)
    sha256s = metadata.loc[ready, "sha256"].astype(str)

    start = len(sha256s) * args.rank // args.world_size
    end = len(sha256s) * (args.rank + 1) // args.world_size
    selected = sha256s.iloc[start:end]

    inputs = []
    for value in selected:
        sha256 = str(value)
        raw_path = raw_files.find(sha256)
        if raw_path is None:
            raise FileNotFoundError(f"raw artifact not found for sha256 {sha256!r}")
        inputs.append({"sha256": sha256, "raw_path": raw_path})

    if not inputs:
        return

    stage = Stage(
        "dump mesh",
        dump_mesh,
        workers=args.num_workers,
        queue_size=args.num_workers,
        work_queue_size=args.num_workers,
        params={
            "blender_path": blender_path,
            "blender_script": BLENDER_SCRIPT,
            "verbose": args.verbose,
        },
        resources={"mesh_files": mesh_files},
    )
    results = Pipeline([stage]).run(inputs)

    records = pd.DataFrame(results, columns=["sha256", MESH_REL_PATH]).drop_duplicates(
        subset="sha256",
        keep="first",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    record_path = workspace.path(f"unmerged_records/trellis2_dump_mesh_{timestamp}_rank{args.rank}.csv")
    records.to_csv(record_path, index=False)


if __name__ == "__main__":
    main()
