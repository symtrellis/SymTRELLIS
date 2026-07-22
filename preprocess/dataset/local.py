import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import pandas as pd

from ..utils import Pipeline, Stage, sha256_file
from .base import DatasetFiles, DatasetWorkspace


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument("--input_dir", type=str, help="Folder containing GLB files")


def copy_local_file(
    sha256: str,
    original_path: str,
    raw_files: DatasetFiles,
) -> Dict[str, object]:
    destination = raw_files.path(sha256)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        shutil.copy2(Path(original_path), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {"sha256": sha256, "raw": raw_files.exists(sha256)}


class LocalDataset(DatasetWorkspace):

    def get_metadata(self, args) -> pd.DataFrame:
        input_dir = Path(args.input_dir).expanduser().resolve()
        glb_files = sorted(input_dir.glob("*.glb"))
        rows = [
            {
                "sha256": sha256_file(path),
                "original_path": str(path.resolve()),
                "aesthetic_score": None,
                "captions": "[]",
            }
            for path in glb_files
        ]
        metadata = pd.DataFrame(rows)
        return metadata.drop_duplicates(subset="sha256", keep="first")

    def download(self, metadata: pd.DataFrame, num_workers: int) -> pd.DataFrame:
        raw_files = self.files("raw", ".glb")
        raw_files.mkdir()
        records = metadata[["sha256", "original_path"]].to_dict("records")
        inputs = [
            {
                "sha256": record["sha256"],
                "original_path": record["original_path"],
            }
            for record in records
        ]

        stage = Stage(
            "download",
            copy_local_file,
            workers=num_workers,
            resources={"raw_files": raw_files},
        )
        results = Pipeline([stage]).run(inputs)
        return pd.DataFrame(results, columns=["sha256", "raw"])
