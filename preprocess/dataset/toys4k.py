import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from zipfile import ZipFile

import pandas as pd

from ..utils import Pipeline, Stage, sha256_file
from .base import DatasetWorkspace


def add_args(parser: ArgumentParser) -> None:
    pass


def extract_toys4k_file(
    sha256: str,
    filename: str,
    archive: ZipFile,
    workspace: DatasetWorkspace,
) -> Dict[str, object]:
    raw_files = workspace.files("raw", Path(filename).suffix)
    destination = raw_files.path(sha256)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        member = archive.getinfo(filename)
        with archive.open(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)

        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != sha256:
            raise ValueError(f"sha256 mismatch for {filename}: expected {sha256}, got {actual_sha256}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {"sha256": sha256, "raw": True}


class Toys4KDataset(DatasetWorkspace):
    def get_metadata(self, args) -> pd.DataFrame:
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/Toys4k.csv")
        return metadata.drop_duplicates(subset="sha256", keep="first")

    def download(self, metadata: pd.DataFrame, num_workers: int) -> pd.DataFrame:
        raw_files = self.files("raw", "")
        records = metadata[["sha256", "file_identifier"]].to_dict("records")
        if not records:
            return pd.DataFrame(columns=["sha256", "raw"])

        source_zip = self.path("toys4k_blend_files.zip")
        if not source_zip.is_file():
            raise FileNotFoundError("Toys4K must be downloaded manually. " f"Place toys4k_blend_files.zip at {source_zip}. " "Download instructions: " "https://github.com/rehg-lab/lowshot-shapebias/tree/main/toys4k")

        raw_files.mkdir()
        inputs = [
            {
                "sha256": record["sha256"],
                "filename": f"toys4k_blend_files/{record['file_identifier']}",
            }
            for record in records
        ]
        with ZipFile(source_zip) as archive:
            stage = Stage(
                "download",
                extract_toys4k_file,
                workers=num_workers,
                resources={"archive": archive, "workspace": self},
            )
            results = Pipeline([stage]).run(inputs)

        return pd.DataFrame(results, columns=["sha256", "raw"]).drop_duplicates(
            subset="sha256",
            keep="first",
        )
