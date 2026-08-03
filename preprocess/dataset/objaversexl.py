import shutil
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import objaverse.xl as oxl
import pandas as pd

from .base import DatasetWorkspace


def add_args(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        type=str,
        default="sketchfab",
        help="Data source to download annotations from (github, sketchfab)",
    )


def store_downloaded_file(
    local_path: str,
    sha256: str,
    workspace: DatasetWorkspace,
    **kwargs,
) -> None:
    source = Path(local_path)
    raw_files = workspace.files("raw", source.suffix)
    destination = raw_files.path(sha256)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class ObjaverseXLDataset(DatasetWorkspace):
    def get_metadata(self, args) -> pd.DataFrame:
        if args.source == "sketchfab":
            return pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
        if args.source == "github":
            return pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
        raise ValueError(f"Invalid source: {args.source}")

    def download(self, metadata: pd.DataFrame, num_workers: int) -> pd.DataFrame:

        raw_files = self.files("raw", "")
        raw_files.mkdir()
        records = metadata[["sha256"]].to_dict("records")
        if not records:
            return pd.DataFrame(columns=["sha256", "raw"])

        with TemporaryDirectory(prefix="objaversexl_", dir="/dev/shm") as temporary_dir:
            annotations = oxl.get_annotations(download_dir=temporary_dir)
            annotations = annotations[annotations["sha256"].isin([record["sha256"] for record in records])]
            oxl.download_objects(
                annotations,
                download_dir=temporary_dir,
                processes=num_workers,
                handle_found_object=partial(store_downloaded_file, workspace=self),
            )

        results = [{"sha256": record["sha256"], "raw": True} for record in records if raw_files.find(record["sha256"]) is not None]
        return pd.DataFrame(results, columns=["sha256", "raw"]).drop_duplicates(
            subset="sha256",
            keep="first",
        )
