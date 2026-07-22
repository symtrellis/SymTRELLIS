import shutil
import tarfile
from argparse import ArgumentParser
from http.client import HTTPException
from pathlib import Path
from typing import Dict, Tuple
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from tqdm import tqdm

from ..utils import Pipeline, Stage, build_tar_index, sha256_file
from .base import DatasetWorkspace


def add_args(parser: ArgumentParser) -> None:
    pass


def extract_abo_file(
    sha256: str,
    filename: str,
    source_tar: Path,
    tar_index: Dict[str, Tuple[int, int]],
    workspace: DatasetWorkspace,
) -> Dict[str, object]:
    if filename not in tar_index:
        raise ValueError(f"Not found in TAR: {filename}")
    offset, size = tar_index[filename]

    raw_files = workspace.files("raw", Path(filename).suffix)
    destination = raw_files.path(sha256)
    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        remaining = size
        with source_tar.open("rb", buffering=0) as source, temporary.open("wb") as target:
            source.seek(offset)
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise IOError(f"Unexpected EOF: {filename}")
                target.write(chunk)
                remaining -= len(chunk)

        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != sha256:
            raise ValueError(f"sha256 mismatch for {filename}: expected {sha256}, got {actual_sha256}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {"sha256": sha256, "raw": raw_files.exists(sha256)}


class ABODataset(DatasetWorkspace):
    def get_metadata(self, args) -> pd.DataFrame:
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ABO.csv")
        return metadata.drop_duplicates(subset="sha256", keep="first")

    def download(self, metadata: pd.DataFrame, num_workers: int) -> pd.DataFrame:
        raw_files = self.files("raw", "")
        records = metadata[["sha256", "file_identifier"]].to_dict("records")
        if not records:
            return pd.DataFrame(columns=["sha256", "raw"])

        source_tar = self.path("abo-3dmodels.tar")
        if source_tar.is_file():
            tar_index = build_tar_index(source_tar)
        else:
            source_url = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
            information_url = "https://amazon-berkeley-objects.s3.amazonaws.com/index.html"
            temporary = source_tar.with_name(f".{source_tar.name}.tmp")
            try:
                with urlopen(source_url) as response, temporary.open("wb") as target:
                    content_length = response.headers.get("Content-Length")
                    total = int(content_length) if content_length is not None else None
                    with tqdm.wrapattr(response, "read", total=total, desc=source_tar.name) as source:
                        shutil.copyfileobj(source, target)

                tar_index = build_tar_index(temporary)
                temporary.replace(source_tar)
            except (HTTPException, URLError, tarfile.TarError) as error:
                raise FileNotFoundError("ABO archive download or validation failed. " f"Place abo-3dmodels.tar at {source_tar}, or download it from " f"{source_url}. More information: {information_url}") from error
            finally:
                if temporary.exists():
                    temporary.unlink()

        raw_files.mkdir()
        inputs = [
            {
                "sha256": record["sha256"],
                "filename": f"3dmodels/original/{record['file_identifier']}",
            }
            for record in records
        ]
        stage = Stage(
            "download",
            extract_abo_file,
            workers=num_workers,
            resources={
                "source_tar": source_tar,
                "tar_index": tar_index,
                "workspace": self,
            },
        )
        results = Pipeline([stage]).run(inputs)
        return pd.DataFrame(results, columns=["sha256", "raw"]).drop_duplicates(
            subset="sha256",
            keep="first",
        )
