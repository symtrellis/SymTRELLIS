import os
import re
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .dataset import args_dict, data_dict

SHARD_NAMES = frozenset(f"{index:02x}" for index in range(256))
ARTIFACT_FILENAME = re.compile(r"(?P<sha256>[0-9a-f]{64})(?:\..*)?")


def find_artifact_dirs(workspace) -> List[str]:
    artifact_dirs = []
    excluded = {"merged_records", "unmerged_records"}

    for current, dirnames, _ in os.walk(workspace.root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = [name for name in dirnames if name not in excluded and not name.startswith(".") and not (current_path / name).is_symlink()]
        if current_path != workspace.root and SHARD_NAMES.issubset(dirnames):
            artifact_dirs.append(current_path.relative_to(workspace.root).as_posix())
            dirnames.clear()

    return sorted(artifact_dirs)


def artifact_sha256s(workspace, rel_path: str) -> Set[str]:
    directory = workspace.path(rel_path)
    if not all((directory / shard).is_dir() and not (directory / shard).is_symlink() for shard in SHARD_NAMES):
        raise ValueError(f"{rel_path!r} does not contain complete 00-ff shard directories")

    sha256s = set()
    for shard in sorted(SHARD_NAMES):
        for path in sorted((directory / shard).iterdir()):
            if not path.is_file():
                continue
            match = ARTIFACT_FILENAME.fullmatch(path.name)
            if match is None:
                continue
            sha256 = match.group("sha256")
            if sha256[:2] != shard:
                continue
            if sha256 in sha256s:
                raise RuntimeError(f"multiple files found for sha256 {sha256!r} in {directory / shard}")
            sha256s.add(sha256)

    return sha256s


def sync_metadata(
    workspace,
    args: Namespace,
    full_check: bool = False,
    folders: Optional[List[str]] = None,
) -> pd.DataFrame:
    creating = not workspace.path("metadata.csv").is_file()
    if creating:
        metadata = workspace.get_metadata(args)
    else:
        metadata = workspace.read_metadata()

    metadata = metadata.set_index("sha256")
    metadata = metadata[~metadata.index.duplicated(keep="first")]

    unmerged = sorted(workspace.path("unmerged_records").glob("*.csv"))
    records = sorted(workspace.path("merged_records").glob("*.csv")) + unmerged if creating else unmerged
    for path in records:
        record = pd.read_csv(path, dtype={"sha256": str}).set_index("sha256")
        record = record.groupby(level=0, sort=False).last()
        index = metadata.index.union(record.index, sort=False)
        columns = metadata.columns.union(record.columns, sort=False)
        metadata = record.combine_first(metadata).reindex(index=index, columns=columns)

    metadata = metadata.reset_index()
    if creating or full_check:
        artifact_dirs = find_artifact_dirs(workspace)
    elif folders is not None:
        artifact_dirs = list(dict.fromkeys(folders))
    else:
        artifact_dirs = []

    for rel_path in artifact_dirs:
        metadata[rel_path] = metadata["sha256"].isin(artifact_sha256s(workspace, rel_path))

    workspace.write_metadata(metadata)
    for path in unmerged:
        shutil.move(str(path), str(workspace.path("merged_records")))

    return metadata


def main() -> None:
    parser = ArgumentParser(description="Build, update, and check dataset metadata")
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    for name, workspace_type in data_dict.items():
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--dataset-dir", required=True)
        checks = subparser.add_mutually_exclusive_group()
        checks.add_argument("--full-check", action="store_true")
        checks.add_argument("--folders", nargs="+", metavar="REL_PATH")
        args_dict[name](subparser)
        subparser.set_defaults(workspace_type=workspace_type)

    args = parser.parse_args()
    workspace = args.workspace_type(args.dataset_dir)
    sync_metadata(workspace, args, full_check=args.full_check, folders=args.folders)


if __name__ == "__main__":
    main()
