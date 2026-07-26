import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from .base import Workspace


def folder_indices(workspace: Workspace, rel_path: str, valid_indices: set[str]) -> set[str]:
    completed_indices = set()

    for path in sorted(workspace.path(rel_path).iterdir()):
        if not path.is_file():
            continue

        idx = path.stem
        if idx not in valid_indices:
            continue
        if idx in completed_indices:
            raise RuntimeError(f"multiple files found for idx {idx!r} in {workspace.path(rel_path)}")

        completed_indices.add(idx)

    return completed_indices


def find_folders(workspace: Workspace, valid_indices: set[str]) -> dict[str, set[str]]:
    excluded = {"shapes", "renders", "merged_records", "unmerged_records"}
    discovered = {}

    for current, dirnames, _ in os.walk(workspace.root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames.sort()

        if current_path == workspace.root:
            dirnames[:] = [name for name in dirnames if name not in excluded]
            continue

        rel_path = current_path.relative_to(workspace.root).as_posix()
        completed_indices = folder_indices(workspace, rel_path, valid_indices)
        if completed_indices:
            discovered[rel_path] = completed_indices

    return dict(sorted(discovered.items()))


def sync_metadata(
    workspace: Workspace,
    full_check: bool = False,
    folders: list[str] | None = None,
) -> pd.DataFrame:
    rebuild = full_check or not workspace.path("metadata.csv").is_file()
    unmerged_records = sorted(workspace.path("unmerged_records").glob("*.csv"))

    if rebuild:
        metadata = workspace.get_metadata()
        valid_indices = set(metadata["idx"])

        for rel_path, completed_indices in find_folders(workspace, valid_indices).items():
            metadata[rel_path] = metadata["idx"].isin(completed_indices)
    else:
        metadata = workspace.read_metadata().set_index("idx")
        valid_indices = set(metadata.index)

        for path in unmerged_records:
            record = pd.read_csv(path, dtype={"idx": str}).set_index("idx")
            record = record.groupby(level=0, sort=False).last()

            if not set(record.index).issubset(valid_indices):
                raise RuntimeError(f"record {path.name!r} contains idx not present in renders")

            for column in record.columns.difference(metadata.columns):
                metadata[column] = False

            columns = metadata.columns.union(record.columns, sort=False)
            metadata = record.combine_first(metadata).reindex(index=metadata.index, columns=columns)

        if folders is not None:
            for rel_path in dict.fromkeys(folders):
                completed_indices = folder_indices(workspace, rel_path, valid_indices)
                metadata[rel_path] = metadata.index.isin(completed_indices)

        metadata = metadata.reset_index()

    workspace.write_metadata(metadata)

    for path in unmerged_records:
        shutil.move(str(path), str(workspace.path("merged_records")))

    return metadata


def main() -> None:
    parser = ArgumentParser(description="Build, update, and check evaluation metadata")
    parser.add_argument("--workspace-dir", required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--full-check", action="store_true")
    checks.add_argument("--folders", nargs="+", metavar="REL_PATH")
    args = parser.parse_args()

    workspace = Workspace(args.workspace_dir)
    sync_metadata(
        workspace,
        full_check=args.full_check,
        folders=args.folders,
    )


if __name__ == "__main__":
    main()
