import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class DatasetFiles:
    """Paths for one configured file type in a dataset workspace."""

    root: Path
    rel_path: str
    suffix: str
    _directory: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rel_path, str) or not self.rel_path:
            raise ValueError("rel_path must be a non-empty string")

        relative = PurePosixPath(self.rel_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in self.rel_path:
            raise ValueError(f"rel_path must be a safe relative path: {self.rel_path!r}")
        if relative.as_posix() != self.rel_path:
            raise ValueError(f"rel_path must be normalized: {self.rel_path!r}")

        if not isinstance(self.suffix, str):
            raise TypeError("suffix must be a string")
        if self.suffix and (not self.suffix.startswith(".") or "/" in self.suffix or "\\" in self.suffix):
            raise ValueError(f"suffix must be empty or start with '.': {self.suffix!r}")

        object.__setattr__(self, "_directory", self.root.joinpath(*relative.parts))

    @property
    def dir(self) -> Path:
        return self._directory

    def path(self, sha256: str) -> Path:
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal string")
        return self.dir / sha256[:2] / f"{sha256}{self.suffix}"

    def exists(self, sha256: str) -> bool:
        return self.path(sha256).is_file()

    def find(self, sha256: str) -> Optional[Path]:
        shard_dir = self.path(sha256).parent
        matches = sorted(path for path in shard_dir.glob(f"{sha256}.*") if path.is_file())
        suffixless = shard_dir / sha256
        if suffixless.is_file():
            matches.insert(0, suffixless)

        if len(matches) > 1:
            raise RuntimeError(f"multiple files found for sha256 {sha256!r} in {shard_dir}")
        return matches[0] if matches else None

    def mkdir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for prefix in range(256):
            (self.dir / f"{prefix:02x}").mkdir(exist_ok=True)


@dataclass(frozen=True)
class DatasetWorkspace:
    """Bind configured dataset file layouts to one root directory."""

    root: Path

    def __init__(self, root: Union[str, Path]) -> None:
        object.__setattr__(self, "root", Path(root).expanduser().resolve())
        self.root.mkdir(parents=True, exist_ok=True)
        self.path("merged_records").mkdir(parents=True, exist_ok=True)
        self.path("unmerged_records").mkdir(parents=True, exist_ok=True)

    def get_metadata(self, args) -> pd.DataFrame:
        raise NotImplementedError

    def download(self, metadata: pd.DataFrame, num_workers: int) -> pd.DataFrame:
        raise NotImplementedError

    def path(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path:
            raise ValueError("rel_path must be a non-empty string")

        relative = PurePosixPath(rel_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in rel_path:
            raise ValueError(f"rel_path must be a safe relative path: {rel_path!r}")
        if relative.as_posix() != rel_path:
            raise ValueError(f"rel_path must be normalized: {rel_path!r}")

        return self.root.joinpath(*relative.parts)

    def files(self, rel_path: str, suffix: str) -> DatasetFiles:
        return DatasetFiles(root=self.root, rel_path=rel_path, suffix=suffix)

    def read_metadata(self) -> pd.DataFrame:
        metadata = pd.read_csv(self.path("metadata.csv"), dtype={"sha256": str})
        return metadata

    def write_metadata(self, metadata: pd.DataFrame) -> None:
        metadata.to_csv(self.path("metadata.csv"), index=False)
