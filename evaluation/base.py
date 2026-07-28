from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pandas as pd


@dataclass(frozen=True)
class Files:
    root: Path
    rel_path: str
    format: str = "{shape_id:06d}_{view_id:03d}"
    dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.rel_path:
            raise ValueError("rel_path must be a non-empty string")

        relative = PurePosixPath(self.rel_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in self.rel_path:
            raise ValueError(f"rel_path must be a safe relative path: {self.rel_path!r}")
        if relative.as_posix() != self.rel_path:
            raise ValueError(f"rel_path must be normalized: {self.rel_path!r}")

        if not self.format:
            raise ValueError("format must be a non-empty string")

        object.__setattr__(self, "dir", self.root.joinpath(*relative.parts))

    def path(self, suffix: str, **kwargs) -> Path:
        return self.dir / f"{self.format.format(**kwargs)}{suffix}"

    def exists(self, **kwargs) -> bool:
        pattern = f"{self.format.format(**kwargs)}.*"
        return any(path.is_file() for path in self.dir.glob(pattern))

    def mkdir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Workspace:
    root: Path
    shape_labels: dict[int, dict[str, str | int]]

    def __init__(self, root: str | Path) -> None:
        root = Path(root).expanduser().resolve()
        object.__setattr__(self, "root", root)

        self.root.mkdir(parents=True, exist_ok=True)
        self.path("merged_records").mkdir(parents=True, exist_ok=True)
        self.path("unmerged_records").mkdir(parents=True, exist_ok=True)

        axes = {"x": ("x", "z"), "y": ("z", "y"), "z": ("y", "x")}
        polyhedral_folds = {"I": 5, "Ih": 5, "O": 4, "Oh": 4, "T": 3, "Td": 3, "Th": 3}

        shape_labels = {}

        for shape_path in sorted(self.path("shapes").iterdir()):
            shape_id_text, group_text, symmetry_axis = shape_path.stem.split("_")

            shape_id = int(shape_id_text)
            symmetry_group = group_text[0].upper() + group_text[1:]
            major_axis, minor_axis = axes[symmetry_axis]

            if shape_id in shape_labels:
                raise RuntimeError(f"multiple shape files found for shape_id {shape_id}")

            if symmetry_group in polyhedral_folds:
                symmetry_fold = polyhedral_folds[symmetry_group]
            elif symmetry_group == "S1":
                symmetry_fold = 1
            else:
                order = int("".join(character for character in symmetry_group if character.isdigit()))
                symmetry_fold = order // 2 if symmetry_group.startswith("S") else order

            shape_labels[shape_id] = {
                "symmetry_group": symmetry_group,
                "symmetry_axis": symmetry_axis,
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "symmetry_fold": symmetry_fold,
            }

        object.__setattr__(self, "shape_labels", shape_labels)

    def get_metadata(self) -> pd.DataFrame:
        rows = []

        for render_path in self.path("renders").rglob("*.png"):
            shape_id = int(render_path.parent.name)
            view_id = int(render_path.stem)
            rows.append(
                {
                    "idx": f"{shape_id:06d}_{view_id:03d}",
                    "shape_id": shape_id,
                    "view_id": view_id,
                }
            )

        return pd.DataFrame(
            sorted(rows, key=lambda row: row["idx"]),
            columns=["idx", "shape_id", "view_id"],
        )

    def path(self, rel_path: str) -> Path:
        if not rel_path:
            raise ValueError("rel_path must be a non-empty string")

        relative = PurePosixPath(rel_path)
        if relative.is_absolute() or ".." in relative.parts or "\\" in rel_path:
            raise ValueError(f"rel_path must be a safe relative path: {rel_path!r}")
        if relative.as_posix() != rel_path:
            raise ValueError(f"rel_path must be normalized: {rel_path!r}")

        return self.root.joinpath(*relative.parts)

    def files(self, rel_path: str, format: str | None = None) -> Files:
        if format is None:
            return Files(root=self.root, rel_path=rel_path)

        return Files(root=self.root, rel_path=rel_path, format=format)

    def read_metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.path("metadata.csv"), dtype={"idx": str})

    def write_metadata(self, metadata: pd.DataFrame) -> None:
        metadata.to_csv(self.path("metadata.csv"), index=False)
