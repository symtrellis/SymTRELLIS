import hashlib
import tarfile
from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_file_bytes(path: PathLike) -> bytes:
    with Path(path).open("rb") as file:
        return file.read()


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    return hashlib.sha256(data).hexdigest()


def write_file_bytes(
    path: PathLike,
    data: bytes | bytearray | memoryview,
    *,
    overwrite: bool = True,
) -> None:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")

    output_path = Path(path)
    if not overwrite and output_path.exists():
        raise FileExistsError(f"File already exists: {output_path}")

    with output_path.open("wb") as file:
        file.write(bytes(data))


def build_tar_index(tar_path: PathLike) -> dict[str, tuple[int, int]]:
    index: dict[str, tuple[int, int]] = {}
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            if member.isreg():
                index[member.name] = (member.offset_data, member.size)
    return index
