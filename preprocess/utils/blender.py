import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from urllib.request import urlretrieve

BLENDER_RELEASES = {
    "3.0.1": "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz",
    "4.5.1": "https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.5/blender-4.5.1-linux-x64.tar.xz",
    "5.1.2": "https://ftp.halifax.rwth-aachen.de/blender/release/Blender5.1/blender-5.1.2-linux-x64.tar.xz",
}


def install_blender(
    version: str = "5.1.2",
    install_dir: Union[str, Path] = "/tmp",
) -> Path:
    installation = f"blender-{version}-linux-x64"
    system_path = Path("/opt") / installation / "blender"
    if system_path.is_file():
        return system_path

    install_root = Path(install_dir).expanduser().resolve()
    blender_path = install_root / installation / "blender"
    if blender_path.is_file():
        return blender_path

    if version not in BLENDER_RELEASES:
        raise ValueError(f"unsupported Blender version: {version}")

    install_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=str(install_root)) as temporary:
        archive_path = Path(temporary) / f"{installation}.tar.xz"
        urlretrieve(BLENDER_RELEASES[version], archive_path)
        with tarfile.open(archive_path, mode="r:xz") as archive:
            archive.extractall(install_root)

    if not blender_path.is_file():
        raise FileNotFoundError(f"Blender {version} executable not found after extraction")
    return blender_path
