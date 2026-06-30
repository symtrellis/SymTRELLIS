import io
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

from .base import BasePairDataset, parse_entry_stem


def single_zip_write(
    zip_file: Path,
    data_plan: Dict[str, Tuple[int, int, int]],
    shm_dir: Path,
) -> None:
    """Copy all planned `.npz` members from one zip archive into mmap chunks.

    Physical meaning:
        Each zip archive stores all transformed latent records for one shape.
        `data_plan` maps each latent record key to its byte location in a shared
        memory chunk file. This worker copies compressed `.npz` member bytes
        into those preallocated chunk files.

    Args:
        zip_file: Path to one shape archive.
        data_plan: Mapping from `"<sha256>/<entry_stem>"` to
            `(chunk_id, start, size)`, where `start` and `size` are byte offsets
            inside the chunk mmap file.
        shm_dir: Directory containing preallocated chunk files named
            `0000`, `0001`, ...

    Returns:
        None. The function writes bytes into mmap files as a side effect.

    Algorithm:
        Read the zip archive into memory, iterate through its members, locate
        the target chunk and byte range from `data_plan`, copy the raw member
        bytes into that mmap, then flush every touched mmap.
    """
    sha256 = zip_file.stem
    mmaps = {}

    zip_bytes = zip_file.read_bytes()

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            key = f"{sha256}/{Path(name).stem}"
            chunk_id, start, size = data_plan[key]
            if chunk_id not in mmaps:
                chunk_pth = shm_dir / f"{chunk_id:04d}"
                if not chunk_pth.is_file():
                    continue
                mmaps[chunk_id] = np.memmap(chunk_pth, mode="r+", dtype=np.uint8)
            mm = mmaps[chunk_id]

            data = zf.read(name)
            mm[start : start + size] = np.frombuffer(data, dtype=np.uint8)

    for mm in mmaps.values():
        mm.flush()


def write_mmap(p: Path, arr: np.ndarray) -> None:
    """Write a numpy array to a standalone mmap file.

    Args:
        p: Output file path.
        arr: Numpy array to write. Its dtype and shape are preserved by the
            caller through separate metadata.

    Returns:
        None. The array bytes are written to `p`.

    Algorithm:
        Create a writable memmap with the same dtype and shape as `arr`, copy
        the array into it, flush, and release the mmap handle.
    """
    mm = np.memmap(str(p), mode="w+", dtype=arr.dtype, shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm


def mmap_arr(p: Path, dtype, shape) -> np.memmap:
    """Open a read-only mmap-backed numpy array.

    Args:
        p: Mmap file path.
        dtype: Numpy dtype used when the file was written.
        shape: Array shape used when the file was written.

    Returns:
        A read-only `np.memmap` view over `p`.

    Algorithm:
        Delegate to `np.memmap` with mode `r`.
    """
    return np.memmap(str(p), dtype=dtype, shape=shape, mode="r")


class SharedMemoryDataBuffer:
    """Prepare zip latent records for shared-memory dataset reads.

    Physical meaning:
        This class builds a byte-level cache for latent `.npz` records. Each
        shape archive remains the logical dataset unit, but every archive member
        is copied into large shared-memory chunk files so DataLoader workers can
        read records without repeatedly opening zip files.

    Data layout:
        `data_plan` maps `"<sha256>/<entry_stem>"` to
        `(chunk_id, start, size)`. The three index arrays written by `load`
        have shape `[num_shapes, num_scale * num_rots * num_perts]` and map a
        shape/copy index to the chunk id, byte start, and byte length.
    """

    align_bytes: int = 64

    def __init__(
        self,
        shape_paths: List[Union[str, Path]],
        grid_size: int,
        num_scale: int,
        num_rots: int,
        num_perts: int,
        chunk_size_bytes: int = 2 * 1024**3,
    ):
        """Build the byte placement plan for a set of shape archives.

        Args:
            shape_paths: List of zip archive paths. Each archive contains `.npz`
                members named by the canonical entry schema parsed by
                `parse_entry_stem`.
            grid_size: Cubic latent grid resolution.
            num_scale: Number of scale variants per shape.
            num_rots: Number of rotation samples per scale.
            num_perts: Number of perturbation samples per rotation.
            chunk_size_bytes: Target maximum bytes per shared-memory chunk
                before alignment.

        Returns:
            None. The constructor records the placement plan but does not create
            mmap files; call `load` for that.

        Algorithm:
            Iterate through every zip member, validate its entry indices, assign
            it a byte range in the current chunk, and advance to a new chunk
            when the current one would exceed `chunk_size_bytes`.
        """
        self.shape_paths = [Path(p) for p in shape_paths]

        self.grid_size = int(grid_size)
        self.num_scale = int(num_scale)
        self.num_rots = int(num_rots)
        self.num_perts = int(num_perts)

        def align_up(x: int) -> int:
            # Align member starts to keep chunk offsets stable and mmap-friendly.
            a = self.align_bytes
            return (x + a - 1) // a * a

        self.chunk_size_bytes = align_up(int(chunk_size_bytes))

        # data key format: "<shape_sha256>/<entry_stem>" -> (chunk_id, offset, length)
        self.data_plan: Dict[str, Tuple[int, int, int]] = {}
        self.chunk_size: Dict[int, int] = {}

        self.filewize_data_plan: Dict[str, Dict[str, Tuple[int, int, int]]] = {}

        self.shm_dir: Optional[Path] = None

        cur_chunk, cur_pos = 0, 0
        for zp in tqdm(self.shape_paths):
            sha256 = zp.stem
            with zipfile.ZipFile(zp, "r") as zf:
                # I make this zip with only names, no folder
                self.filewize_data_plan[sha256] = {}

                for info in zf.infolist():
                    name = info.filename
                    if not name.lower().endswith(".npz"):
                        raise NameError(f"{str(zp)} has in correctly formated file: {name}!")

                    (
                        scale_idx,
                        rot_idx,
                        pert_idx,
                    ) = parse_entry_stem(Path(name).stem)

                    assert scale_idx < self.num_scale
                    assert rot_idx < self.num_rots
                    assert pert_idx < self.num_perts

                    key = f"{sha256}/{Path(name).stem}"

                    # assign offset
                    size = int(info.file_size)
                    if cur_pos + size > self.chunk_size_bytes:
                        cur_chunk += 1
                        cur_pos = 0

                    self.data_plan[key] = (cur_chunk, cur_pos, size)
                    self.filewize_data_plan[sha256][key] = (cur_chunk, cur_pos, size)

                    cur_pos += align_up(size)
                    self.chunk_size[cur_chunk] = align_up(cur_pos)

        self.info_str: str = ""

    def load(self, num_workers: int = 24) -> str:
        """Materialize the planned cache into shared memory.

        Args:
            num_workers: Maximum number of forked worker processes used to copy
                zip member bytes into chunk files.

        Returns:
            A JSON string containing the shared-memory directory, index array
            shapes, dataset dimensions, and `grid_size`. This string is passed
            to `SharedMemoryPairDataset`.

        Algorithm:
            1. Create one file per planned byte chunk under `/dev/shm`.
            2. Build index arrays mapping `[shape_idx, copy_idx]` to byte
               location `(chunk_id, start, size)`.
            3. Use multiprocessing workers to copy each zip archive into the
               preallocated chunks.
            4. Write the index arrays as mmap files and return their metadata.
        """
        if self.shm_dir is not None:
            self.cleanup()

        # create shm chunk and also the json's plan
        self.shm_dir = Path(
            tempfile.mkdtemp(
                prefix="data_buffer_",
                dir="/dev/shm",
            )
        ).resolve()

        def os_create(p: Path, size: int):
            fd = os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
            os.ftruncate(fd, size)
            os.close(fd)

        for idx, size in self.chunk_size.items():
            os_create(self.shm_dir / f"{idx:04d}", size)

        task_zipfiles: Set[Path] = set()

        sha256_file = {p.stem: p for p in self.shape_paths}
        sha256_index = {p.stem: i for i, p in enumerate(self.shape_paths)}

        N = len(self.shape_paths)
        M = self.num_scale * self.num_rots * self.num_perts

        # Index arrays map one transformed copy to its byte location in shm.
        chunk_arr = np.full((N, M), fill_value=-1, dtype=np.int64)
        start_arr = np.empty((N, M), dtype=np.int64)
        size_arr = np.empty((N, M), dtype=np.int64)

        for key, (idx, start, size) in self.data_plan.items():
            sha256 = key.split("/")[0]
            (
                scale_idx,
                rot_idx,
                pert_idx,
            ) = parse_entry_stem(key.split("/")[1])

            shape_idx = sha256_index[sha256]
            copy_idx = scale_idx * self.num_rots * self.num_perts + rot_idx * self.num_perts + pert_idx

            chunk_arr[shape_idx, copy_idx] = idx
            start_arr[shape_idx, copy_idx] = start
            size_arr[shape_idx, copy_idx] = size

            task_zipfiles.add(sha256_file[sha256])

        assert (chunk_arr >= 0).all()

        # write memory
        num_workers = min(int(num_workers), max(1, os.cpu_count() or 1))

        args_list = [
            (
                task_zipfile,
                self.filewize_data_plan[task_zipfile.stem],
                self.shm_dir,
            )
            for task_zipfile in task_zipfiles
        ]

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            pbar = tqdm(total=len(args_list), desc=f"load zip -> {self.shm_dir}")

            def done(_) -> None:
                pbar.update(1)

            def err(e) -> None:
                pbar.update(1)
                raise e

            results = []
            for args in args_list:
                r = pool.apply_async(
                    single_zip_write,
                    args=args,
                    callback=done,
                    error_callback=err,
                )
                results.append(r)

            for r in results:
                r.get()

        # write loading information to string
        write_mmap(self.shm_dir / "chunk_arr", chunk_arr)
        write_mmap(self.shm_dir / "start_arr", start_arr)
        write_mmap(self.shm_dir / "size_arr", size_arr)

        self.info_str = json.dumps(
            {
                "shm": str(self.shm_dir.absolute()),
                "cs": chunk_arr.shape,
                "ss": start_arr.shape,
                "ls": size_arr.shape,
                "nshape": len(self.shape_paths),
                "nscale": self.num_scale,
                "nrots": self.num_rots,
                "nperts": self.num_perts,
                "grid_size": self.grid_size,
            }
        )
        return self.info_str

    def cleanup(self) -> None:
        """Remove the current shared-memory cache directory.

        Returns:
            None. The method deletes `self.shm_dir` if it exists and resets the
            cached info string.
        """
        if self.shm_dir is not None:
            shutil.rmtree(self.shm_dir, ignore_errors=True)
        self.shm_dir = None
        self.info_str = ""

    @contextmanager
    def loaded(self, num_workers: int = 24):
        """Context manager that loads and then cleans up shared memory.

        Args:
            num_workers: Worker count forwarded to `load`.

        Yields:
            The JSON info string returned by `load`.

        Algorithm:
            Materialize the shared-memory cache on entry and always call
            `cleanup` on exit.
        """
        p = self.load(num_workers=num_workers)
        try:
            yield p
        finally:
            self.cleanup()


class SharedMemoryPairDataset(BasePairDataset):
    """Pair dataset backend that reads latent records from shared-memory chunks.

    The dataset consumes the JSON info string produced by
    `SharedMemoryDataBuffer.load`. It lazily opens mmap files in each process
    and reconstructs `.npz` records from byte ranges stored in shared memory.
    """

    def __init__(
        self,
        info_str: str,
    ) -> None:
        """Initialize the shared-memory pair dataset.

        Args:
            info_str: JSON metadata produced by `SharedMemoryDataBuffer.load`.

        Returns:
            None. Mmap files are not opened until `load_mmap` or `read_item`.

        Algorithm:
            Parse the info string, validate index array shapes, initialize the
            base pair index space, and store mmap metadata for lazy loading.
        """
        info = json.loads(info_str)
        assert info["cs"] == info["ss"]
        assert info["cs"] == info["ls"]
        assert info["cs"][1] == info["nscale"] * info["nrots"] * info["nperts"]

        super().__init__(
            grid_size=info["grid_size"],
            num_shapes=info["nshape"],
            num_scale=info["nscale"],
            num_rots=info["nrots"],
            num_perts=info["nperts"],
        )

        self.shm_dir: Path = Path(info["shm"])
        self.chunk_shape: Tuple[int, int] = info["cs"]
        self.start_shape: Tuple[int, int] = info["ss"]
        self.size_shape: Tuple[int, int] = info["ls"]

        self.mmap_loaded: bool = False
        self.mmap: Dict[int, np.memmap] = {}
        self.num_chunk: int = -1

    def load_mmap(self) -> None:
        """Open all mmap files referenced by this dataset.

        Returns:
            None. The method populates index-array memmaps and raw byte chunk
            memmaps on the dataset instance.

        Algorithm:
            Open the three index arrays, infer the number of data chunks from
            `chunk_arr`, and open each chunk file as a read-only uint8 memmap.
        """
        self.chunk_arr = mmap_arr(self.shm_dir / "chunk_arr", dtype=np.int64, shape=self.chunk_shape)
        self.start_arr = mmap_arr(self.shm_dir / "start_arr", dtype=np.int64, shape=self.start_shape)
        self.size_arr = mmap_arr(self.shm_dir / "size_arr", dtype=np.int64, shape=self.size_shape)

        self.num_chunk = int(self.chunk_arr.max() + 1)
        for cid in range(self.num_chunk):
            self.mmap[cid] = np.memmap(self.shm_dir / f"{cid:04d}", mode="r", dtype=np.uint8)

        self.mmap_loaded = True

    def read_npz(self, chunk_id, start, size) -> Dict:
        """Read one `.npz` record from a shared-memory byte range.

        Args:
            chunk_id: Integer id of the raw byte chunk.
            start: Start byte offset inside the chunk.
            size: Number of bytes to read.

        Returns:
            A dictionary of numpy arrays loaded from the `.npz` byte payload.

        Algorithm:
            Slice the uint8 mmap range, copy it into a `BytesIO` buffer, and
            load it with `np.load(..., allow_pickle=False)`.
        """
        mm = self.mmap[chunk_id]
        with np.load(
            io.BytesIO(mm[start : start + size].tobytes()),
            allow_pickle=False,
        ) as z:
            out = {k: z[k] for k in z.files}
        return out

    def read_item(
        self,
        shape_idx: int,
        scale_idx: int,
        rot_idx_src: int,
        pert_idx_src: int,
        rot_idx_dst: int,
        pert_idx_dst: int,
    ) -> Tuple[Dict, Dict]:
        """Load source and destination latent records for one pair.

        Args:
            shape_idx: Shape archive index.
            scale_idx: Shared scale index for both records.
            rot_idx_src: Source rotation index.
            pert_idx_src: Source perturbation index.
            rot_idx_dst: Destination rotation index.
            pert_idx_dst: Destination perturbation index.

        Returns:
            `(data_src, data_dst)`. Each dictionary contains numpy arrays loaded
            from one `.npz` record. Required keys are `transform` and `feats`;
            sparse records may also contain `coords`.

        Algorithm:
            Lazily open mmap files, convert source/destination transform indices
            into linear copy indices, use the index arrays to find byte ranges,
            and decode both `.npz` records from shared memory.
        """

        if not self.mmap_loaded:
            self.load_mmap()

        copy_idx_src = scale_idx * self.num_rots * self.num_perts + rot_idx_src * self.num_perts + pert_idx_src
        chunk_idx_src = self.chunk_arr[shape_idx, copy_idx_src]
        start_idx_src = self.start_arr[shape_idx, copy_idx_src]
        size_idx_src = self.size_arr[shape_idx, copy_idx_src]

        copy_idx_dst = scale_idx * self.num_rots * self.num_perts + rot_idx_dst * self.num_perts + pert_idx_dst
        chunk_idx_dst = self.chunk_arr[shape_idx, copy_idx_dst]
        start_idx_dst = self.start_arr[shape_idx, copy_idx_dst]
        size_idx_dst = self.size_arr[shape_idx, copy_idx_dst]

        data_src = self.read_npz(chunk_idx_src, start_idx_src, size_idx_src)
        data_dst = self.read_npz(chunk_idx_dst, start_idx_dst, size_idx_dst)

        return data_src, data_dst
