import queue
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import torch


def pair_collate(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Collate pair samples into batched sparse tensors.

    Each input sample is produced by `build_pair_sample` and represents one
    ordered source/destination latent pair. Source and destination rows are
    sparse latent-grid entries. The collate step prepends a batch id column to
    each coordinate row so downstream sparse modules can distinguish samples.

    Args:
        batch: List of sample dictionaries. For each item:
            `coords_src`: integer numpy array `[Nsrc, 3]`;
            `feats_src`: float numpy array `[Nsrc, C]`;
            `coords_dst`: integer numpy array `[Ndst, 3]`;
            `feats_dst`: float numpy array `[Ndst, C]`;
            `O_dst2src`: float numpy array `[3, 3]`;
            `t_dst2src`: float numpy array `[3]`;
            `s_dst2src`: integer numpy array `[1]`;
            `grid_size`: integer numpy array `[1]`.

    Returns:
        A dictionary of torch tensors:
            `coords_src`: int32 tensor `[sum_Nsrc, 4]`;
            `feats_src`: float32 tensor `[sum_Nsrc, C]`;
            `coords_dst`: int32 tensor `[sum_Ndst, 4]`;
            `feats_dst`: float32 tensor `[sum_Ndst, C]`;
            `O_dst2src`: float32 tensor `[B, 3, 3]`;
            `t_dst2src`: float32 tensor `[B, 3]`;
            `s_dst2src`: int64 tensor `[B]`;
            `grid_size`: int64 tensor `[B]`.

    Physical meaning:
        `O_dst2src` and `t_dst2src` map destination positions into the source
        coordinate frame. `s_dst2src` is the orientation token for that mapping.

    Algorithm:
        Convert numpy arrays to torch tensors, prefix each sparse coordinate
        row with its sample id, concatenate sparse rows across the batch, and
        stack per-sample transforms.
    """
    B = len(batch)

    coords_src_list = []
    feats_src_list = []
    coords_dst_list = []
    feats_dst_list = []

    O_dst2src_list = []
    t_dst2src_list = []
    s_dst2src_list = []
    grid_size_list = []

    for i, item in enumerate(batch):
        coords_src = torch.from_numpy(item["coords_src"]).to(dtype=torch.int32)
        coords_dst = torch.from_numpy(item["coords_dst"]).to(dtype=torch.int32)

        feats_src = torch.from_numpy(item["feats_src"]).to(dtype=torch.float32)
        feats_dst = torch.from_numpy(item["feats_dst"]).to(dtype=torch.float32)

        num_src = coords_src.shape[0]
        num_dst = coords_dst.shape[0]

        # First coordinate column is the batch id; remaining columns are grid xyz.
        batch_src = torch.full((num_src, 1), i, dtype=torch.int32)
        batch_dst = torch.full((num_dst, 1), i, dtype=torch.int32)
        coords_src_list.append(torch.cat([batch_src, coords_src], dim=1))
        coords_dst_list.append(torch.cat([batch_dst, coords_dst], dim=1))

        feats_src_list.append(feats_src)
        feats_dst_list.append(feats_dst)

        # One destination-to-source transform relation per sample.
        O_dst2src = torch.as_tensor(item["O_dst2src"], dtype=torch.float32)
        t_dst2src = torch.as_tensor(item["t_dst2src"], dtype=torch.float32)
        s_dst2src = torch.as_tensor(item["s_dst2src"], dtype=torch.int64)
        grid_size = torch.as_tensor(item["grid_size"], dtype=torch.int64)

        O_dst2src_list.append(O_dst2src)
        t_dst2src_list.append(t_dst2src)
        s_dst2src_list.append(s_dst2src)
        grid_size_list.append(grid_size)

    out = {
        "coords_src": torch.cat(coords_src_list, dim=0),
        "feats_src": torch.cat(feats_src_list, dim=0),
        "coords_dst": torch.cat(coords_dst_list, dim=0),
        "feats_dst": torch.cat(feats_dst_list, dim=0),
        "O_dst2src": torch.stack(O_dst2src_list, dim=0),
        "t_dst2src": torch.stack(t_dst2src_list, dim=0),
        "s_dst2src": torch.cat(s_dst2src_list, dim=0),
        "grid_size": torch.cat(grid_size_list, dim=0),
    }
    return out


def move_tree_to_device(x: Any, device: torch.device) -> Any:
    """Move every tensor in a nested object tree to a device.

    Args:
        x: A tensor, dict, list, tuple, or arbitrary non-tensor leaf. Dict,
            list, and tuple values are traversed recursively.
        device: Target torch device.

    Returns:
        An object with the same nested structure as `x`. Tensor leaves are moved
        to `device` with `non_blocking=True`; non-tensor leaves are returned
        unchanged.

    Algorithm:
        Recursively map tensors through `Tensor.to(device=..., non_blocking=True)`.
    """
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_tree_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [move_tree_to_device(v, device) for v in x]
        return type(x)(y)
    return x


def record_stream_tree(x: Any, stream: torch.cuda.Stream) -> None:
    """Record a CUDA stream on every CUDA tensor in a nested object tree.

    Args:
        x: A tensor, dict, list, tuple, or arbitrary non-tensor leaf.
        stream: CUDA stream that will consume tensors in `x`.

    Returns:
        None. CUDA tensor leaves are updated in place through
        `Tensor.record_stream`.

    Physical meaning:
        Recording the consumer stream keeps CUDA tensor storage alive until that
        stream has finished using it. CPU tensors and non-tensor leaves are
        ignored.

    Algorithm:
        Recursively traverse containers and call `record_stream` on CUDA tensor
        leaves.
    """
    if torch.is_tensor(x) and x.is_cuda:
        x.record_stream(stream)
        return
    if isinstance(x, dict):
        for v in x.values():
            record_stream_tree(v, stream)
        return
    if isinstance(x, (list, tuple)):
        for v in x:
            record_stream_tree(v, stream)
        return


class Prefetcher:
    """One-batch-ahead device prefetcher for DataLoader iteration.

    Usage:
        for batch in Prefetcher(loader, device):
            ...

    Args:
        loader: Any iterable that yields batch objects. Batches may be nested
            tensor trees supported by `move_tree_to_device`.
        device: Target torch device or device string.

    Yields:
        Batches on `device`. For CPU devices, this is exactly the loader output.
        For CUDA devices, tensor leaves have already been copied to the target
        device on a dedicated prefetch stream.

    Algorithm:
        CPU mode directly wraps the loader iterator. CUDA mode starts one
        daemon worker thread. The worker reads one CPU batch, moves it to the
        target GPU on a dedicated CUDA stream, records an event, and places the
        result in a queue of size one. The consumer waits on the event before
        returning the batch and records the current stream on all CUDA tensors.
    """

    def __init__(self, loader: Any, device: torch.device | str):
        """Create a prefetcher over `loader`.

        Args:
            loader: Iterable batch source.
            device: Target torch device or device string.

        Returns:
            None. CUDA mode starts a background worker thread immediately.
        """
        self.device = torch.device(device)
        self.loader = loader

        self.use_cuda = self.device.type == "cuda"
        self._it: Iterator = iter(loader)

        if not self.use_cuda:
            return

        self._stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))
        self._q: queue.Queue[Optional[Tuple[Any, torch.cuda.Event]]] = queue.Queue(maxsize=1)
        self._exc: Optional[Exception] = None
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def __iter__(self):
        """Return this prefetcher as its own iterator."""
        return self

    def _worker_loop(self) -> None:
        """Load and transfer batches on the dedicated CUDA stream.

        Returns:
            None. The worker emits `(batch, event)` pairs into the queue and
            stores any exception for the consumer thread to re-raise.
        """
        try:
            while not self._stop.is_set():
                try:
                    cpu_batch = next(self._it)
                except StopIteration:
                    self._q.put(None)
                    return

                with torch.cuda.stream(self._stream):
                    gpu_batch = move_tree_to_device(cpu_batch, self.device)
                    ev = cast(torch.cuda.Event, torch.cuda.Event())
                    ev.record(self._stream)

                self._q.put((gpu_batch, ev))
        except Exception as e:
            self._exc = e
            try:
                self._q.put_nowait(None)
            except Exception:
                pass

    def __next__(self):
        """Return the next batch, waiting for CUDA prefetch work if needed.

        Returns:
            The next batch from the wrapped loader. CUDA mode returns a batch on
            `self.device`; CPU mode returns the original loader batch.

        Raises:
            StopIteration: When the wrapped loader is exhausted.
            Exception: Re-raises exceptions captured in the CUDA worker thread.
        """
        if not self.use_cuda:
            return next(self._it)

        if self._exc is not None:
            raise self._exc

        item = self._q.get()
        if item is None:
            self._stop.set()
            raise StopIteration

        batch, ev = item

        cur = cast(torch.cuda.Stream, torch.cuda.current_stream(device=self.device))
        cur.wait_event(cast(Any, ev))
        record_stream_tree(batch, cur)

        return batch

    def close(self) -> None:
        """Ask the CUDA worker thread to stop.

        Returns:
            None. CPU mode has no background worker and does nothing.
        """
        if not self.use_cuda:
            return
        self._stop.set()
