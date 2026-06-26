"""Sparse lattice radius-neighbor backend."""

from __future__ import annotations

import math
import threading

import torch

_EXT = None
_EXT_LOCK = threading.Lock()


def _load_sparse_lattice_ext():
    global _EXT

    if _EXT is not None:
        return _EXT

    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT

        from . import _sparse_lattice_ext

        _EXT = _sparse_lattice_ext
        return _EXT


def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tensor.ndim}")

    for dim, expected in enumerate(shape):
        if expected is not None and tensor.shape[dim] != expected:
            raise ValueError(f"{name}.shape[{dim}] must be {expected}, got {tensor.shape[dim]}")


def radius_nbr_edges_sparse_lattice(
    query_pos: torch.Tensor,
    query_bid: torch.Tensor,
    key_coords: torch.Tensor,
    key_bid: torch.Tensor,
    radius: float,
    nbr_offsets: torch.Tensor,
    coord_min: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find radius-neighbor edges on a sparse integer lattice.

    This backend supports arbitrary int32 lattice coordinates, including
    negative coordinates. It internally shifts coordinates by `coord_min` and
    indexes keys in 4x4x4 coarse buckets.

    Args:
        query_pos: Float tensor with shape [Nq, 3]. Continuous query positions
            in the same grid-index coordinate system as `key_coords`.
        query_bid: Int32 tensor with shape [Nq]. Batch id for each query row.
        key_coords: Int32 tensor with shape [Nk, 3]. Sparse integer lattice
            coordinates. For every key row, `key_coords - coord_min` must be
            non-negative.
        key_bid: Int32 tensor with shape [Nk]. Batch id for each key row.
        radius: Search radius in grid-index units. Must be finite and positive.
        nbr_offsets: Int32 tensor with shape [L, 3], usually returned by
            `lattice_ball_offsets(radius)`.
        coord_min: Int32 tensor with shape [3]. Coordinate origin used to shift
            arbitrary lattice coordinates into the non-negative bucket space.

    Returns:
        qids: Int64 tensor with shape [E]. Query row index for each edge.
        kids: Int64 tensor with shape [E]. Key row index for each edge.

    Notes:
        All tensors must live on the same CPU or CUDA device. The wrapper makes
        contiguous copies before calling the extension. Duplicate key rows that
        map to the same batch, bucket, and 4x4x4 local slot are not supported by
        the CUDA backend. Large absolute coordinates should use float64
        `query_pos`, or already shifted local float32 positions, so sub-cell
        residuals stay accurate.
    """
    _check_shape("query_pos", query_pos, (None, 3))
    _check_shape("query_bid", query_bid, (query_pos.shape[0],))
    _check_shape("key_coords", key_coords, (None, 3))
    _check_shape("key_bid", key_bid, (key_coords.shape[0],))
    _check_shape("nbr_offsets", nbr_offsets, (None, 3))
    _check_shape("coord_min", coord_min, (3,))

    if query_pos.dtype not in (torch.float32, torch.float64):
        raise TypeError("query_pos must be float32 or float64")
    if query_bid.dtype != torch.int32:
        raise TypeError("query_bid must be int32")
    if key_coords.dtype != torch.int32:
        raise TypeError("key_coords must be int32")
    if key_bid.dtype != torch.int32:
        raise TypeError("key_bid must be int32")
    if nbr_offsets.dtype != torch.int32:
        raise TypeError("nbr_offsets must be int32")
    if coord_min.dtype != torch.int32:
        raise TypeError("coord_min must be int32")
    radius_f = float(radius)
    if not math.isfinite(radius_f) or radius_f <= 0.0:
        raise ValueError("radius must be finite and positive")

    device = query_pos.device
    if device.type not in ("cpu", "cuda"):
        raise NotImplementedError("radius_nbr_edges_sparse_lattice supports CPU and CUDA tensors")

    for name, tensor in (
        ("query_bid", query_bid),
        ("key_coords", key_coords),
        ("key_bid", key_bid),
        ("nbr_offsets", nbr_offsets),
        ("coord_min", coord_min),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same device as query_pos")

    Nq = int(query_pos.shape[0])
    Nk = int(key_coords.shape[0])
    L = int(nbr_offsets.shape[0])
    if Nq == 0 or L == 0 or Nk == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        return empty, empty

    query_pos = query_pos.contiguous()
    query_bid = query_bid.contiguous()
    key_coords = key_coords.contiguous()
    key_bid = key_bid.contiguous()
    nbr_offsets = nbr_offsets.contiguous()
    coord_min = coord_min.contiguous()

    ext = _load_sparse_lattice_ext()
    if device.type == "cpu":
        kids_by_offset = ext.radius_nbr_kids_by_offset_sparse_lattice_cpu(
            query_pos,
            query_bid,
            key_coords,
            key_bid,
            radius_f,
            nbr_offsets,
            coord_min,
        )
    else:
        kids_by_offset = ext.radius_nbr_kids_by_offset_sparse_lattice_cuda(
            query_pos,
            query_bid,
            key_coords,
            key_bid,
            radius_f,
            nbr_offsets,
            coord_min,
        )

    mask = kids_by_offset >= 0
    qids, _ = mask.nonzero(as_tuple=True)
    kids = kids_by_offset[mask]
    return qids, kids
