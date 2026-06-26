"""Sparse lattice radius-neighbor backend."""

from __future__ import annotations

import os
import tempfile
import threading

import torch
from torch.utils.cpp_extension import load


_EXT = None
_EXT_LOCK = threading.Lock()


def _load_sparse_lattice_ext():
    global _EXT

    if _EXT is not None:
        return _EXT

    with _EXT_LOCK:
        if _EXT is not None:
            return _EXT

        this_dir = os.path.dirname(os.path.abspath(__file__))
        ext_dir = os.path.join(this_dir, "sparse_lattice_ext")
        build_dir = os.environ.get(
            "SYMTRELLIS_SPARSE_LATTICE_BUILD_DIR",
            os.path.join(tempfile.gettempdir(), "symtrellis_sparse_lattice_ext"),
        )
        os.makedirs(build_dir, exist_ok=True)

        _EXT = load(
            name="symtrellis_sparse_lattice_ext",
            sources=[
                os.path.join(ext_dir, "bindings.cpp"),
                os.path.join(ext_dir, "radius_nbr_edges_sparse_lattice.cu"),
            ],
            build_directory=build_dir,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=bool(int(os.environ.get("SYMTRELLIS_EXT_VERBOSE", "0"))),
        )
        return _EXT


def _check_shape(name: str, tensor: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if tensor.ndim != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dimensions, got {tensor.ndim}")

    for dim, expected in enumerate(shape):
        if expected is not None and tensor.shape[dim] != expected:
            raise ValueError(
                f"{name}.shape[{dim}] must be {expected}, got {tensor.shape[dim]}"
            )


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

    Large absolute coordinates should use float64 query positions, or already
    shifted local float32 positions, so sub-cell residuals stay accurate.
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
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    device = query_pos.device
    if not device.type == "cuda":
        raise NotImplementedError("radius_nbr_edges_sparse_lattice currently has only a CUDA backend")

    for name, tensor in (
        ("query_bid", query_bid),
        ("key_coords", key_coords),
        ("key_bid", key_bid),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as query_pos")

    nbr_offsets = nbr_offsets.to(device=device, dtype=torch.int32, non_blocking=True)
    coord_min = coord_min.to(device=device, dtype=torch.int32, non_blocking=True)

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
    kids_by_offset = ext.radius_nbr_kids_by_offset_sparse_lattice_cuda(
        query_pos,
        query_bid,
        key_coords,
        key_bid,
        float(radius),
        nbr_offsets,
        coord_min,
    )

    mask = kids_by_offset >= 0
    qids, _ = mask.nonzero(as_tuple=True)
    kids = kids_by_offset[mask]
    return qids, kids
