import torch

def radius_nbr_kids_by_offset_sparse_lattice_cpu(
    query_pos: torch.Tensor,
    query_bid: torch.Tensor,
    key_coords: torch.Tensor,
    key_bid: torch.Tensor,
    radius: float,
    nbr_offsets: torch.Tensor,
    coord_min: torch.Tensor,
) -> torch.Tensor:
    """Return CPU kids-by-offset table for sparse lattice radius neighbors.

    Args:
        query_pos: Float tensor [Nq, 3] with query positions in lattice coordinates.
        query_bid: Int32 tensor [Nq] with query batch ids.
        key_coords: Int32 tensor [Nk, 3] with sparse key lattice coordinates.
        key_bid: Int32 tensor [Nk] with key batch ids.
        radius: Positive finite search radius.
        nbr_offsets: Int32 tensor [L, 3] with integer offset candidates.
        coord_min: Int32 tensor [3] used to shift key coordinates before bucketing.

    Returns:
        Int64 tensor [Nq, L]. Entry (i, l) is the key row matched by offset l for
        query i, or -1 when no valid key exists within radius.
    """
    ...

def radius_nbr_kids_by_offset_sparse_lattice_cuda(
    query_pos: torch.Tensor,
    query_bid: torch.Tensor,
    key_coords: torch.Tensor,
    key_bid: torch.Tensor,
    radius: float,
    nbr_offsets: torch.Tensor,
    coord_min: torch.Tensor,
) -> torch.Tensor:
    """Return CUDA kids-by-offset table for sparse lattice radius neighbors.

    Args and returns match ``radius_nbr_kids_by_offset_sparse_lattice_cpu``.
    """
    ...
