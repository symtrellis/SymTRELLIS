"""Dense lookup backend for sparse lattice radius neighbor search."""

import torch


def radius_nbr_edges_dense_lattice(
    query_pos: torch.Tensor,
    query_bid: torch.Tensor,
    key_coords: torch.Tensor,
    key_bid: torch.Tensor,
    radius: float,
    nbr_offsets: torch.Tensor,
    grid_size: int = 64,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find radius neighbors using a dense lattice lookup table.

    This backend builds a dense lookup tensor with shape
    [B, grid_size ** 3], so it is intended for small cubic grids. `key_coords`
    must be integer lattice coordinates in [0, grid_size - 1]. Duplicate key
    coordinates within the same batch are not supported.

    Args:
        query_pos: Float tensor with shape [Nq, 3]. Continuous query positions
            in grid-index units.
        query_bid: Integer tensor with shape [Nq]. Batch id for each query row.
        key_coords: Integer tensor with shape [Nk, 3]. Sparse lattice
            coordinates in [0, grid_size - 1].
        key_bid: Integer tensor with shape [Nk]. Batch id for each key row.
            Unique batch ids must match `query_bid`.
        radius: Search radius in grid-index units.
        nbr_offsets: Integer tensor with shape [L, 3], usually returned by
            `lattice_ball_offsets(radius)`.
        grid_size: Cubic lattice size. Dense memory cost is
            O(B * grid_size ** 3).
        chunk_size: Number of query rows processed per chunk.

    Returns:
        qids: Int64 tensor with shape [E]. Query row index for each edge.
        kids: Int64 tensor with shape [E]. Key row index for each edge.
    """
    uq = query_bid.unique()
    uk = key_bid.unique()
    assert uq.shape == uk.shape
    assert (uq == uk).all()

    G = grid_size
    G2 = G * G
    G3 = G * G2
    radius2 = radius * radius

    device = query_pos.device
    Nq = query_pos.shape[0]
    Nk = key_coords.shape[0]
    B = int(key_bid.max().item()) + 1

    # Build a dense hash table from batch/grid coordinate to key row index.
    lookup = torch.full((B, G3), -1, dtype=torch.int64, device=device)
    glin = key_coords[:, 0] + G * key_coords[:, 1] + G2 * key_coords[:, 2]
    lookup[key_bid, glin] = torch.arange(Nk, device=device, dtype=torch.int64)

    qids_list = []
    kids_list = []

    for s in range(0, Nq, chunk_size):
        e = min(s + chunk_size, Nq)

        # Enumerate candidate key coordinates around each query by integer offsets.
        q = query_pos[s:e]  # [M, 3]
        q_bid = query_bid[s:e]  # [M]
        q_base = torch.floor(q).to(dtype=key_coords.dtype)  # [M, 3]
        cand = q_base[:, None, :] + nbr_offsets[None, :, :]  # [M, L, 3]

        # Keep candidates inside the grid and present in the sparse key set.
        cx, cy, cz = cand.unbind(dim=-1)
        in_range = (cx >= 0) & (cx < G) & (cy >= 0) & (cy < G) & (cz >= 0) & (cz < G)
        clin = cx + G * cy + G2 * cz
        clin_safe = clin.clamp(0, G3 - 1)

        cand_id = lookup[q_bid[:, None], clin_safe]  # [M, L]
        ok = in_range & (cand_id >= 0)

        # Apply the true radius test.
        dist2 = ((cand - q[:, None, :]) ** 2).sum(dim=-1)  # [M, L]
        ok = ok & (dist2 <= radius2)

        edge = torch.argwhere(ok)
        if edge.numel() == 0:
            continue

        # Flatten valid candidate entries into edge lists.
        qids = edge[:, 0] + s
        kids = cand_id[ok]
        qids_list.append(qids)
        kids_list.append(kids)

    if not qids_list:
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        return empty, empty

    qids = torch.cat(qids_list, dim=0)
    kids = torch.cat(kids_list, dim=0)
    return qids, kids
