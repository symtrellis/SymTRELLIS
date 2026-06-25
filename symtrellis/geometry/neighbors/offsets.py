import math

import torch


@torch.no_grad()
def lattice_ball_offsets(
    radius: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return integer offset candidates for radius neighbor search.

    Grid coordinates are integer cell centers. Query coordinates may be
    continuous in the same grid-index coordinate system.

    Args:
        radius: Search radius in grid-index units. Must be positive.
        device: Device for the returned tensor.

    Returns:
        nbr_offsets: Int32 tensor with shape [L, 3]. Each row is an integer
            offset from `floor(query)` to a candidate grid coordinate.
    """
    assert radius > 0.0

    R = int(math.ceil(radius))
    a = torch.arange(-R, R + 2, device=device, dtype=torch.int32)
    grid = torch.cartesian_prod(a, a, a)  # [L^3, 3]

    v = grid.to(torch.float32)
    d = torch.clamp(-v, min=0.0) + torch.clamp(v - 1.0, min=0.0)
    dist2 = (d * d).sum(dim=1)

    nbr_offsets = grid[dist2 <= radius * radius + 1e-12]
    return nbr_offsets
