import torch


def pos2grid(pos: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Quantize normalized positions to integer grid coordinates.

    This is the standard latent-grid convention used by the supported
    voxel-based generation models: integer grid coordinates index voxel
    centers in a normalized canonical cube.

    Convention: grid index `i` denotes the cell center at
    `(i + 0.5) / grid_size - 0.5` in the fixed domain [-0.5, 0.5].
    When adapting to a new voxel-based model, verify that its latent grid uses
    the same voxel-center offset and normalized coordinate range.

    Args:
        pos: Float tensor with shape [..., 3]. Coordinates are in the
            normalized cube [-0.5, 0.5].
        grid_size: Number of grid cells per axis. This function assumes a
            cubic grid.

    Returns:
        Long tensor with shape [..., 3]. Values are clamped to
        [0, grid_size - 1].
    """
    xyz = torch.round((pos + 0.5) * grid_size - 0.5).long()
    xyz = xyz.clamp(0, grid_size - 1)

    return xyz


def grid2pos(grid: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Convert integer grid coordinates to normalized cell-center positions.

    This is the standard latent-grid convention used by the supported
    voxel-based generation models: integer grid coordinates index voxel
    centers in a normalized canonical cube.
    When adapting to a new voxel-based model, verify that its latent grid uses
    the same voxel-center offset and normalized coordinate range.

    Args:
        grid: Integer tensor with shape [..., 3]. Coordinates index grid cells
            in [0, grid_size - 1].
        grid_size: Number of grid cells per axis. This function assumes a
            cubic grid.

    Returns:
        Float tensor with shape [..., 3]. Coordinates are cell centers in the
        normalized cube [-0.5, 0.5].
    """
    return (grid.float() + 0.5) / grid_size - 0.5


def t_abs2grid(
    t_abs: torch.Tensor,
    O: torch.Tensor,
    grid_size: int,
) -> torch.Tensor:
    """Convert a normalized-position translation to grid-index translation.

    The returned translation is compatible with transforms applied directly to
    integer grid coordinates:

        grid_out = grid_in @ O.T + t_grid

    under the position convention used by `grid2pos`:

        pos = (grid + 0.5) / grid_size - 0.5

    Args:
        t_abs: Float tensor with shape [..., 3]. Translation in normalized
            position units.
        O: Float tensor with shape [..., 3, 3]. Orthogonal transform in the
            same direction as `t_abs`.
        grid_size: Number of grid cells per axis. This function assumes the
            source and destination grids share the same cubic resolution.

    Returns:
        Float tensor with shape [..., 3]. Translation in grid-index units.
    """
    coeff = (grid_size - 1) * 0.5
    t_grid = t_abs * grid_size + coeff * (1.0 - O.sum(dim=-1))

    return t_grid
