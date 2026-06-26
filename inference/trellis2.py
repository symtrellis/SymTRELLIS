"""TRELLIS.2-specific inference adapters.

This file contains only model/layout adapters needed to connect TRELLIS.2 to
the generic `symtrellis.flow` interfaces. Generic flow logic, CFG, and symmetry
projection guidance stay in `symtrellis.flow`.
"""

from typing import Optional, Type

import torch

from symtrellis.flow import AffineFlowStep, BaseFlowPredictor, BaseInitialNoiseSampler

# Per-channel de-normalization constants for TRELLIS.2 shape sparse latents.
# Sparse-structure latents are used directly and do not use these constants.
# fmt: off
TRELLIS2_SHAPE_LATENT_MEAN = [
     0.781296,  0.018091, -0.495192, -0.558457,  1.060530,  0.093252,  1.518149, -0.933218,
    -0.732996,  2.604095, -0.118341, -2.143904,  0.495076, -2.179512, -2.130751, -0.996944,
     0.261421, -2.217463,  1.260067, -0.150213,  3.790713,  1.481266, -1.046058, -1.523667,
    -0.059621,  2.220780,  1.621212,  0.877230,  0.567247, -3.175944, -3.186688,  1.578665,
]
TRELLIS2_SHAPE_LATENT_STD = [
     5.972266,  4.706852,  5.445010,  5.209927,  5.320220,  4.547237,  5.020802,  5.444004,
     5.226681,  5.683095,  4.831436,  5.286469,  5.652043,  5.367606,  5.525084,  4.730578,
     4.805265,  5.124013,  5.530808,  5.619001,  5.103930,  5.417670,  5.269677,  5.547194,
     5.634698,  5.235274,  6.110351,  5.511298,  6.237273,  4.879207,  5.347008,  5.405691,
]
# fmt: on


class TRELLIS2FlowPredictor(BaseFlowPredictor):
    """Adapt a TRELLIS.2 flow model to the repository's velocity convention.

    TRELLIS.2 model time follows the diffusion convention where `0` denotes
    data and `1` denotes noise. `AffineFlowStep` uses the opposite convention:
    `t = 0` is noise and `t = 1` is data. This adapter flips the time and the
    predicted velocity sign so solvers can use one common flow convention.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the TRELLIS.2 model module."""
        self.model = model

    def predict_velocity(self, step: AffineFlowStep, cond=None, **kwargs):
        """Predict velocity in the `AffineFlowStep` convention.

        Args:
            step: Current flow state. `step.x_t` is either dense sparse-structure
                latent or TRELLIS.2 `SparseTensor` shape latent.
            cond: TRELLIS.2 conditioning tensor passed to the wrapped model.
            **kwargs: Extra TRELLIS.2 model arguments.

        Returns:
            Velocity with the same layout as `step.x_t`.
        """

        # Convert from SymTRELLIS flow time to TRELLIS.2 model time.
        t = 1 - step.t
        x_t = step.x_t

        # TRELLIS.2 models expect the diffusion timestep scaled to [0, 1000].
        t = torch.tensor(
            [1000 * t] * x_t.shape[0],
            device=x_t.device,
            dtype=torch.float32,
        )

        # Flip the sign to convert TRELLIS.2 velocity to SymTRELLIS velocity.
        v_pred = -self.model(x_t, t, cond, **kwargs)

        return v_pred


class TRELLIS2SparseStructureLatentNoiseSampler(BaseInitialNoiseSampler):
    """Sample dense TRELLIS.2 sparse-structure latent noise.

    The sparse-structure stage uses a dense latent tensor with layout
    `[batch_size, feat_dim, grid_size, grid_size, grid_size]`.
    """

    def sample(
        self,
        batch_size: int,
        grid_size: int,
        feat_dim: int,
        seed: int,
        device: str,
        **kwargs,
    ):
        """Return deterministic dense Gaussian noise for a fixed seed."""
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        noise = torch.randn(
            batch_size,
            feat_dim,
            grid_size,
            grid_size,
            grid_size,
            device=device,
            generator=g,
        )
        return noise


class TRELLIS2ShapeLatentNoiseSampler(BaseInitialNoiseSampler):
    """Sample TRELLIS.2 shape sparse latent noise.

    The shape stage uses a sparse tensor object with:

        feats: [num_coords, feat_dim]
        coords: [num_coords, 4], where columns are [batch, x, y, z]

    `sp_class` is expected to be TRELLIS.2's sparse tensor class.
    """

    def sample(
        self,
        sp_class: Type,
        coords: torch.Tensor,
        feat_dim: int,
        grid_size: int,
        seed: Optional[int],
        device: str,
        **kwargs,
    ):
        """Return a sparse latent sample in the provided `sp_class` container."""
        if seed is None:
            # Without a seed, sample each sparse row independently.
            feats = torch.randn(
                coords.shape[0],
                feat_dim,
                device=device,
            )
        else:
            # With a seed, sample a dense template and gather by sparse coords.
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            feats_template = torch.randn(
                grid_size,
                grid_size,
                grid_size,
                feat_dim,
                generator=g,
                device=device,
            )
            feats = feats_template[coords[:, 1], coords[:, 2], coords[:, 3]]

        return sp_class(feats=feats, coords=coords)


def trellis2_sparse_structure_latent_to_sparse_view(
    sparse_structure_latent: torch.Tensor,
    coords: torch.Tensor,
):
    """Gather dense sparse-structure latent features at sparse coordinates.

    Args:
        sparse_structure_latent: Dense tensor with shape `[B, C, G, G, G]`.
        coords: Integer tensor with shape `[N, 4]` and columns
            `[batch, x, y, z]`.

    Returns:
        Sparse feature view with shape `[N, C]`.
    """
    sparse_view = sparse_structure_latent.permute(0, 2, 3, 4, 1)
    sparse_view = sparse_view[
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        coords[:, 3],
    ]

    return sparse_view


def trellis2_sparse_view_to_sparse_structure_latent(
    sparse_view: torch.Tensor,
    coords: torch.Tensor,
    grid_size: int,
    batch_size: int,
):
    """Scatter sparse features back to dense sparse-structure latent layout.

    Args:
        sparse_view: Sparse features with shape `[N, C]`.
        coords: Integer tensor with shape `[N, 4]` and columns
            `[batch, x, y, z]`.
        grid_size: Dense latent resolution `G`.
        batch_size: Number of batch items `B`.

    Returns:
        Dense tensor with shape `[B, C, G, G, G]`.
    """
    device = sparse_view.device

    sparse_structure_latent = torch.empty(
        batch_size,
        grid_size,
        grid_size,
        grid_size,
        sparse_view.shape[-1],
        device=device,
    )

    sparse_structure_latent[
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        coords[:, 3],
    ] = sparse_view

    return sparse_structure_latent.permute(0, 4, 1, 2, 3)


def trellis2_shape_latent_to_sparse_view(
    shape_latent,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
):
    """Convert TRELLIS.2 normalized shape latent feats to mapper feature space.

    Args:
        shape_latent: TRELLIS.2 sparse tensor with normalized `feats`.
        mean: Optional `[1, C]` or broadcastable per-channel mean.
        std: Optional `[1, C]` or broadcastable per-channel standard deviation.

    Returns:
        De-normalized sparse feature tensor with shape `[N, C]`.
    """
    device = shape_latent.feats.device
    dtype = shape_latent.feats.dtype
    if mean is None:
        mean = torch.tensor(TRELLIS2_SHAPE_LATENT_MEAN, device=device, dtype=dtype)[None]
    if std is None:
        std = torch.tensor(TRELLIS2_SHAPE_LATENT_STD, device=device, dtype=dtype)[None]

    return shape_latent.feats * std + mean


def trellis2_shape_sparse_view_to_latent(
    sparse_view: torch.Tensor,
    coords: torch.Tensor,
    sp_class: Type,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
):
    """Convert mapper feature space back to TRELLIS.2 shape sparse latent.

    Args:
        sparse_view: De-normalized sparse features with shape `[N, C]`.
        coords: Sparse tensor coordinates with shape `[N, 4]`.
        sp_class: TRELLIS.2 sparse tensor class.
        mean: Optional `[1, C]` or broadcastable per-channel mean.
        std: Optional `[1, C]` or broadcastable per-channel standard deviation.

    Returns:
        `sp_class(feats=..., coords=coords)` with normalized feature values.
    """
    device = sparse_view.device
    dtype = sparse_view.dtype
    if mean is None:
        mean = torch.tensor(TRELLIS2_SHAPE_LATENT_MEAN, device=device, dtype=dtype)[None]
    if std is None:
        std = torch.tensor(TRELLIS2_SHAPE_LATENT_STD, device=device, dtype=dtype)[None]

    return sp_class(feats=(sparse_view - mean) / std, coords=coords)


def trellis2_dense_grid_coords(batch_size: int, grid_size: int, device: torch.device | str) -> torch.Tensor:
    """Build full dense-grid coordinates in `[batch, x, y, z]` format.

    This is used by the dense sparse-structure stage when every voxel in the
    latent grid is present and needs a row id for sparse-view projection.

    Returns:
        Int32 tensor with shape `[batch_size * grid_size**3, 4]`.
    """
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            torch.arange(grid_size, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    batch_ids = torch.arange(batch_size, device=device)[:, None].expand(batch_size, grid.shape[0]).reshape(-1, 1)
    grid = grid.repeat(batch_size, 1)

    return torch.cat([batch_ids, grid], dim=1).to(dtype=torch.int32)


def trellis2_sparse_structure_logits_to_coords(
    logits: torch.Tensor,
    target_resolution: int,
) -> torch.Tensor:
    """Convert sparse-structure decoder logits to shape-stage coordinates.

    Args:
        logits: Occupancy logits with shape `[B, 1, G, G, G]`.
        target_resolution: Target sparse shape-latent grid resolution.

    Returns:
        Int32 coordinates `[N, 4]` with columns `[batch, x, y, z]`.
    """
    occ = logits > 0
    pool_size = logits.shape[-1] // target_resolution
    occ = torch.nn.functional.max_pool3d(occ.float(), pool_size, pool_size, 0) > 0.5

    return torch.argwhere(occ)[:, [0, 2, 3, 4]].to(dtype=torch.int32)


__all__ = [
    "TRELLIS2_SHAPE_LATENT_MEAN",
    "TRELLIS2_SHAPE_LATENT_STD",
    "TRELLIS2FlowPredictor",
    "TRELLIS2ShapeLatentNoiseSampler",
    "TRELLIS2SparseStructureLatentNoiseSampler",
    "trellis2_dense_grid_coords",
    "trellis2_shape_latent_to_sparse_view",
    "trellis2_shape_sparse_view_to_latent",
    "trellis2_sparse_structure_latent_to_sparse_view",
    "trellis2_sparse_structure_logits_to_coords",
    "trellis2_sparse_view_to_sparse_structure_latent",
]
