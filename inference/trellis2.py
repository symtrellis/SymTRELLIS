"""TRELLIS.2-specific inference adapters.

This file contains only model/layout adapters needed to connect TRELLIS.2 to
the generic `symtrellis.flow` interfaces. Generic flow logic, CFG, and symmetry
projection guidance stay in `symtrellis.flow`.
"""

import gc
from typing import Any, Callable, Dict, Optional, Tuple, Type, cast

import cumesh
import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from flex_gemm.ops.grid_sample import grid_sample_3d
from PIL import Image
from trellis2.modules.sparse import SparseTensor
from trellis2.representations.mesh import Mesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from symtrellis.flow import AffineFlowStep, BaseFlowPredictor, BaseInitialNoiseSampler, SymmetryProjectionNoiseSampler
from symtrellis.mapper import SymmetryProjector

# Official TRELLIS.2 sampler defaults expressed in SymTRELLIS flow/CFG convention.
TRELLIS2_SPARSE_STRUCTURE_STEPS = 12
TRELLIS2_SPARSE_STRUCTURE_RESCALE_T = 5.0
TRELLIS2_SPARSE_STRUCTURE_CFG_STRENGTH = 7.5
TRELLIS2_SPARSE_STRUCTURE_CFG_INTERVAL = (0.0, 0.4)
TRELLIS2_SPARSE_STRUCTURE_CFG_RESCALE = 0.7
TRELLIS2_NOISE_LANCZOS_STEPS = 24
TRELLIS2_NOISE_SPECTRAL_FLOOR = 0.5

TRELLIS2_SHAPE_LATENT_STEPS = 12
TRELLIS2_SHAPE_LATENT_RESCALE_T = 3.0
TRELLIS2_SHAPE_LATENT_CFG_STRENGTH = 7.5
TRELLIS2_SHAPE_LATENT_CFG_INTERVAL = (0.0, 0.4)
TRELLIS2_SHAPE_LATENT_CFG_RESCALE = 0.5

TRELLIS2_TEXTURE_LATENT_STEPS = 12
TRELLIS2_TEXTURE_LATENT_RESCALE_T = 3.0
TRELLIS2_TEXTURE_LATENT_CFG_STRENGTH = 1.0
TRELLIS2_TEXTURE_LATENT_CFG_INTERVAL = (0.1, 0.4)
TRELLIS2_TEXTURE_LATENT_CFG_RESCALE = 0.0
TRELLIS2_PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}

# Per-channel de-normalization constants for TRELLIS.2 shape and texture sparse latents.
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

TRELLIS2_TEXTURE_LATENT_MEAN = [
     3.501659,  2.212398,  2.226094,  0.251093, -0.026248, -0.687364,  0.439898, -0.928075,
     0.029398, -0.339596, -0.869527,  1.038479, -0.972385,  0.126042, -1.129303,  0.455149,
    -1.209521,  2.069067,  0.544735,  2.569128, -0.323407,  2.293000, -1.925608, -1.217717,
     1.213905,  0.971588, -0.023631,  0.106750,  2.021786,  0.250524, -0.662387, -0.768862,
]
TRELLIS2_TEXTURE_LATENT_STD = [
     2.665652,  2.743913,  2.765121,  2.595319,  3.037293,  2.291316,  2.144656,  2.911822,
     2.969419,  2.501689,  2.154811,  3.163343,  2.621215,  2.381943,  3.186697,  3.021588,
     2.295916,  3.234985,  3.233086,  2.260140,  2.874801,  2.810596,  3.292720,  2.674999,
     2.680878,  2.372054,  2.451546,  2.353556,  2.995195,  2.379849,  2.786195,  2.775190,
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
                latent or TRELLIS.2 `SparseTensor` latent.
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


def coefficient_noise_rescale(
    noise_symm: torch.Tensor,
    projector: SymmetryProjector,
    symmetry_strength: float,
    self_include: bool,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Restore the expected row norm derived from the mixed projection coefficients."""
    coeff = projector.coeff
    source_rows = projector.rows_src[coeff.e_ids_src]
    destination_rows = projector.rows_dst[coeff.e_ids_dst]
    edge_pair_ids = destination_rows * projector.num_rows + source_rows
    diagonal_pair_ids = torch.arange(projector.num_rows, device=noise_symm.device) * (projector.num_rows + 1)
    pair_ids, pair_inverse = torch.unique(
        torch.cat([edge_pair_ids, diagonal_pair_ids]),
        sorted=True,
        return_inverse=True,
    )
    edge_pair_inverse = pair_inverse[: edge_pair_ids.shape[0]]

    pair_weights = noise_symm.new_zeros(pair_ids.shape[0])
    pair_weights.index_add_(0, edge_pair_inverse, coeff.w)
    pair_lowrank = noise_symm.new_zeros(pair_ids.shape[0], coeff.s.shape[1])
    pair_lowrank.index_add_(0, edge_pair_inverse, coeff.w[:, None] * coeff.s)

    pair_destination_rows = torch.div(pair_ids, projector.num_rows, rounding_mode="floor")
    pair_source_rows = pair_ids.remainder(projector.num_rows)
    counts = projector.counts_dst[pair_destination_rows]
    denominator = counts + 1.0 if self_include else counts.clamp_min(1.0)

    identity_coeff = symmetry_strength * pair_weights / denominator
    lowrank_coeff = symmetry_strength * pair_lowrank / denominator[:, None]
    diagonal_pairs = pair_destination_rows == pair_source_rows
    identity_coeff[diagonal_pairs] += 1.0 - symmetry_strength
    if self_include:
        identity_coeff[diagonal_pairs] += symmetry_strength / denominator[diagonal_pairs]

    left = coeff.Ut.T
    right = coeff.V.T
    if std is not None:
        channel_std = std.reshape(-1)
        left = channel_std[:, None] * left
        right = right / channel_std[None, :]

    trace_basis = (right * left.T).sum(dim=1)
    lowrank_gram = (left.T @ left) * (right @ right.T)
    lowrank_trace = lowrank_coeff @ trace_basis
    lowrank_norm = ((lowrank_coeff @ lowrank_gram) * lowrank_coeff).sum(dim=1)
    block_norm = identity_coeff.square() * noise_symm.shape[1] + 2.0 * identity_coeff * lowrank_trace + lowrank_norm

    output_variance = noise_symm.new_zeros(projector.num_rows)
    output_variance.index_add_(0, pair_destination_rows, block_norm)
    output_variance = output_variance / noise_symm.shape[1]
    scale = output_variance.clamp_min(torch.finfo(noise_symm.dtype).eps).rsqrt()[:, None]
    return noise_symm * scale


def lanczos_noise_rescale(
    noise_symm: torch.Tensor,
    projector: SymmetryProjector,
    symmetry_strength: float,
    self_include: bool,
    std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the clipped spectral correction of the mixed projection operator."""
    noise_norm = noise_symm.norm()
    if noise_norm == 0:
        return noise_symm

    basis = []
    diagonal = []
    off_diagonal = []
    previous_vector = torch.zeros_like(noise_symm)
    vector = noise_symm / noise_norm
    previous_beta = noise_symm.new_zeros(())
    eps = torch.finfo(noise_symm.dtype).eps

    for iteration in range(TRELLIS2_NOISE_LANCZOS_STEPS):
        basis.append(vector)

        if std is None:
            projected_transpose = projector.transposed_project(vector, self_include=self_include)
        else:
            projected_transpose = projector.transposed_project(vector / std, self_include=self_include) * std
        work = (1.0 - symmetry_strength) * vector + symmetry_strength * projected_transpose

        if std is None:
            projected = projector.forward_project(work, self_include=self_include)
        else:
            projected = projector.forward_project(work * std, self_include=self_include) / std
        work = (1.0 - symmetry_strength) * work + symmetry_strength * projected

        work = work - previous_beta * previous_vector
        alpha = torch.sum(vector * work)
        diagonal.append(alpha)
        work = work - alpha * vector
        for basis_vector in basis:
            work = work - torch.sum(basis_vector * work) * basis_vector

        if iteration + 1 == TRELLIS2_NOISE_LANCZOS_STEPS:
            break

        beta = work.norm()
        if beta <= eps:
            break
        off_diagonal.append(beta)
        previous_vector = vector
        vector = work / beta
        previous_beta = beta

    tridiagonal = torch.diag(torch.stack(diagonal).float())
    if off_diagonal:
        off_diagonal_tensor = torch.stack(off_diagonal).float()
        tridiagonal.diagonal(1).copy_(off_diagonal_tensor)
        tridiagonal.diagonal(-1).copy_(off_diagonal_tensor)

    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    singular_values = eigenvalues.clamp_min(0.0).sqrt()
    threshold = max(1.0 - symmetry_strength, TRELLIS2_NOISE_SPECTRAL_FLOOR)
    multipliers = torch.zeros_like(singular_values)
    nonzero = singular_values > torch.finfo(singular_values.dtype).eps
    multipliers[nonzero] = singular_values[nonzero].clamp(threshold, 1.0) / singular_values[nonzero]
    spectral_weights = eigenvectors @ (multipliers * eigenvectors[0])

    basis_tensor = torch.stack(basis)
    weight_shape = (spectral_weights.shape[0],) + (1,) * noise_symm.ndim
    corrected = (basis_tensor * spectral_weights.to(noise_symm.dtype).reshape(weight_shape)).sum(dim=0)
    return corrected * noise_norm


class TRELLIS2SparseStructureSymmetryProjectionNoiseSampler(SymmetryProjectionNoiseSampler):

    def __init__(
        self,
        sampler: BaseInitialNoiseSampler,
        symmetry_strength: float = 1.0,
        rescale_type: str = "lanczos",
        rescale_strength: float = 1.0,
    ) -> None:
        assert rescale_type in ("global", "voxel", "coefficient", "lanczos")

        super().__init__(
            sampler=sampler,
            symmetry_strength=symmetry_strength,
        )
        self.rescale_type = rescale_type
        self.rescale_strength = rescale_strength

    def sample(
        self,
        projector,
        to_sparse_view,
        to_original_view,
        self_include: bool,
        **kwargs,
    ):
        noise_native = self.sampler.sample(**kwargs)
        if self.symmetry_strength == 0.0:
            return noise_native

        projected_rows = projector.forward_project(
            feats=to_sparse_view(noise_native),
            self_include=self_include,
        )
        noise_projected = to_original_view(projected_rows)

        noise_symm = self.symmetry_strength * noise_projected + (1.0 - self.symmetry_strength) * noise_native

        if self.rescale_strength == 0.0:
            return noise_symm

        if self.rescale_type == "global":
            native_norm = noise_native.flatten(1).norm(dim=1)
            symm_norm = noise_symm.flatten(1).norm(dim=1)
            scale = native_norm / symm_norm.clamp_min(torch.finfo(noise_native.dtype).eps)
            scale = scale[:, None, None, None, None]
            noise_rescaled = noise_symm * scale
        elif self.rescale_type == "voxel":
            scale = noise_native.norm(dim=1, keepdim=True) / noise_symm.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(torch.finfo(noise_native.dtype).eps)
            noise_rescaled = noise_symm * scale
        elif self.rescale_type == "coefficient":
            noise_rescaled = to_original_view(
                coefficient_noise_rescale(
                    noise_symm=to_sparse_view(noise_symm),
                    projector=projector,
                    symmetry_strength=self.symmetry_strength,
                    self_include=self_include,
                )
            )
        else:
            noise_rescaled = to_original_view(
                lanczos_noise_rescale(
                    noise_symm=to_sparse_view(noise_symm),
                    projector=projector,
                    symmetry_strength=self.symmetry_strength,
                    self_include=self_include,
                )
            )

        return self.rescale_strength * noise_rescaled + (1.0 - self.rescale_strength) * noise_symm


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


def trellis2_occ_to_visualization_mesh(
    occ: torch.Tensor,
    y_up: bool = False,
) -> trimesh.Trimesh:
    """Convert dense occupancy `[R, R, R]` to a blocky boundary preview mesh."""
    grid_res = occ.shape[0]
    device = occ.device
    occ = occ.bool()

    occupied = torch.argwhere(occ)
    directions = torch.tensor(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        device=device,
        dtype=torch.long,
    )
    face_corners = torch.tensor(
        [
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ],
        device=device,
        dtype=torch.long,
    )

    neighbor = occupied[:, None, :] + directions[None, :, :]
    inside = ((neighbor >= 0) & (neighbor < grid_res)).all(dim=-1)
    clamped_neighbor = neighbor.clamp(0, grid_res - 1)
    neighbor_occ = occ[
        clamped_neighbor[..., 0],
        clamped_neighbor[..., 1],
        clamped_neighbor[..., 2],
    ]
    exposed = (~inside) | (~neighbor_occ)

    exposed_indices = torch.argwhere(exposed)
    voxel_ids = exposed_indices[:, 0]
    face_ids = exposed_indices[:, 1]

    corners = occupied[voxel_ids, None, :] + face_corners[face_ids]
    vertices = corners.to(torch.float32).reshape(-1, 3) / float(grid_res) - 0.5
    if y_up:
        z_up_to_y_up = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ],
            device=device,
            dtype=vertices.dtype,
        )
        vertices = vertices @ z_up_to_y_up.T

    face_base = (
        torch.arange(
            exposed_indices.shape[0],
            device=device,
            dtype=torch.long,
        )
        * 4
    )
    faces = torch.stack(
        [
            torch.stack([face_base + 0, face_base + 1, face_base + 2], dim=1),
            torch.stack([face_base + 0, face_base + 2, face_base + 3], dim=1),
        ],
        dim=1,
    ).reshape(-1, 3)

    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy().astype(np.float32),
        faces=faces.detach().cpu().numpy().astype(np.int64),
        process=False,
    )
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


class TRELLIS2SparseStructureView:
    """View adapter between TRELLIS.2 dense sparse-structure latent and sparse rows."""

    def __init__(
        self,
        coords: torch.Tensor,
        grid_size: int,
        batch_size: int,
    ) -> None:
        self.coords = coords
        self.grid_size = grid_size
        self.batch_size = batch_size

    def to_sparse_view(self, sparse_structure_latent: torch.Tensor) -> torch.Tensor:
        """Gather `[B, C, G, G, G]` latent features into `[N, C]` sparse rows."""
        return trellis2_sparse_structure_latent_to_sparse_view(
            sparse_structure_latent=sparse_structure_latent,
            coords=self.coords,
        )

    def to_original_view(self, sparse_view: torch.Tensor) -> torch.Tensor:
        """Scatter `[N, C]` sparse rows back to `[B, C, G, G, G]` latent layout."""
        return trellis2_sparse_view_to_sparse_structure_latent(
            sparse_view=sparse_view,
            coords=self.coords,
            grid_size=self.grid_size,
            batch_size=self.batch_size,
        )


class TRELLIS2SparseLatentNoiseSampler(BaseInitialNoiseSampler):
    """Sample TRELLIS.2 sparse latent noise.

    Shape and texture stages use sparse tensor objects with:

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


class TRELLIS2SparseLatentSymmetryProjectionNoiseSampler(SymmetryProjectionNoiseSampler):

    def __init__(
        self,
        sampler: BaseInitialNoiseSampler,
        symmetry_strength: float = 1.0,
        rescale_type: str = "lanczos",
        rescale_strength: float = 1.0,
    ) -> None:
        assert rescale_type in ("global", "voxel", "coefficient", "lanczos")

        super().__init__(
            sampler=sampler,
            symmetry_strength=symmetry_strength,
        )
        self.rescale_type = rescale_type
        self.rescale_strength = rescale_strength

    def sample(
        self,
        projector,
        self_include: bool,
        **kwargs,
    ):
        noise_native = self.sampler.sample(**kwargs)
        if self.symmetry_strength == 0.0:
            return noise_native

        std = noise_native.feats.new_tensor(TRELLIS2_SHAPE_LATENT_STD)[None]
        projected_vectors = projector.forward_project(
            feats=noise_native.feats * std,
            self_include=self_include,
        )
        projected_rows = projected_vectors / std

        noise_symm = self.symmetry_strength * projected_rows + (1.0 - self.symmetry_strength) * noise_native.feats

        if self.rescale_strength == 0.0:
            return noise_native.replace(noise_symm)

        if self.rescale_type == "global":
            batch_indices = noise_native.coords[:, 0].long()
            batch_count = int(batch_indices.max().item()) + 1

            native_norm = noise_symm.new_zeros(batch_count)
            symm_norm = noise_symm.new_zeros(batch_count)

            native_norm.scatter_add_(
                0,
                batch_indices,
                noise_native.feats.square().sum(dim=1),
            )
            symm_norm.scatter_add_(
                0,
                batch_indices,
                noise_symm.square().sum(dim=1),
            )

            scale = native_norm.sqrt() / symm_norm.sqrt().clamp_min(torch.finfo(noise_symm.dtype).eps)
            scale = scale[batch_indices, None]
            noise_rescaled = noise_symm * scale
        elif self.rescale_type == "voxel":
            scale = noise_native.feats.norm(
                dim=1,
                keepdim=True,
            ) / noise_symm.norm(
                dim=1,
                keepdim=True,
            ).clamp_min(torch.finfo(noise_symm.dtype).eps)
            noise_rescaled = noise_symm * scale
        elif self.rescale_type == "coefficient":
            noise_rescaled = coefficient_noise_rescale(
                noise_symm=noise_symm,
                projector=projector,
                symmetry_strength=self.symmetry_strength,
                self_include=self_include,
                std=std,
            )
        else:
            noise_rescaled = lanczos_noise_rescale(
                noise_symm=noise_symm,
                projector=projector,
                symmetry_strength=self.symmetry_strength,
                self_include=self_include,
                std=std,
            )

        noise = self.rescale_strength * noise_rescaled + (1.0 - self.rescale_strength) * noise_symm
        return noise_native.replace(noise)


TRELLIS2ShapeLatentNoiseSampler = TRELLIS2SparseLatentNoiseSampler


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


class TRELLIS2ShapeLatentView:
    """View adapter between TRELLIS.2 shape `SparseTensor` and mapper sparse rows."""

    def __init__(
        self,
        coords: torch.Tensor,
        sp_class: Type,
    ) -> None:
        self.coords = coords
        self.sp_class = sp_class

    def to_sparse_view(self, shape_latent) -> torch.Tensor:
        """Convert TRELLIS.2 normalized shape latent feats to `[N, C]` rows."""
        return trellis2_shape_latent_to_sparse_view(shape_latent)

    def to_original_view(self, sparse_view: torch.Tensor):
        """Convert `[N, C]` rows back to TRELLIS.2 normalized shape latent."""
        return trellis2_shape_sparse_view_to_latent(
            sparse_view=sparse_view,
            coords=self.coords,
            sp_class=self.sp_class,
        )


TRELLIS2TextureLatentNoiseSampler = TRELLIS2SparseLatentNoiseSampler


def trellis2_texture_latent_to_sparse_view(
    texture_latent,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
):
    """Convert TRELLIS.2 normalized texture latent feats to mapper feature space.

    Args:
        texture_latent: TRELLIS.2 sparse tensor with normalized `feats`.
        mean: Optional `[1, C]` or broadcastable per-channel mean.
        std: Optional `[1, C]` or broadcastable per-channel standard deviation.

    Returns:
        De-normalized sparse feature tensor with shape `[N, C]`.
    """
    device = texture_latent.feats.device
    dtype = texture_latent.feats.dtype
    if mean is None:
        mean = torch.tensor(TRELLIS2_TEXTURE_LATENT_MEAN, device=device, dtype=dtype)[None]
    if std is None:
        std = torch.tensor(TRELLIS2_TEXTURE_LATENT_STD, device=device, dtype=dtype)[None]

    return texture_latent.feats * std + mean


def trellis2_texture_sparse_view_to_latent(
    sparse_view: torch.Tensor,
    coords: torch.Tensor,
    sp_class: Type,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
):
    """Convert mapper feature space back to TRELLIS.2 texture sparse latent.

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
        mean = torch.tensor(TRELLIS2_TEXTURE_LATENT_MEAN, device=device, dtype=dtype)[None]
    if std is None:
        std = torch.tensor(TRELLIS2_TEXTURE_LATENT_STD, device=device, dtype=dtype)[None]

    return sp_class(feats=(sparse_view - mean) / std, coords=coords)


class TRELLIS2TextureLatentView:
    """View adapter between TRELLIS.2 texture `SparseTensor` and mapper sparse rows."""

    def __init__(
        self,
        coords: torch.Tensor,
        sp_class: Type,
    ) -> None:
        self.coords = coords
        self.sp_class = sp_class

    def to_sparse_view(self, texture_latent) -> torch.Tensor:
        """Convert TRELLIS.2 normalized texture latent feats to `[N, C]` rows."""
        return trellis2_texture_latent_to_sparse_view(texture_latent)

    def to_original_view(self, sparse_view: torch.Tensor):
        """Convert `[N, C]` rows back to TRELLIS.2 normalized texture latent."""
        return trellis2_texture_sparse_view_to_latent(
            sparse_view=sparse_view,
            coords=self.coords,
            sp_class=self.sp_class,
        )


def preprocess_image(
    image: Image.Image,
    rembg_model,
    target_size: int = 512,
    extend_scale: float = 1.05,
) -> Image.Image:

    has_alpha = False
    if image.mode == "RGBA":
        alpha = np.array(image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True

    # remove background
    if has_alpha:
        processed_image = image
    else:
        image = image.convert("RGB")
        processed_image = rembg_model(image)

    processed_image_np = np.array(processed_image)
    alpha = processed_image_np[:, :, 3]

    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)
    size = max(int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
    size = int(size * extend_scale)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2

    processed_image = processed_image.crop(bbox)
    max_size = max(processed_image.size)
    scale = min(1, target_size / max_size)
    if scale < 1:
        processed_image = processed_image.resize(
            (
                int(processed_image.width * scale),
                int(processed_image.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )
    processed_image = np.array(processed_image).astype(np.float32) / 255
    processed_image = processed_image[:, :, :3] * processed_image[:, :, 3:4]
    processed_image = Image.fromarray((processed_image * 255).astype(np.uint8))

    return processed_image


def trelli2_mesh_to_glb(
    shape_mesh: Mesh,
    res: int,
    device: torch.device,
    texture_size: Optional[int] = None,
    pbr_voxel: Optional[SparseTensor] = None,
    remesh: bool = True,
    decimation_target: int = 500000,
    remesh_band: float = 1.0,
    remesh_project: float = 0.0,
    report: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> trimesh.Trimesh:
    has_texture = pbr_voxel is not None
    if has_texture and remesh:
        total_steps = 13
    elif has_texture:
        total_steps = 15
    elif remesh:
        total_steps = 6
    else:
        total_steps = 8
    step = 0

    vertices = shape_mesh.vertices.detach().contiguous().float().to(device)
    faces = shape_mesh.faces.detach().contiguous().int().to(device)

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    del vertices, faces
    gc.collect()
    torch.cuda.empty_cache()

    step += 1
    if report is not None:
        report({"progress": step / total_steps, "stage": "prepare_mesh"})

    mesh.fill_holes(max_hole_perimeter=3e-2)

    source_vertices, source_faces = mesh.read()
    source_bvh = None
    if remesh or pbr_voxel is not None:
        source_bvh = cumesh.cuBVH(source_vertices, source_faces)

    step += 1
    if report is not None:
        report({"progress": step / total_steps, "stage": "source_mesh"})

    if remesh:
        resolution = int(res)
        center = torch.tensor([0.0, 0.0, 0.0], device=device)
        scale = 1.0

        remesh_vertices, remesh_faces = cumesh.remeshing.remesh_narrow_band_dc(
            source_vertices,
            source_faces,
            center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=False,
            bvh=source_bvh,
        )

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "remesh"})

        del mesh
        gc.collect()
        torch.cuda.empty_cache()

        mesh = cumesh.CuMesh()
        mesh.init(remesh_vertices, remesh_faces)

        del remesh_vertices, remesh_faces
        gc.collect()
        torch.cuda.empty_cache()

        mesh.simplify(decimation_target, verbose=False)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "simplify"})

    else:
        mesh.simplify(decimation_target * 3, verbose=False)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "simplify_rough"})

        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "cleanup_1"})

        mesh.simplify(decimation_target, verbose=False)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "simplify_target"})

        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)

        mesh.unify_face_orientations()

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "cleanup_2"})

    material = None
    out_uvs = None

    if pbr_voxel is None:
        mesh.compute_vertex_normals()
        out_vertices, out_faces = mesh.read()
        out_normals = mesh.read_vertex_normals()

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "compute_normals"})
    else:
        assert texture_size is not None
        assert source_bvh is not None
        texture_resolution = texture_size
        pbr_voxel = pbr_voxel.to(device)

        out_vertices, out_faces, out_uvs, out_vmaps = cast(
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            mesh.uv_unwrap(
                compute_charts_kwargs={
                    "threshold_cone_half_angle_rad": np.radians(90.0),
                    "refine_iterations": 0,
                    "global_iterations": 1,
                    "smooth_strength": 1,
                },
                return_vmaps=True,
                verbose=False,
            ),
        )

        out_vertices = out_vertices.to(device)
        out_faces = out_faces.to(device)
        out_uvs = out_uvs.to(device)
        out_vmaps = out_vmaps.to(device)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "uv_unwrap"})

        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps]

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "compute_normals"})

        ctx = dr.RasterizeCudaContext()
        uvs_rast = torch.cat(
            [
                out_uvs * 2 - 1,
                torch.zeros_like(out_uvs[:, :1]),
                torch.ones_like(out_uvs[:, :1]),
            ],
            dim=-1,
        ).unsqueeze(0)

        rast = torch.zeros(
            (1, texture_resolution, texture_resolution, 4),
            device=device,
            dtype=torch.float32,
        )

        for i in range(0, out_faces.shape[0], 100000):
            rast_chunk, _ = cast(
                Tuple[torch.Tensor, torch.Tensor],
                dr.rasterize(
                    ctx,
                    uvs_rast,
                    out_faces[i : i + 100000],
                    resolution=[texture_resolution, texture_resolution],
                ),
            )
            mask_chunk = rast_chunk[..., 3:4] > 0
            rast_chunk[..., 3:4] += i
            rast = torch.where(mask_chunk, rast_chunk, rast)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "rasterize_uv"})

        mask = rast[0, ..., 3] > 0

        pos = cast(Tuple[torch.Tensor, torch.Tensor], dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces))[0][0]

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "interpolate_positions"})

        valid_pos = pos[mask]
        _, face_id, uvw = source_bvh.unsigned_distance(valid_pos, return_uvw=True)
        assert uvw is not None
        source_tri_vertices = source_vertices[source_faces[face_id.long()]]
        valid_pos = (source_tri_vertices * uvw.unsqueeze(-1)).sum(dim=1)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "project_to_source_mesh"})

        attrs = torch.zeros(
            texture_resolution,
            texture_resolution,
            pbr_voxel.shape[1],
            device=device,
        )

        attrs[mask] = grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((valid_pos + 0.5) * res).reshape(1, -1, 3),
            mode="trilinear",
        )

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "sample_pbr"})

        mask_np = mask.cpu().numpy()
        mask_inv = (~mask_np).astype(np.uint8)

        base_color = np.clip(
            attrs[..., TRELLIS2_PBR_ATTR_LAYOUT["base_color"]].cpu().numpy() * 255,
            0,
            255,
        ).astype(np.uint8)
        metallic = np.clip(
            attrs[..., TRELLIS2_PBR_ATTR_LAYOUT["metallic"]].cpu().numpy() * 255,
            0,
            255,
        ).astype(np.uint8)
        roughness = np.clip(
            attrs[..., TRELLIS2_PBR_ATTR_LAYOUT["roughness"]].cpu().numpy() * 255,
            0,
            255,
        ).astype(np.uint8)
        alpha = np.clip(
            attrs[..., TRELLIS2_PBR_ATTR_LAYOUT["alpha"]].cpu().numpy() * 255,
            0,
            255,
        ).astype(np.uint8)

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "extract_pbr_channels"})

        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        material = PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode="OPAQUE",
            doubleSided=True if not remesh else False,
        )

        step += 1
        if report is not None:
            report({"progress": step / total_steps, "stage": "build_material"})

    z_up_to_y_up = out_vertices.new_tensor(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ],
    )
    out_vertices = out_vertices @ z_up_to_y_up.T
    out_normals = out_normals @ z_up_to_y_up.T

    vertices_np = out_vertices.cpu().numpy().copy()
    faces_np = out_faces.cpu().numpy()
    normals_np = out_normals.cpu().numpy().copy()

    visual = None
    if out_uvs is not None:
        out_uvs[:, 1] = 1 - out_uvs[:, 1]
        uvs_np = out_uvs.cpu().numpy().copy()
        visual = TextureVisuals(uv=uvs_np, material=material)

    step += 1
    if report is not None:
        report({"progress": step / total_steps, "stage": "finalize_mesh"})

    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=visual,
    )


__all__ = [
    "TRELLIS2FlowPredictor",
    "TRELLIS2_SPARSE_STRUCTURE_CFG_INTERVAL",
    "TRELLIS2_SPARSE_STRUCTURE_CFG_RESCALE",
    "TRELLIS2_SPARSE_STRUCTURE_CFG_STRENGTH",
    "TRELLIS2_SPARSE_STRUCTURE_RESCALE_T",
    "TRELLIS2_SPARSE_STRUCTURE_STEPS",
    "TRELLIS2SparseStructureLatentNoiseSampler",
    "TRELLIS2SparseStructureSymmetryProjectionNoiseSampler",
    "TRELLIS2SparseStructureView",
    "trellis2_dense_grid_coords",
    "trelli2_mesh_to_glb",
    "trellis2_occ_to_visualization_mesh",
    "trellis2_sparse_structure_latent_to_sparse_view",
    "trellis2_sparse_structure_logits_to_coords",
    "trellis2_sparse_view_to_sparse_structure_latent",
    "TRELLIS2SparseLatentNoiseSampler",
    "TRELLIS2SparseLatentSymmetryProjectionNoiseSampler",
    "TRELLIS2_SHAPE_LATENT_CFG_INTERVAL",
    "TRELLIS2_SHAPE_LATENT_CFG_RESCALE",
    "TRELLIS2_SHAPE_LATENT_CFG_STRENGTH",
    "TRELLIS2_SHAPE_LATENT_MEAN",
    "TRELLIS2_SHAPE_LATENT_RESCALE_T",
    "TRELLIS2_SHAPE_LATENT_STD",
    "TRELLIS2_SHAPE_LATENT_STEPS",
    "TRELLIS2ShapeLatentNoiseSampler",
    "TRELLIS2ShapeLatentView",
    "trellis2_shape_latent_to_sparse_view",
    "trellis2_shape_sparse_view_to_latent",
    "TRELLIS2_TEXTURE_LATENT_CFG_INTERVAL",
    "TRELLIS2_TEXTURE_LATENT_CFG_RESCALE",
    "TRELLIS2_TEXTURE_LATENT_CFG_STRENGTH",
    "TRELLIS2_TEXTURE_LATENT_MEAN",
    "TRELLIS2_TEXTURE_LATENT_RESCALE_T",
    "TRELLIS2_TEXTURE_LATENT_STD",
    "TRELLIS2_TEXTURE_LATENT_STEPS",
    "TRELLIS2TextureLatentNoiseSampler",
    "TRELLIS2TextureLatentView",
    "trellis2_texture_latent_to_sparse_view",
    "trellis2_texture_sparse_view_to_latent",
]
