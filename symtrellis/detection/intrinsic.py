"""Intrinsic symmetry scoring on a sampled mesh.

This module builds a reusable intrinsic basis on a fixed mesh and uses it to
score many candidate transforms.

.. math::
    C = (G \\Phi)^T (P_T \\Phi), \\qquad
    e_k = \\lVert C^k - I \\rVert_F

Code symbols:
    `samples` -> sampled points :math:`p_i`
    `phi` -> sampled intrinsic eigenfunctions :math:`\\Phi`
    `Gphi` -> reduced mass product :math:`G\\Phi`
    `coeff` -> intrinsic transform operator :math:`C`
"""

import math
from typing import Tuple, Union

import torch
from pytorch3d.ops import knn_points


def laplace_beltrami_operator(
    verts: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the cotangent Laplace-Beltrami operator and lumped vertex mass.

    Purpose:
        Build the mesh-space cotangent stiffness matrix and lumped mass used by
        the reduced intrinsic basis.

    .. math::
        L_{ij} = -\\frac{1}{2} \\cot\\theta_{ij}, \\qquad
        L_{ii} = \\frac{1}{2} \\sum_j \\cot\\theta_{ij}

    .. math::
        m_i = \\sum_{f \\ni i} \\frac{\\operatorname{area}(f)}{3}

    Code symbols:
        `L` -> sparse cotangent stiffness matrix :math:`L \\in R^{N \\times N}`
        `mass` -> lumped vertex area vector :math:`m \\in R^N`
        `dbl_area` -> twice the triangle area :math:`2A_f`
        `cot_i`, `cot_j`, `cot_k` -> triangle corner cotangents

    Args:
        verts: Tensor with shape [N, 3].
        faces: Long tensor with shape [F, 3].
        eps: Minimum area and mass clamp.

    Returns:
        L: Sparse COO tensor with shape [N, N].
        mass: Tensor with shape [N].
    """
    num_verts = verts.shape[0]
    device = verts.device
    dtype = verts.dtype

    i, j, k = faces.unbind(dim=-1)
    vi, vj, vk = verts[i], verts[j], verts[k]

    e_ij = vj - vi
    e_ik = vk - vi
    e_jk = vk - vj

    dbl_area = torch.cross(e_ij, e_ik, dim=1).norm(dim=1).clamp_min(eps)

    cot_i = (e_ij * e_ik).sum(dim=1) / dbl_area
    cot_j = -(e_ij * e_jk).sum(dim=1) / dbl_area
    cot_k = (e_ik * e_jk).sum(dim=1) / dbl_area

    rows_w = torch.cat([i, j, i, k, j, k], dim=0)
    cols_w = torch.cat([j, i, k, i, k, j], dim=0)
    vals_w = torch.cat([cot_k, cot_k, cot_j, cot_j, cot_i, cot_i], dim=0)

    degree = torch.zeros(num_verts, dtype=dtype, device=device)
    degree.scatter_add_(0, rows_w, vals_w)

    diag_idx = torch.arange(num_verts, device=device, dtype=torch.long)

    rows_L = torch.cat([rows_w, diag_idx], dim=0)
    cols_L = torch.cat([cols_w, diag_idx], dim=0)
    vals_L = torch.cat([-0.5 * vals_w, 0.5 * degree], dim=0)

    L = torch.sparse_coo_tensor(
        torch.stack([rows_L, cols_L], dim=0),
        vals_L,
        (num_verts, num_verts),
        device=device,
        dtype=dtype,
    ).coalesce()

    third_area = 0.5 * dbl_area / 3.0

    mass = torch.zeros(num_verts, dtype=dtype, device=device)
    mass.scatter_add_(0, i, third_area)
    mass.scatter_add_(0, j, third_area)
    mass.scatter_add_(0, k, third_area)

    mass = mass.clamp_min(eps)

    return L, mass


def sample_coverage_radius(
    samples: torch.Tensor,
    target_overlap: float = 8.0,
    k_rho: int = 8,
) -> torch.Tensor:
    """Estimate the compact-support radius for vertex-to-sample projection.

    Purpose:
        Estimate the projection support radius from local sample spacing.

    .. math::
        \\rho = \\operatorname{median}(d_{\\text{knn}})\\,
        \\sqrt{\\frac{\\text{target_overlap}}{\\pi}}

    Code symbols:
        `radius` -> support radius :math:`\\rho`
        `sample_knn_dist` -> local sample-sample KNN distances
        `median_sample_knn_dist` -> :math:`\\operatorname{median}(d_{\\text{knn}})`

    Args:
        samples: Tensor with shape [M, 3].
        target_overlap: Desired approximate number of overlapping sample disks.
        k_rho: Number of sample neighbors used to estimate local spacing.

    Returns:
        radius: Scalar tensor.
    """
    # Approximate local sample spacing by sample-sample KNN distances.
    sample_knn_dist2, _, _ = knn_points(samples[None], samples[None], K=k_rho)
    sample_knn_dist = torch.sqrt(sample_knn_dist2[0, :, 1:])
    median_sample_knn_dist = sample_knn_dist.median()

    radius = median_sample_knn_dist * math.sqrt(target_overlap / math.pi)

    return radius


def downsample_conversion_matrix(
    verts: torch.Tensor,
    samples: torch.Tensor,
    radius: Union[float, torch.Tensor],
    k_proj: int = 32,
) -> torch.Tensor:
    """Build the sparse vertex-to-sample conversion matrix.

    Purpose:
        Transfer vertex-space quantities to the sampled space with a compact
        cubic kernel.

    .. math::
        r_{ij} = \\frac{\\lVert v_i - p_j \\rVert}{\\rho}

    .. math::
        \\tilde U_{ij} =
        \\begin{cases}
        2r_{ij}^3 - 3r_{ij}^2 + 1, & r_{ij} \\le 1 \\\\
        0, & r_{ij} > 1
        \\end{cases}

    Code symbols:
        `downsample_matrix` -> conversion matrix :math:`\\tilde U \\in R^{N \\times M}`
        `radius` -> support radius :math:`\\rho`
        `r` -> normalized vertex-sample distance :math:`r_{ij}`
        `vals` -> nonzero projection weights :math:`\\tilde U_{ij}`

    Args:
        verts: Tensor with shape [N, 3].
        samples: Tensor with shape [M, 3].
        radius: Projection support radius rho.
        k_proj: Number of nearest samples considered per vertex.

    Returns:
        downsample_matrix: Sparse COO tensor with shape [N, M].
    """
    num_verts, num_samples = verts.shape[0], samples.shape[0]
    device = verts.device
    dtype = verts.dtype

    # compute from vertices to samples
    dist2, nn_idx, _ = knn_points(
        verts[None],  # [1, N, 3]
        samples[None],  # [1, M, 3]
        K=k_proj,
        return_nn=False,
    )

    dist = torch.sqrt(dist2[0])  # [N, K]
    nn_idx = nn_idx[0]  # [N, K]

    radius_mask = dist <= radius

    rows = torch.arange(num_verts, device=device)[:, None].expand_as(nn_idx)[radius_mask]
    cols = nn_idx[radius_mask]

    r = dist[radius_mask] / radius
    vals = 2.0 * r**3 - 3.0 * r**2 + 1.0

    downsample_matrix = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        vals,
        (num_verts, num_samples),
        device=device,
        dtype=dtype,
    ).coalesce()

    return downsample_matrix


def dense_rbf_matrix(
    samples: torch.Tensor,
    k_local: int = 24,
) -> torch.Tensor:
    """Build a dense adaptive RBF affinity matrix on samples.

    Purpose:
        Build a dense sample-sample affinity used to regularize the reduced
        intrinsic operator.

    .. math::
        K_{ij} = \\exp\\left(
            -\\frac{\\lVert p_i - p_j \\rVert^2}{4\\sigma_i\\sigma_j}
        \\right)

    Code symbols:
        `rbf_matrix` -> adaptive RBF affinity :math:`K`
        `sigma` -> local bandwidth :math:`\\sigma_i`
        `dist2` -> squared sample distance :math:`\\lVert p_i - p_j \\rVert^2`

    Args:
        samples: Tensor with shape [M, 3].
        k_local: Neighbor rank used to estimate adaptive bandwidth.

    Returns:
        rbf_matrix: Dense tensor with shape [M, M].
    """
    knn = knn_points(samples[None], samples[None], K=k_local + 1, return_nn=False)
    knn_d2 = knn.dists[0]  # [M, k_local + 1]
    sigma = knn_d2[:, -1].sqrt().clamp_min(1e-6)

    sample_norm2 = (samples * samples).sum(dim=1)
    dist2 = (sample_norm2[:, None] + sample_norm2[None, :] - 2.0 * (samples @ samples.T)).clamp_min(0.0)

    denom = (4.0 * sigma[:, None] * sigma[None, :]).clamp_min(1e-12)
    rbf_matrix = torch.exp(-dist2 / denom)
    rbf_matrix.fill_diagonal_(0.0)

    return rbf_matrix


def build_intrinsic_basis(
    verts: torch.Tensor,
    faces: torch.Tensor,
    samples: torch.Tensor,
    intrinsic_dim: int,
    mu: float = 0.15,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reusable intrinsic basis data for transform scoring.

    Purpose:
        Construct the reduced intrinsic eigenspace on samples and return only
        the data needed for repeated transform scoring.

    .. math::
        G = \\tilde U^T M \\tilde U

    .. math::
        S^T G S = I

    .. math::
        L_{mix} =
        (1 - \\mu)\\frac{L_{cot}^{white}}{\\operatorname{tr}(L_{cot}^{white})}
        + \\mu\\frac{L_{rbf}^{white}}{\\operatorname{tr}(L_{rbf}^{white})}

    .. math::
        \\Phi = S Z

    Code symbols:
        `samples` -> sampled points :math:`p_i`
        `downsample_matrix` -> :math:`\\tilde U`
        `G` -> reduced mass matrix :math:`G`
        `whitening_matrix` -> whitening basis :math:`S`
        `eigvecs` -> eigenvectors :math:`Z`
        `phi` -> sampled eigenfunctions :math:`\\Phi`
        `Gphi` -> reduced mass product :math:`G\\Phi`

    Args:
        verts: Tensor with shape [N, 3].
        faces: Long tensor with shape [F, 3].
        samples: Tensor with shape [M, 3]. Its row order defines the indexing
            used later by `intrinsic_fold_error`.
        intrinsic_dim: Number of non-constant eigenfunctions to keep.
        mu: Mixture weight between cotangent and RBF Laplacians.
        eps: Numerical clamp used by normalization and whitening.

    Returns:
        samples: Tensor with shape [M, 3].
        phi: Tensor with shape [M, intrinsic_dim].
        Gphi: Tensor with shape [M, intrinsic_dim].
    """
    L, mass = laplace_beltrami_operator(verts, faces)
    radius = sample_coverage_radius(samples)
    downsample_matrix = downsample_conversion_matrix(verts, samples, radius)
    rbf_matrix = dense_rbf_matrix(samples)

    device = mass.device
    dtype = mass.dtype
    num_samples = downsample_matrix.shape[1]

    idx = downsample_matrix.indices()
    vals = downsample_matrix.values()
    rows = idx[0]  # vertex ids
    cols = idx[1]  # sample ids

    # sample_mass is the reduced lumped mass, w = U_bar^T mass.
    row_sum = torch.sparse.sum(downsample_matrix, dim=1).to_dense().clamp_min(eps)
    vals_bar = vals / row_sum[rows]

    sample_mass = torch.zeros(num_samples, device=device, dtype=dtype)
    sample_mass.scatter_add_(0, cols, mass[rows] * vals_bar)
    sample_mass = sample_mass.clamp_min(eps)

    # normalized_rbf is the density-normalized sample affinity K_hat.
    rbf_norm = (rbf_matrix @ sample_mass).clamp_min(eps)
    normalized_rbf = rbf_matrix / (rbf_norm[:, None] * rbf_norm[None, :]).clamp_min(eps)
    normalized_rbf = 0.5 * (normalized_rbf + normalized_rbf.T)
    normalized_rbf.fill_diagonal_(0.0)

    sqrt_sample_mass = sample_mass.sqrt()
    weighted_rbf = normalized_rbf * (sqrt_sample_mass[:, None] * sqrt_sample_mass[None, :])
    weighted_rbf = 0.5 * (weighted_rbf + weighted_rbf.T)
    weighted_rbf.fill_diagonal_(0.0)

    rbf_degree = weighted_rbf.sum(dim=1)
    L_rbf = torch.diag(rbf_degree) - weighted_rbf
    L_rbf = 0.5 * (L_rbf + L_rbf.T)

    # G is the reduced mass matrix, G = U_tilde^T M U_tilde.
    mass_weighted_downsample_matrix = torch.sparse_coo_tensor(
        idx,
        vals * mass[rows],
        downsample_matrix.shape,
        device=device,
        dtype=dtype,
    ).coalesce()

    G = (downsample_matrix.transpose(0, 1) @ mass_weighted_downsample_matrix).to_dense()
    G = 0.5 * (G + G.T)

    # L_cot is the reduced cotangent stiffness, U_tilde^T L U_tilde.
    LU = L @ downsample_matrix
    L_cot = (downsample_matrix.transpose(0, 1) @ LU).to_dense()
    L_cot = 0.5 * (L_cot + L_cot.T)

    # whitening_matrix is S, chosen so S^T G S = I.
    mass_eigvals, mass_eigvecs = torch.linalg.eigh(G)

    # Drop near-null mass directions before whitening.
    keep = mass_eigvals >= torch.minimum(
        eps * mass_eigvals.max().clamp_min(1.0),
        mass_eigvals[-min(256, mass_eigvals.numel())],
    )
    mass_eigvals = mass_eigvals[keep]
    mass_eigvecs = mass_eigvecs[:, keep]
    whitening_matrix = mass_eigvecs / torch.sqrt(mass_eigvals)[None, :]  # [m, r]

    L_cot_white = whitening_matrix.T @ L_cot @ whitening_matrix
    L_rbf_white = whitening_matrix.T @ L_rbf @ whitening_matrix

    L_cot_white = 0.5 * (L_cot_white + L_cot_white.T)
    L_rbf_white = 0.5 * (L_rbf_white + L_rbf_white.T)

    # L_mix is the final trace-normalized reduced intrinsic operator.
    trace_cot = torch.trace(L_cot_white).clamp_min(eps)
    trace_rbf = torch.trace(L_rbf_white).clamp_min(eps)

    L_cot_norm = L_cot_white / trace_cot
    L_rbf_norm = L_rbf_white / trace_rbf

    L_mix = (1.0 - mu) * L_cot_norm + mu * L_rbf_norm
    L_mix = 0.5 * (L_mix + L_mix.T)

    # eigvecs is Z, the eigenbasis of the whitened reduced operator.
    _, eigvecs = torch.linalg.eigh(L_mix)

    # phi is Phi = S Z, then the constant eigenfunction is skipped.
    phi = whitening_matrix @ eigvecs  # [M, r]
    phi = phi[:, 1 : 1 + intrinsic_dim]
    Gphi = G @ phi

    return samples, phi, Gphi


def intrinsic_fold_error(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    max_fold: int = 15,
) -> torch.Tensor:
    """Score how close candidate transforms are to finite intrinsic symmetries.

    Purpose:
        Estimate the intrinsic operator induced by each candidate transform and
        measure how close its powers are to identity.

    .. math::
        T(x) = x R^T + t

    .. math::
        C = (G\\Phi)^T(P_T\\Phi)

    .. math::
        e_k = \\lVert C^k - I \\rVert_F, \\qquad k = 2,\\ldots,K

    Code symbols:
        `samples` -> sampled points :math:`p_i`
        `phi` -> sampled eigenfunctions :math:`\\Phi`
        `Gphi` -> reduced mass product :math:`G\\Phi`
        `R`, `t` -> candidate transform :math:`T`
        `map_idx` -> nearest-neighbor sample map :math:`P_T`
        `coeff` -> intrinsic operator :math:`C`
        `intrinsic_errs` -> fold errors :math:`e_k`

    Args:
        samples: Tensor with shape [M, 3]. Must use the same row order used to
            build `phi` and `Gphi`.
        phi: Tensor with shape [M, D].
        Gphi: Tensor with shape [M, D].
        R: Tensor with shape [B, 3, 3].
        t: Tensor with shape [B, 3].
        max_fold: Maximum fold order to score.

    Returns:
        intrinsic_errs: Tensor with shape [B, max_fold - 1]. Column 0 scores
            order 2, column 1 scores order 3, and so on.
    """
    batch_size = R.shape[0]
    intrinsic_dim = phi.shape[1]

    X = samples[None].expand(batch_size, -1, -1)
    Y = X @ R.transpose(1, 2) + t[:, None, :]

    knn = knn_points(Y, X, K=1)
    map_idx = knn.idx[..., 0]
    phi_map = phi[map_idx]

    coeff = torch.matmul(Gphi.T, phi_map)  # [B, D, D]
    coeff_u, _, coeff_vh = torch.linalg.svd(coeff)
    coeff_ortho = coeff_u @ coeff_vh

    # intrinsic_errs[:, k - 2] stores ||C^k - I||_F for k >= 2.
    I = torch.eye(intrinsic_dim, device=coeff.device, dtype=coeff.dtype).expand(batch_size, -1, -1)
    intrinsic_errs = torch.empty(batch_size, max_fold - 1, device=coeff.device, dtype=coeff.dtype)

    coeff_power = coeff_ortho @ coeff_ortho  # C^2
    for i in range(max_fold - 1):
        diff = coeff_power - I
        intrinsic_errs[:, i] = torch.linalg.matrix_norm(diff, ord="fro", dim=(-2, -1))

        if i + 2 < max_fold:
            coeff_power = coeff_power @ coeff_ortho

    return intrinsic_errs
