"""ICP-based symmetry detectors on sampled mesh geometry.

The detector functions operate on precomputed surface samples and intrinsic
basis data. They estimate candidate extrinsic transforms with ICP, filter them
with geometric constraints, cluster equivalent candidates, and use intrinsic
fold errors as a consistency score.
"""

import math
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
from pytorch3d.structures import Meshes
from sklearn.cluster import DBSCAN

from symtrellis.symmetry.transforms import axis_point_rotation_transform, plane_reflection_transform

from .fitting import (
    acos_dist,
    fit_direction,
    fit_reflection_plane,
    get_c2_axis,
    get_reflection_plane,
    get_rotation_axis_angle,
    get_rotation_axis_point,
)
from .icp import iterative_closest_point
from .intrinsic import intrinsic_fold_error
from .sampling import (
    sample_axis_parallel_reflections,
    sample_axis_perp_c2_rotations,
    sample_axis_perp_reflections,
    sample_mesh_farthest_points,
    sample_random_rotations,
)


def gram_schmidt_frame(axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build an orthonormal frame for the plane perpendicular to an axis.

    This returns two unit vectors spanning `axis^perp`. It is used whenever
    candidate directions around a known axis need to be represented by planar
    angles.

    Computation:
        1. Pick a seed axis that is not nearly parallel to `axis`.
        2. Project the seed into `axis^perp` and normalize it as `e1`.
        3. Compute `e2 = axis x e1` and normalize it.

    Args:
        axis: Tensor with shape [3]. Expected to be non-zero.

    Returns:
        e1: Tensor with shape [3].
        e2: Tensor with shape [3].
    """
    device = axis.device
    dtype = axis.dtype

    seed_axis = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    if torch.abs((seed_axis * axis).sum()) > 0.9:
        seed_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    e1 = seed_axis - (seed_axis * axis).sum() * axis
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(axis, e1)
    e2 = e2 / e2.norm()

    return e1, e2


def estimate_fold_from_errors(
    errors: torch.Tensor,  # [num_errors, num_folds],
    fold_min: int = 2,
    norm_p: float = 4.0,
    min_n_delta: float = 1e-3,
):
    """Estimate a symmetry fold from per-candidate fold errors.

    The input stores one error row per candidate transform and one column per
    possible fold order. The function returns the smallest fold whose aggregate
    score is close enough to the best score.

    Computation:
        1. Clamp errors to non-negative values and compute a p-mean score per
           fold.
        2. Find the best scoring fold.
        3. Use the variation of the best fold score across candidates to set
           an adaptive tolerance.
        4. Return the smallest fold within the tolerated score range.

    Args:
        errors: Tensor with shape [num_errors, num_folds].
        fold_min: Fold represented by column 0.
        norm_p: Power used by the p-mean aggregation.
        min_n_delta: Minimum relative tolerance around the best score.

    Returns:
        Fold order as a Python int.
    """
    e = errors.clamp_min(0.0)
    error_power = e.pow(norm_p)
    mean_error_power = error_power.mean(dim=0)
    fold_score = mean_error_power.pow(1.0 / norm_p)

    best_fold_idx = int(torch.argmin(fold_score).item())
    best_score = fold_score[best_fold_idx]

    num_errors = errors.shape[0]
    if num_errors > 1:
        best_error_power = error_power[:, best_fold_idx]  # [num_errors]
        best_mean_error_power = best_error_power.mean()
        best_std_error_power = best_error_power.std(unbiased=True)
        best_coeff_var = best_std_error_power / best_mean_error_power.clamp_min(1e-12)
        adaptive_delta = (best_coeff_var / (math.sqrt(num_errors) * norm_p)).item()
    else:
        adaptive_delta = 0.0

    delta = max(float(min_n_delta), float(adaptive_delta))
    candidate_mask = fold_score <= ((1.0 + delta) * best_score)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    fold = int(candidate_indices.min().item()) + fold_min

    return fold


def estimate_fold_from_angles(
    thetas: torch.Tensor,
    max_fold: int,
    min_n_delta: float = 1e-3,
) -> tuple[int, torch.Tensor]:
    """Estimate fold and angular phase from unoriented planar directions.

    The input angles represent unoriented axes or plane normals, so `theta` and
    `theta + pi` are equivalent. The function scores each fold with a circular
    moment and returns the smallest fold close to the best moment.

    Computation:
        1. Sweep fold orders from 1 to `max_fold`.
        2. Score each fold by the magnitude of sum(exp(i * 2 * k * theta)).
        3. Pick the smallest fold close to the best score.
        4. Estimate the phase for that fold from the same circular moment.

    Args:
        thetas: Tensor with shape [N].
        max_fold: Maximum fold order to test.
        min_n_delta: Relative tolerance around the best score.

    Returns:
        fold: Estimated fold order.
        phase: Scalar tensor giving the angular phase of the fold pattern.
    """
    fold_orders = torch.arange(1, max_fold + 1, device=thetas.device, dtype=thetas.dtype)
    phase_angles = 2.0 * thetas[:, None] * fold_orders[None, :]
    phase_real = phase_angles.cos()
    phase_imag = phase_angles.sin()

    scores = (phase_real.sum(dim=0) ** 2 + phase_imag.sum(dim=0) ** 2).sqrt()

    best_fold_idx = int(torch.argmax(scores).item())
    best_score = scores[best_fold_idx]

    candidate_mask = scores >= ((1.0 - min_n_delta) * best_score)
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    fold = int(candidate_indices.min().item()) + 1

    phase_real = torch.cos(2.0 * fold * thetas).sum()
    phase_imag = torch.sin(2.0 * fold * thetas).sum()
    phase = torch.atan2(phase_imag, phase_real) / (2.0 * fold)

    return fold, phase


def detect_rotation_axes(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    num_icp_iter: int = 60,
    num_icp_init: int = 512,
    angle_thresh: float = 8.0,
    dbscan_eps_deg: float = 3.0,
    dbscan_min_samples: int = 6,
    max_fold: int = 15,
) -> List[Dict]:
    """Detect rotational symmetry axes from sampled geometry.

    This finds candidate proper rotations with ICP, groups rotations that share
    the same unoriented axis, estimates a fold for each cluster, and fits one
    axis point for each detected axis.

    Computation:
        1. Initialize ICP from quasi-random proper rotations.
        2. Keep non-trivial rotations whose angle is above `angle_thresh`.
        3. Cluster unoriented rotation axes by angular distance.
        4. Score fold orders with both extrinsic angle error and intrinsic
           basis error.
        5. Fit one dominant axis direction and one point on that axis for each
           cluster.

    Args:
        samples: Tensor with shape [num_samples, 3].
        phi: Intrinsic basis tensor with shape [num_samples, intrinsic_dim].
        Gphi: Mass-weighted intrinsic basis with shape [num_samples, intrinsic_dim].

    Returns:
        A list of rotation-axis candidate dictionaries, sorted by cluster ratio.
    """
    device = samples.device

    # Run ICP from random rotations to discover candidate extrinsic rotations.
    R_init = sample_random_rotations(num_icp_init, device=device)
    t_init = torch.zeros(num_icp_init, 3, device=device, dtype=samples.dtype)
    s_init = torch.ones(num_icp_init, device=device, dtype=samples.dtype)

    X = samples[None].expand(num_icp_init, -1, -1)

    R, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(R_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    # Reject near-identity rotations before estimating axes and folds.
    axis, angle = get_rotation_axis_angle(R)
    mask = angle > np.deg2rad(angle_thresh)

    if mask.sum() < dbscan_min_samples:
        return []

    R = R[mask]
    t = t[mask]
    angle = angle[mask]
    rmse = rmse[mask]
    axis = axis[mask]

    # Group rotations that vote for the same unoriented axis.
    direction_dist = acos_dist(axis)
    labels = DBSCAN(
        eps=np.deg2rad(dbscan_eps_deg),
        min_samples=dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(direction_dist.detach().cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(labels).to(device)

    # Extrinsic fold error measures how close angle * fold is to 2pi multiples.
    fold_orders = torch.arange(2, max_fold + 1, device=device)
    extrinsic_errors = angle[:, None] * fold_orders[None, :]
    extrinsic_errors = torch.remainder(extrinsic_errors + math.pi, 2.0 * math.pi) - math.pi
    extrinsic_errors = extrinsic_errors.abs()

    # Intrinsic fold error tests the transform action in the intrinsic basis.
    intrinsic_errors = intrinsic_fold_error(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        R=R,
        t=t,
        max_fold=max_fold,
    )

    # Fit one axis and one axis point for each DBSCAN cluster.
    rotation_candidates = []
    for cluster_label in labels.unique().tolist():
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label

        fold_extrinsic = estimate_fold_from_errors(
            errors=extrinsic_errors[cluster_mask],
            fold_min=2,
        )
        fold_intrinsic = estimate_fold_from_errors(
            errors=intrinsic_errors[cluster_mask],
            fold_min=2,
        )

        axis_cluster = fit_direction(axis[cluster_mask])
        q_cluster = get_rotation_axis_point(R[cluster_mask], t[cluster_mask], axis_cluster)

        rotation_candidates.append(
            {
                "ratio": cluster_mask.sum().item() / labels.shape[0],
                "dbscan_label": cluster_label,
                "fold_e": fold_extrinsic,
                "fold_i": fold_intrinsic,
                "axis": axis_cluster.tolist(),
                "q": q_cluster.tolist(),
                "rmse": rmse[cluster_mask].mean().item(),
            }
        )

    return sorted(rotation_candidates, key=lambda x: -x["ratio"])


def detect_reflection_planes(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    num_icp_iter: int = 60,
    num_icp_init: int = 1024,
    angle_thresh: float = 3.0,
    rel_dist_thresh: float = 0.05,
    dbscan_eps_deg: float = 3.0,
    dbscan_min_samples: int = 6,
) -> List[Dict]:
    """Detect global reflection planes from sampled geometry.

    This finds candidate reflection transforms with ICP, filters transforms
    that are close to pure reflections, clusters reflection normals, and fits
    one plane for each cluster.

    Computation:
        1. Initialize ICP from improper orthogonal transforms.
        2. Extract reflection normals and reject non-reflection-like results.
        3. Cluster unoriented normals by angular distance.
        4. Validate each cluster with intrinsic fold error.
        5. Fit one plane normal and offset per cluster.

    Args:
        samples: Tensor with shape [num_samples, 3].
        phi: Intrinsic basis tensor with shape [num_samples, intrinsic_dim].
        Gphi: Mass-weighted intrinsic basis with shape [num_samples, intrinsic_dim].

    Returns:
        A list of reflection-plane candidate dictionaries, sorted by RMSE.
    """
    device = samples.device

    # Run ICP from mirrored rotations to discover candidate reflections.
    O_init = -sample_random_rotations(num_icp_init, device=device)  # mirrored
    t_init = torch.zeros(num_icp_init, 3, device=device, dtype=samples.dtype)
    s_init = torch.ones(num_icp_init, device=device, dtype=samples.dtype)

    X = samples[None].expand(num_icp_init, -1, -1)

    O, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(O_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    n, _ = get_reflection_plane(O, t)
    trace_O = O[:, 0, 0] + O[:, 1, 1] + O[:, 2, 2]
    angle = (0.5 * (trace_O + 1)).clamp(-1, 1).arccos()
    t_para = t - (t * n).sum(dim=-1)[:, None] * n
    ext = (samples.amax(dim=0) - samples.amin(dim=0)).max()

    # Keep transforms that look like pure reflections with small in-plane drift.
    mask = (angle < np.deg2rad(angle_thresh)) * (t_para.norm(dim=1) < ext * rel_dist_thresh)

    if mask.sum() < dbscan_min_samples:
        return []

    O = O[mask]
    t = t[mask]
    n = n[mask]
    rmse = rmse[mask]

    # Group reflection normals up to sign.
    direction_dist = acos_dist(n)
    labels = DBSCAN(
        eps=np.deg2rad(dbscan_eps_deg),
        min_samples=dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(direction_dist.detach().cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(labels).to(device)

    # Intrinsic error is used as a reflection-cluster validation score.
    intrinsic_errors = intrinsic_fold_error(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        R=O,
        t=t,
        max_fold=15,
    )

    # Fit one plane for each normal cluster.
    reflection_candidates = []
    for cluster_label in labels.unique().tolist():
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label

        fold_intrinsic_validation = estimate_fold_from_errors(
            errors=intrinsic_errors[cluster_mask],
            fold_min=2,
        )

        offsets_cluster = 0.5 * (t[cluster_mask] * n[cluster_mask]).sum(dim=-1)
        normal_cluster, offset_cluster = fit_reflection_plane(n[cluster_mask], offsets_cluster)

        reflection_candidates.append(
            {
                "ratio": cluster_mask.sum().item() / labels.shape[0],
                "dbscan_label": cluster_label,
                "fold_i_val": fold_intrinsic_validation,
                "n": normal_cluster.tolist(),
                "c": offset_cluster.tolist(),
                "rmse": rmse[cluster_mask].mean().item(),
            }
        )

    return sorted(reflection_candidates, key=lambda x: x["rmse"])


def detect_reflection_planes_containing_axis(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_iter: int = 120,
    num_icp_init: int = 1024,
    angle_thresh: float = 3.0,
    rel_dist_thresh: float = 0.05,
    dbscan_eps_deg: float = 3.0,
    dbscan_min_samples: int = 6,
    max_fold: int = 15,
) -> List[Dict]:
    """Detect reflection planes that contain a known axis.

    This detects mirror planes whose normals are perpendicular to the given
    axis and whose plane contains the line `q + lambda * axis`.

    Computation:
        1. Initialize ICP from uniformly sampled reflection planes containing
           the known axis.
        2. Keep pure reflection candidates whose normals are perpendicular to
           the known axis and whose plane contains `q`.
        3. Estimate the angular fold pattern of those plane normals around the
           axis.
        4. Cluster reflection normals and validate clusters intrinsically.
        5. Fit each plane and snap its normal/offset to the estimated fold
           pattern.

    Args:
        samples: Tensor with shape [num_samples, 3].
        phi: Intrinsic basis tensor with shape [num_samples, intrinsic_dim].
        Gphi: Mass-weighted intrinsic basis with shape [num_samples, intrinsic_dim].
        axis: Tensor with shape [3]. Known axis direction.
        q: Tensor with shape [3]. A point on the known axis.

    Returns:
        A list of reflection-plane candidate dictionaries, sorted by RMSE.
    """

    device = samples.device
    axis = axis / axis.norm()

    O_init, t_init = sample_axis_parallel_reflections(
        axis=axis,
        q=q,
        num_icp_init=num_icp_init,
    )
    s_init = torch.ones(num_icp_init, device=device, dtype=samples.dtype)

    X = samples[None].expand(num_icp_init, -1, -1)

    O, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(O_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    n, reflect_offset = get_reflection_plane(O, t)

    # Filter 0: candidate should be close to a pure reflection.
    trace_O = O[:, 0, 0] + O[:, 1, 1] + O[:, 2, 2]
    angle = (0.5 * (trace_O + 1)).clamp(-1, 1).arccos()
    t_para = t - (t * n).sum(dim=-1)[:, None] * n
    ext = (samples.amax(dim=0) - samples.amin(dim=0)).max()

    mask_0 = (angle < np.deg2rad(angle_thresh)) * (t_para.norm(dim=1) < ext * rel_dist_thresh)

    # Filter 1: a plane containing the axis has normal perpendicular to axis.
    parallel_metric = (n * axis[None]).sum(dim=1).abs().asin()
    mask_1 = parallel_metric < np.deg2rad(angle_thresh)

    # Filter 2: the known point q on the axis should lie on the plane.
    contain_metric = ((n * q[None]).sum(dim=1) - reflect_offset).abs()
    mask_2 = contain_metric < ext * rel_dist_thresh

    mask = mask_0 * mask_1 * mask_2

    if mask.sum() < dbscan_min_samples:
        return []

    O = O[mask]
    t = t[mask]
    n = n[mask]
    rmse = rmse[mask]

    e1, e2 = gram_schmidt_frame(axis)

    # Estimate the fold pattern of mirror normals in axis^perp.
    x = (n * e1[None]).sum(dim=1)
    y = (n * e2[None]).sum(dim=1)
    thetas = torch.cat([torch.atan2(y, x), torch.atan2(-y, -x)], dim=0)

    fold, phase = estimate_fold_from_angles(thetas, max_fold=max_fold)
    delta = torch.pi / fold

    # Cluster mirror-plane normals up to sign.
    direction_dist = acos_dist(n)

    labels = DBSCAN(
        eps=np.deg2rad(dbscan_eps_deg),
        min_samples=dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(direction_dist.detach().cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(labels).to(device)

    # Validate each reflection cluster with the intrinsic basis score.
    intrinsic_errors = intrinsic_fold_error(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        R=O,
        t=t,
        max_fold=15,
    )

    # Fit each plane and snap it to the estimated angular fold pattern.
    reflection_candidates = []
    for cluster_label in labels.unique().tolist():
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label

        fold_intrinsic_validation = estimate_fold_from_errors(
            errors=intrinsic_errors[cluster_mask],
            fold_min=2,
        )

        offsets_cluster = 0.5 * (t[cluster_mask] * n[cluster_mask]).sum(dim=-1)
        normal_cluster, offset_cluster = fit_reflection_plane(n[cluster_mask], offsets_cluster)

        angle_cluster = torch.atan2(normal_cluster.dot(e2), normal_cluster.dot(e1))
        index_cluster = ((angle_cluster - phase) / delta).round()
        angle_cor = phase + delta * index_cluster
        normal_cor = angle_cor.cos() * e1 + angle_cor.sin() * e2
        const_cor = normal_cor.dot(q)

        reflection_candidates.append(
            {
                "ratio": cluster_mask.sum().item() / labels.shape[0],
                "dbscan_label": cluster_label,
                "fold_i_val": fold_intrinsic_validation,
                "n": normal_cluster.tolist(),
                "c": offset_cluster.tolist(),
                "n_cor": normal_cor.tolist(),
                "c_cor": const_cor.tolist(),
                "rmse": rmse[cluster_mask].mean().item(),
                "fold_pred": fold,
            }
        )

    return sorted(reflection_candidates, key=lambda x: x["rmse"])


def detect_reflection_planes_perpendicular_to_axis(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_iter: int = 60,
    num_icp_init: int = 1024,
    angle_thresh: float = 3.0,
    rel_dist_thresh: float = 0.05,
    dbscan_eps_rel: float = 0.01,
    dbscan_min_samples: int = 6,
) -> List[Dict]:
    """Detect reflection planes perpendicular to a known axis.

    This detects mirror planes whose normal is the given axis direction. These
    planes differ primarily by their offset along the axis.

    Computation:
        1. Initialize ICP from planes perpendicular to `axis` and sampled along
           the object extent.
        2. Keep pure reflection candidates with small in-plane translation and
           normals aligned with `axis`.
        3. Cluster candidates by plane offset.
        4. Validate clusters with intrinsic fold error.
        5. Fit the plane and correct its normal to exactly match `axis`.

    Args:
        samples: Tensor with shape [num_samples, 3].
        phi: Intrinsic basis tensor with shape [num_samples, intrinsic_dim].
        Gphi: Mass-weighted intrinsic basis with shape [num_samples, intrinsic_dim].
        axis: Tensor with shape [3]. Known axis direction and plane normal.
        q: Tensor with shape [3]. A point on the known axis.

    Returns:
        A list of reflection-plane candidate dictionaries, sorted by RMSE.
    """

    device = samples.device
    axis = axis / axis.norm()

    O_init, t_init = sample_axis_perp_reflections(
        verts=samples,
        axis=axis,
        q=q,
        num_icp_init=num_icp_init,
    )
    s_init = torch.ones(num_icp_init, device=device, dtype=samples.dtype)

    X = samples[None].expand(num_icp_init, -1, -1)

    O, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(O_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    n, _ = get_reflection_plane(O, t)
    dot = (n * axis[None]).sum(dim=1)
    n = n * dot.sign()[..., None]

    # Filter 0: candidate should be close to a pure reflection.
    trace_O = O[:, 0, 0] + O[:, 1, 1] + O[:, 2, 2]
    angle = (0.5 * (trace_O + 1)).clamp(-1, 1).arccos()
    mask_0 = angle < np.deg2rad(angle_thresh)

    # Filter 1: in-plane translation should be small.
    t_para = t - (t * n).sum(dim=-1)[:, None] * n
    ext = (samples.amax(dim=0) - samples.amin(dim=0)).max()
    mask_1 = t_para.norm(dim=1) < ext * rel_dist_thresh

    # Filter 2: reflection normal should align with the known axis.
    perp_metric = dot.abs().acos()
    mask_2 = perp_metric < np.deg2rad(angle_thresh)

    mask = mask_0 * mask_1 * mask_2

    if mask.sum() < dbscan_min_samples:
        return []

    O = O[mask]
    t = t[mask]
    n = n[mask]
    rmse = rmse[mask]

    plane_offsets = (t * n).sum(dim=1) / 2
    offset_dist = (plane_offsets[..., None] - plane_offsets[None, ...]).abs() / ext
    labels = DBSCAN(
        eps=dbscan_eps_rel,
        min_samples=dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(offset_dist.detach().cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(labels).to(device)

    # Validate each offset cluster with the intrinsic basis score.
    intrinsic_errors = intrinsic_fold_error(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        R=O,
        t=t,
        max_fold=15,
    )

    # Fit each offset cluster and correct the plane normal to the known axis.
    reflection_candidates = []
    for cluster_label in labels.unique().tolist():
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label

        fold_intrinsic_validation = estimate_fold_from_errors(
            errors=intrinsic_errors[cluster_mask],
            fold_min=2,
        )

        t_cluster = t[cluster_mask]
        offsets_cluster = 0.5 * (t_cluster * n[cluster_mask]).sum(dim=-1)
        normal_cluster, offset_cluster = fit_reflection_plane(n[cluster_mask], offsets_cluster)

        normal_cor = axis
        const_cor = 0.5 * (t_cluster * axis[None]).sum(dim=-1).mean()

        reflection_candidates.append(
            {
                "ratio": cluster_mask.sum().item() / labels.shape[0],
                "dbscan_label": cluster_label,
                "fold_i_val": fold_intrinsic_validation,
                "n": normal_cluster.tolist(),
                "c": offset_cluster.tolist(),
                "n_cor": normal_cor.tolist(),
                "c_cor": const_cor.tolist(),
                "rmse": rmse[cluster_mask].mean().item(),
            }
        )

    return sorted(reflection_candidates, key=lambda x: x["rmse"])


def detect_c2_axes_perpendicular_to_axis(
    samples: torch.Tensor,
    phi: torch.Tensor,
    Gphi: torch.Tensor,
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_iter: int = 60,
    num_icp_init: int = 512,
    angle_thresh: float = 3.0,
    rel_dist_thresh: float = 0.05,
    dbscan_eps_deg: float = 3.0,
    dbscan_min_samples: int = 6,
    max_fold: int = 15,
) -> List[Dict]:
    """Detect C2 axes perpendicular to a known axis.

    This detects 180-degree rotation axes that are perpendicular to a known
    main axis and intersect that main axis. The detected C2 axes can indicate
    dihedral-type structure around the main axis.

    Computation:
        1. Initialize ICP from C2 rotations whose axes are perpendicular to the
           known axis.
        2. Keep candidates that are near half-turns, have no screw component,
           are perpendicular to the known axis, and intersect the known axis.
        3. Estimate the angular fold pattern of C2 axes around the known axis.
        4. Estimate a common center on the known axis from line-line geometry.
        5. Cluster C2 axes, validate clusters intrinsically, and snap each axis
           to the estimated fold pattern.

    Args:
        samples: Tensor with shape [num_samples, 3].
        phi: Intrinsic basis tensor with shape [num_samples, intrinsic_dim].
        Gphi: Mass-weighted intrinsic basis with shape [num_samples, intrinsic_dim].
        axis: Tensor with shape [3]. Known main axis direction.
        q: Tensor with shape [3]. A point on the known main axis.

    Returns:
        A list of C2-axis candidate dictionaries, sorted by RMSE.
    """

    device = samples.device

    R_init, t_init = sample_axis_perp_c2_rotations(
        verts=samples,
        axis=axis,
        q=q,
        num_icp_init=num_icp_init,
    )
    s_init = torch.ones(num_icp_init, device=device, dtype=samples.dtype)

    X = samples[None].expand(num_icp_init, -1, -1)

    R, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(R_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    # Filter 0: candidate should be close to a 180-degree rotation.
    trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = (0.5 * (1 - trace_R)).clamp(-1, 1).arccos()
    mask_0 = angle < np.deg2rad(angle_thresh)

    # Filter 1: reject screw-like translation along the C2 axis.
    n = get_c2_axis(R)
    ext = (samples.amax(dim=0) - samples.amin(dim=0)).max()
    mask_1 = (t * n).sum(dim=-1).abs() < ext * rel_dist_thresh

    # Filter 2: C2 axis should be perpendicular to the known axis.
    mask_2 = (n * axis[None]).sum(dim=1).abs().asin() < np.deg2rad(angle_thresh)

    # Filter 3: C2 axis should intersect the known axis.
    # distance between two 3D lines:
    #   L1: p + s * n
    #   L2: q + u * axis
    #
    # if not parallel:
    #   dist = |(q - p) · (n x axis)| / ||n x axis||
    #
    # if nearly parallel:
    #   dist = ||(q - p) x n|| / ||n||
    I = torch.eye(3, device=device, dtype=R.dtype)[None].expand(R.shape[0], -1, -1)
    A = I - R
    c2_axis_points = torch.linalg.lstsq(A, t[..., None]).solution[..., 0]

    cross_na = torch.cross(n, axis[None].expand_as(n), dim=-1)
    cross_na_norm = cross_na.norm(dim=-1)
    line_delta = q[None] - c2_axis_points

    line_dist = torch.abs((line_delta * cross_na).sum(dim=-1)) / cross_na_norm.clamp_min(1e-8)
    mask_3 = line_dist < ext * rel_dist_thresh

    mask = mask_0 * mask_1 * mask_2 * mask_3

    if mask.sum() < dbscan_min_samples:
        return []

    R = R[mask]
    t = t[mask]
    n = n[mask]
    rmse = rmse[mask]

    c2_axis_points = c2_axis_points[mask]

    e1, e2 = gram_schmidt_frame(axis)

    # Estimate the fold pattern of C2 axis directions in axis^perp.
    x = (n * e1[None]).sum(dim=1)
    y = (n * e2[None]).sum(dim=1)
    thetas = torch.cat([torch.atan2(y, x), torch.atan2(-y, -x)], dim=0)

    fold, phase = estimate_fold_from_angles(thetas, max_fold=max_fold)
    delta = torch.pi / fold

    # Estimate the common center on the known axis from closest line points.
    point_delta = c2_axis_points - q[None]
    axis_dot = (n * axis[None]).sum(dim=-1)
    c2_axis_point_proj = (n * point_delta).sum(dim=-1)
    main_axis_point_proj = (axis[None] * point_delta).sum(dim=-1)

    main_axis_coord = (main_axis_point_proj - axis_dot * c2_axis_point_proj) / (
        1.0 - axis_dot * axis_dot
    ).clamp_min(1e-8)

    # Store each candidate axis's closest point on the known axis.
    axis_hits = q[None] + main_axis_coord[:, None] * axis[None]  # [M, 3]

    center_coord = main_axis_coord.mean()
    center = q + center_coord * axis  # [3]

    # Cluster C2 axes up to sign.
    direction_dist = acos_dist(n)

    labels = DBSCAN(
        eps=np.deg2rad(dbscan_eps_deg),
        min_samples=dbscan_min_samples,
        metric="precomputed",
    ).fit_predict(direction_dist.detach().cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(labels).to(device)

    # Validate each C2-axis cluster with the intrinsic basis score.
    intrinsic_errors = intrinsic_fold_error(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        R=R,
        t=t,
        max_fold=15,
    )

    # Fit each C2-axis cluster and snap it to the estimated fold pattern.
    rotation_candidates = []
    for cluster_label in labels.unique().tolist():
        if cluster_label == -1:
            continue
        cluster_mask = labels == cluster_label

        fold_intrinsic_validation = estimate_fold_from_errors(
            errors=intrinsic_errors[cluster_mask],
            fold_min=2,
        )

        axis_c2_cluster = fit_direction(n[cluster_mask])
        q_c2_cluster = get_rotation_axis_point(R[cluster_mask], t[cluster_mask], axis_c2_cluster)

        # Snap the fitted C2 axis to the closest phase slot of the fold pattern.
        angle_cluster = torch.atan2(axis_c2_cluster.dot(e2), axis_c2_cluster.dot(e1))
        index_cluster = ((angle_cluster - phase) / delta).round()
        angle_cor = phase + delta * index_cluster
        axis_c2_cor = angle_cor.cos() * e1 + angle_cor.sin() * e2
        axis_c2_cor = axis_c2_cor / axis_c2_cor.norm()

        q_c2_cor = center

        rotation_candidates.append(
            {
                "ratio": cluster_mask.sum().item() / labels.shape[0],
                "dbscan_label": cluster_label,
                "fold_i_val": fold_intrinsic_validation,
                "fold_c2": fold,
                "axis": axis_c2_cluster.tolist(),
                "q": q_c2_cluster.tolist(),
                "axis_cor": axis_c2_cor.tolist(),
                "q_cor": q_c2_cor.tolist(),
                "rmse": rmse[cluster_mask].mean().item(),
            }
        )

    return sorted(rotation_candidates, key=lambda x: x["rmse"])


def icp_refine_rotation(
    axis_init: torch.Tensor,
    q_init: torch.Tensor,
    fold: int,
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int = 65536,
    num_icp_iter: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine one rotational symmetry candidate with single-transform ICP.

    This takes an initial rotation axis and fold, constructs the corresponding
    one-step rotation, and refines it against dense sampled surface points.

    Computation:
        1. Sample surface points from the mesh for the refinement stage.
        2. Build the rotation transform for angle `2pi / fold`.
        3. Run ICP from this single initial transform.
        4. Extract the refined axis and a point on the refined axis.

    Args:
        axis_init: Tensor with shape [3]. Initial rotation axis direction.
        q_init: Tensor with shape [3]. Initial point on the rotation axis.
        fold: Rotation fold order.
        verts: Tensor with shape [num_verts, 3].
        faces: Long tensor with shape [num_faces, 3].

    Returns:
        axis: Tensor with shape [3].
        q: Tensor with shape [3].
        rmse: Tensor with shape [1].
    """
    device, dtype = verts.device, verts.dtype

    # Use a denser sample set for final local refinement.
    mesh = Meshes(verts[None], faces[None])
    samples = sample_mesh_farthest_points(
        mesh,
        num_points=num_samples,
    )

    # Initialize ICP with the rotation implied by the candidate fold.
    R_init, t_init = axis_point_rotation_transform(
        axis=axis_init.to(device=device, dtype=dtype),
        q=q_init.to(device=device, dtype=dtype),
        angle=torch.tensor(torch.pi * 2 / fold, device=device, dtype=dtype),
    )
    R_init, t_init = R_init[None, :], t_init[None, :]
    s_init = torch.ones(1, device=device, dtype=samples.dtype)

    X = samples[None].expand(1, -1, -1)  # source

    # Refine the transform against the same sampled surface.
    R, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(R_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    # Convert the refined transform back to axis-point representation.
    axis, _ = get_rotation_axis_angle(R)
    axis = axis[0]
    q = get_rotation_axis_point(R, t, axis)

    return axis, q, rmse


def icp_refine_reflection(
    normal_init: torch.Tensor,
    q_init: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int = 65536,
    num_icp_iter: int = 192,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine one reflection-plane candidate with single-transform ICP.

    This takes an initial plane normal and point, constructs the corresponding
    reflection transform, and refines it against dense sampled surface points.

    Computation:
        1. Sample surface points from the mesh for the refinement stage.
        2. Build the reflection transform from the initial plane.
        3. Run ICP from this single initial transform.
        4. Extract the refined plane normal and offset.

    Args:
        normal_init: Tensor with shape [3]. Initial plane normal.
        q_init: Tensor with shape [3]. Initial point on the plane.
        verts: Tensor with shape [num_verts, 3].
        faces: Long tensor with shape [num_faces, 3].

    Returns:
        n: Tensor with shape [3]. Refined plane normal.
        c: Scalar tensor. Refined plane offset in `x @ n = c`.
        rmse: Tensor with shape [1].
    """
    device, dtype = verts.device, verts.dtype

    # Use a denser sample set for final local refinement.
    mesh = Meshes(verts[None], faces[None])
    samples = sample_mesh_farthest_points(
        mesh,
        num_points=num_samples,
    )

    # Initialize ICP with the reflection implied by the candidate plane.
    O_init, t_init = plane_reflection_transform(
        normal=normal_init.to(device=device, dtype=dtype),
        q=q_init.to(device=device, dtype=dtype),
    )
    O_init, t_init = O_init[None, :], t_init[None, :]
    s_init = torch.ones(1, device=device, dtype=samples.dtype)

    X = samples[None].expand(1, -1, -1)  # source

    # Refine the transform against the same sampled surface.
    O, t, _, _, rmse = iterative_closest_point(
        X=X,
        Y=X,
        init_transform=(O_init, t_init, s_init),
        max_iterations=num_icp_iter,
        relative_rmse_thr=1e-6,
        estimate_scale=False,
        allow_improper=False,
    )
    rmse = cast(torch.Tensor, rmse)

    # Convert the refined transform back to plane normal-offset form.
    n, c = get_reflection_plane(O, t)
    n = n[0]
    c = c[0]

    return n, c, rmse
