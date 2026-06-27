"""Batched ICP for similarity transforms.

Transforms use the row-vector convention:

    X_new = s * (X @ R.T) + t

`R` is the orthogonal factor of the similarity transform. It is usually a
proper rotation, but can be improper when `allow_improper=True`.
"""

from typing import Callable, Optional, Tuple

import torch
from pytorch3d.ops import knn_points


def apply_similarity_transform(
    X: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """Apply batched similarity transforms to point clouds.

    Args:
        X: Tensor with shape [B, N, 3].
        R: Tensor with shape [B, 3, 3]. Orthogonal transform factors.
        t: Tensor with shape [B, 3].
        s: Tensor with shape [B].

    Returns:
        X_new: Tensor with shape [B, N, 3].
    """
    return s[:, None, None] * (X @ R.transpose(1, 2)) + t[:, None, :]


def corresponding_points_alignment(
    X: torch.Tensor,
    Y: torch.Tensor,
    estimate_scale=False,
    allow_improper=False,
    eps=1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate batched similarity transforms from paired points.

    Args:
        X: Tensor with shape [B, N, 3]. Source points.
        Y: Tensor with shape [B, N, 3]. Target points matched row-wise to `X`.
        estimate_scale: If true, estimate scalar scale `s`; otherwise use 1.
        allow_improper: If true, allow det(R) = -1 in the Kabsch update.
        eps: Minimum variance used by scale estimation.

    Returns:
        R: Tensor with shape [B, 3, 3].
        t: Tensor with shape [B, 3].
        s: Tensor with shape [B].
    """
    Xmu = X.mean(dim=1, keepdim=True)
    Ymu = Y.mean(dim=1, keepdim=True)

    Xc = X - Xmu
    Yc = Y - Ymu

    XYcov = Xc.transpose(1, 2) @ Yc / X.shape[1]

    U, S, Vh = torch.linalg.svd(XYcov)

    E = torch.eye(X.shape[-1], device=X.device, dtype=X.dtype)[None].repeat(X.shape[0], 1, 1)
    # By default, force the alignment update to be proper; reflections are opt-in.
    if not allow_improper:
        E[:, -1, -1] = torch.det(U @ Vh)

    A = U @ E @ Vh

    if estimate_scale:
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(dim=1)
        Xcov = (Xc.square().sum(dim=(1, 2)) / X.shape[1]).clamp_min(eps)
        s = trace_ES / Xcov
    else:
        s = X.new_ones(X.shape[0])

    A = U @ E @ Vh
    t = Ymu[:, 0, :] - s[:, None] * (Xmu @ A)[:, 0, :]
    R = A.transpose(1, 2)

    return R, t, s


def compose_similarity_transform(
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    dR: torch.Tensor,
    dt: torch.Tensor,
    ds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose a current similarity transform with an incremental update.

    Args:
        R: Tensor with shape [B, 3, 3]. Current orthogonal factors.
        t: Tensor with shape [B, 3]. Current translations.
        s: Tensor with shape [B]. Current scales.
        dR: Tensor with shape [B, 3, 3]. Incremental orthogonal factors.
        dt: Tensor with shape [B, 3]. Incremental translations.
        ds: Tensor with shape [B]. Incremental scales.

    Returns:
        R_new: Tensor with shape [B, 3, 3].
        t_new: Tensor with shape [B, 3].
        s_new: Tensor with shape [B].
    """
    R_new = dR @ R
    s_new = ds * s
    t_new = ds[:, None] * (t[:, None, :] @ dR.transpose(1, 2))[:, 0, :] + dt

    return R_new, t_new, s_new


def icp_step(
    X0: torch.Tensor,
    Y: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    s: torch.Tensor,
    estimate_scale=False,
    allow_improper=False,
    constraint_fn: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one ICP correspondence and alignment step.

    Args:
        X0: Tensor with shape [B, N, 3]. Original source points.
        Y: Tensor with shape [B, M, 3]. Target points.
        R: Tensor with shape [B, 3, 3]. Current orthogonal factors.
        t: Tensor with shape [B, 3]. Current translations.
        s: Tensor with shape [B]. Current scales.
        estimate_scale: If true, estimate scalar scale updates.
        allow_improper: If true, allow det(R) = -1 updates.
        constraint_fn: Optional projection applied to `(R_new, t_new, s_new)`.

    Returns:
        R_new: Tensor with shape [B, 3, 3].
        t_new: Tensor with shape [B, 3].
        s_new: Tensor with shape [B].
        Xt_new: Tensor with shape [B, N, 3].
        rmse: Tensor with shape [B].
    """

    Xt = apply_similarity_transform(X0, R, t, s)
    Ynn = knn_points(Xt, Y, K=1, return_nn=True).knn[:, :, 0, :]

    dR, dt, ds = corresponding_points_alignment(
        Xt,
        Ynn,
        estimate_scale=estimate_scale,
        allow_improper=allow_improper,
    )
    R_new, t_new, s_new = compose_similarity_transform(R, t, s, dR, dt, ds)

    if constraint_fn is not None:
        R_new, t_new, s_new = constraint_fn(
            R_new,
            t_new,
            s_new,
        )

    Xt_new = apply_similarity_transform(X0, R_new, t_new, s_new)
    rmse = ((Xt_new - Ynn).square().sum(dim=-1).mean(dim=1)).sqrt()

    return R_new, t_new, s_new, Xt_new, rmse


def iterative_closest_point(
    X: torch.Tensor,
    Y: torch.Tensor,
    init_transform: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    estimate_scale: bool = False,
    allow_improper: bool = False,
    constraint_fn: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ]
    ] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Run batched ICP until convergence or a fixed iteration limit.

    Args:
        X: Tensor with shape [B, N, 3]. Source points.
        Y: Tensor with shape [B, M, 3]. Target points.
        init_transform: Tuple `(R, t, s)` with shapes [B, 3, 3], [B, 3], [B].
        max_iterations: Maximum number of ICP steps.
        relative_rmse_thr: Stop when all batch items improve by at most this ratio.
        estimate_scale: If true, estimate scalar scale updates.
        allow_improper: If true, allow det(R) = -1 updates.
        constraint_fn: Optional projection applied after each alignment step.

    Returns:
        R: Tensor with shape [B, 3, 3].
        t: Tensor with shape [B, 3].
        s: Tensor with shape [B].
        Xt: Tensor with shape [B, N, 3], or None when no iteration runs.
        rmse: Tensor with shape [B], or None when no iteration runs.
    """

    R, t, s = init_transform

    X0, Xt = X, None
    prev_rmse, rmse = None, None

    for i in range(max_iterations):
        R, t, s, Xt, rmse = icp_step(
            X0,
            Y,
            R,
            t,
            s,
            estimate_scale=estimate_scale,
            allow_improper=allow_improper,
            constraint_fn=constraint_fn,
        )

        if prev_rmse is not None:
            relative_rmse = (prev_rmse - rmse) / prev_rmse.clamp_min(1e-12)
            if (relative_rmse <= relative_rmse_thr).all():
                break

        prev_rmse = rmse

    return R, t, s, Xt, rmse
