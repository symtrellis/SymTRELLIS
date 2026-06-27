"""Small geometry fitting utilities for symmetry detection.

These functions analyze already-estimated rigid or orthogonal transforms. They
do not construct symmetry transforms; construction lives in
`symtrellis.symmetry.transforms`.
"""

import torch
from pytorch3d.transforms import so3_log_map


def fit_direction(directions: torch.Tensor) -> torch.Tensor:
    """Fit one unoriented 3D direction from multiple direction observations.

    Args:
        directions: Tensor with shape [N, 3]. Each row is a unit or non-zero
            direction. The sign is treated as ambiguous, so `v` and `-v`
            represent the same direction.

    Returns:
        direction: Tensor with shape [3]. Dominant unoriented direction.
    """
    # Average vv^T so opposite signs vote for the same axis.
    direction_mat = (directions[:, :, None] @ directions[:, None, :]).mean(dim=0)
    _, eigvecs = torch.linalg.eigh(direction_mat)
    direction = eigvecs[:, -1]
    direction = direction / direction.norm()

    return direction


def acos_dist(directions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise unoriented angular distances between 3D directions.

    Args:
        directions: Tensor with shape [N, 3].

    Returns:
        dist: Tensor with shape [N, N]. Entry `(i, j)` is the angle between
            directions `i` and `j`, with opposite signs treated as equivalent.
    """
    return torch.acos((directions @ directions.t()).clamp(-1.0, 1.0).abs())


def get_rotation_axis_angle(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract rotation axes and angles from rotation matrices.

    Args:
        R: Tensor with shape [N, 3, 3]. Proper rotation matrices.

    Returns:
        axis: Tensor with shape [N, 3].
        angle: Tensor with shape [N].
    """
    log_rot = so3_log_map(R)
    angle = log_rot.norm(dim=-1)
    axis = log_rot / angle.clamp_min(1e-8)[..., None]

    return axis, angle


def get_c2_axis(R: torch.Tensor) -> torch.Tensor:
    """Extract axes from 180-degree rotation matrices.

    Args:
        R: Tensor with shape [N, 3, 3]. C2 rotation matrices.

    Returns:
        axis: Tensor with shape [N, 3]. Each row is one half-turn axis.
    """
    # For a half-turn, the axis is the eigenvector with eigenvalue +1.
    axis = torch.linalg.eigh(R + R.transpose(-1, -2))[-1][..., -1]
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    return axis


def get_rotation_axis_point(
    R: torch.Tensor,
    t: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor:
    """Fit one point on a shared rotation axis from several transforms.

    The input transforms use the row-vector convention `x_new = x @ R.T + t`.

    Args:
        R: Tensor with shape [N, 3, 3]. Rotation matrices.
        t: Tensor with shape [N, 3]. Translation vectors paired with `R`.
        axis: Tensor with shape [3]. Shared rotation axis direction.

    Returns:
        q: Tensor with shape [3]. A point on the fitted rotation axis, with
            the component along `axis` removed to fix the gauge.
    """
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand(R.shape[0], -1, -1)
    A = I - R
    d = t @ axis
    rhs = t - d[:, None] * axis[None, :]

    # Solve all fixed-axis equations as one least-squares system.
    q = torch.linalg.lstsq(A.reshape(-1, 3), rhs.reshape(-1)).solution
    q = q - axis * (axis @ q)

    return q


def get_reflection_plane(
    O: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract reflection planes from reflection transforms.

    The input transforms use the row-vector convention `x_new = x @ O.T + t`.

    Args:
        O: Tensor with shape [N, 3, 3]. Orthogonal reflection matrices.
        t: Tensor with shape [N, 3]. Translation vectors paired with `O`.

    Returns:
        normal: Tensor with shape [N, 3]. Plane normal directions.
        offset: Tensor with shape [N]. Plane offsets in the form
            `x @ normal = offset`.
    """
    I = torch.eye(3, device=O.device, dtype=O.dtype)[None]
    # The reflection normal spans the null space of O + I.
    normal = torch.linalg.svd(O + I)[-1][:, -1, :]
    normal = normal / normal.norm(dim=1, keepdim=True)
    offset = 0.5 * (t * normal).sum(dim=-1)

    return normal, offset


def fit_reflection_plane(
    normals: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one reflection plane from multiple plane observations.

    Args:
        normals: Tensor with shape [N, 3]. Observed plane normals.
        offsets: Tensor with shape [N]. Observed plane offsets, where each
            plane is represented as `x @ normals[i] = offsets[i]`.

    Returns:
        normal: Tensor with shape [3]. Fitted plane normal.
        offset: Scalar tensor. Fitted plane offset.
    """
    normal = fit_direction(normals)
    points = offsets[:, None] * normals
    offset = (points.mean(dim=0) * normal).sum()

    return normal, offset
