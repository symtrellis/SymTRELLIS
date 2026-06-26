from typing import Tuple

import torch
from pytorch3d.transforms import axis_angle_to_matrix


def axis_point_rotation_transform(
    axis: torch.Tensor,
    q: torch.Tensor,
    angle,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a rotation around an axis passing through a point.

    The returned transform uses the row-vector convention:

        x_new = x @ R.T + t

    Args:
        axis: Tensor with shape [3]. Direction of the rotation axis.
        q: Tensor with shape [3]. A point on the rotation axis.
        angle: Scalar rotation angle in radians. Positive direction follows
            the right-hand rule around `axis`.

    Returns:
        R: Tensor with shape [3, 3].
        t: Tensor with shape [3].

    `q` is fixed by the transform: `q == q @ R.T + t`, up to floating-point
    error.
    """
    a = axis / axis.norm().clamp_min(1e-12)
    angle = torch.as_tensor(angle, dtype=a.dtype, device=a.device)

    rotvec = a * angle  # [3]
    R = axis_angle_to_matrix(rotvec[None])[0]  # [3, 3]
    t = q - q @ R.T  # [3]

    return R, t


def axis_point_half_turn_transform(
    axis: torch.Tensor,
    q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a 180-degree rotation around an axis through a point.

    This is the C2, or half-turn, special case of
    `axis_point_rotation_transform`. It uses the closed-form matrix
    `R = 2 aa^T - I` for the normalized axis direction `a`.

    The returned transform uses the row-vector convention:

        x_new = x @ R.T + t

    Args:
        axis: Tensor with shape [3]. Direction of the half-turn axis.
        q: Tensor with shape [3]. A point on the axis.

    Returns:
        R: Tensor with shape [3, 3].
        t: Tensor with shape [3].

    `q` is fixed by the transform: `q == q @ R.T + t`, up to floating-point
    error.
    """
    a = axis / axis.norm().clamp_min(1e-12)
    I = torch.eye(3, device=a.device, dtype=a.dtype)
    R = 2 * a[:, None] * a[None, :] - I
    t = q - q @ R.T

    return R, t


def plane_reflection_transform(
    normal: torch.Tensor,
    q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a mirror reflection across a plane.

    The plane is defined by a normal direction and one point on the plane. The
    returned transform uses the row-vector convention:

        x_new = x @ O.T + t

    Args:
        normal: Tensor with shape [3]. Plane normal direction.
        q: Tensor with shape [3]. A point on the reflection plane.

    Returns:
        O: Tensor with shape [3, 3].
        t: Tensor with shape [3].

    Every point on the plane is fixed by the transform. The determinant of `O`
    is -1.
    """
    n = normal / normal.norm().clamp_min(1e-12)
    I = torch.eye(3, device=n.device, dtype=n.dtype)
    O = I - 2 * n[:, None] * n[None, :]
    t = q - q @ O.T

    return O, t
