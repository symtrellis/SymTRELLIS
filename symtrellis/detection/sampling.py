"""Sampling utilities used by symmetry detection.

This module samples surface points and initial rigid or reflection transforms
for ICP-based symmetry detection. All returned transforms use the row-vector
convention `x_new = x @ R.T + t`.
"""

import math
from typing import Tuple, Union, cast

import torch
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import quaternion_to_matrix


def sample_mesh_farthest_points(
    meshes: Meshes,
    num_points: int,
    multiplier: float = 8.0,
) -> torch.Tensor:
    """Sample approximately farthest surface points from one mesh.

    Args:
        meshes: PyTorch3D mesh batch. Current detection code expects one mesh.
        num_points: Number of output points.
        multiplier: Number of dense random samples is `multiplier * num_points`.

    Returns:
        points: Tensor with shape [num_points, 3].
    """
    dense = cast(
        torch.Tensor,
        sample_points_from_meshes(
            meshes,
            num_samples=int(multiplier * num_points),
            return_normals=False,
            return_textures=False,
        ),
    )
    pts, _ = sample_farthest_points(points=dense, K=num_points)

    return pts[0]


def sample_random_rotations(
    size: int,
    device: Union[torch.device, str],
    seed: int = 42,
) -> torch.Tensor:
    """Sample quasi-random 3D rotation matrices.

    Args:
        size: Number of rotations.
        device: Output device.
        seed: Sobol scramble seed.

    Returns:
        R: Tensor with shape [size, 3, 3].
    """
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)

    u = sobol.draw(size).to(device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    r1 = torch.sqrt(1.0 - u1)
    r2 = torch.sqrt(u1)
    t1 = 2.0 * math.pi * u2
    t2 = 2.0 * math.pi * u3

    # Shoemake: (x, y, z, w) -> PyTorch3D wants (w, x, y, z)
    qx = r1 * torch.sin(t1)
    qy = r1 * torch.cos(t1)
    qz = r2 * torch.sin(t2)
    qw = r2 * torch.cos(t2)
    q = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion_to_matrix(q)  # [B, 3, 3]


def sample_axis_perp_c2_rotations(
    verts: torch.Tensor,
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_init: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample C2 rotations around axes perpendicular to a main axis.

    Each sampled 180-degree rotation axis is perpendicular to `axis`,
    intersects the main line defined by (`axis`, `q`), and passes through the
    midpoint of `verts` along the main axis direction.

    Args:
        verts: Tensor with shape [N, 3]. Mesh vertices or sampled points.
        axis: Tensor with shape [3]. Main axis direction.
        q: Tensor with shape [3]. A point on the main axis.
        num_icp_init: Number of initial transforms.

    Returns:
        R: Tensor with shape [num_icp_init, 3, 3].
        t: Tensor with shape [num_icp_init, 3].
    """
    device = verts.device
    dtype = verts.dtype

    axis = axis / axis.norm()

    # Projection range of verts along the main axis line (`axis`, `q`).
    s = (verts - q[None, :]) @ axis
    c = q + s.mean() * axis

    # Build an orthonormal basis e1, e2 of axis^\perp.
    tmp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    if torch.abs((tmp * axis).sum()) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    e1 = tmp - (tmp * axis).sum() * axis
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(axis, e1)
    e2 = e2 / e2.norm()

    # Uniformly sample directions in the perpendicular plane.
    theta = torch.linspace(
        0.0,
        math.pi,
        steps=num_icp_init + 1,
        device=device,
        dtype=dtype,
    )[:-1]

    u = torch.cos(theta)[:, None] * e1[None, :] + torch.sin(theta)[:, None] * e2[None, :]  # [K, 3]

    # 180-degree rotation around axis u: R = 2 u u^T - I
    I = torch.eye(3, device=device, dtype=dtype)[None, :, :]
    R = 2.0 * (u[:, :, None] @ u[:, None, :]) - I  # [K, 3, 3]

    # Keep point c fixed under x' = x @ R.T + t:
    # c = c @ R.T + t  =>  t = c - c @ R.T
    t = (
        c[None, :]
        - torch.bmm(
            c[None, None, :].expand(num_icp_init, -1, -1),
            R.transpose(1, 2),
        )[:, 0, :]
    )  # [K, 3]

    return R, t


def sample_axis_parallel_reflections(
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_init: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample reflection planes that contain a main axis.

    The plane normals are uniformly sampled in the plane perpendicular to
    `axis`, so every reflection plane contains the line defined by (`axis`,
    `q`).

    Args:
        axis: Tensor with shape [3]. Main axis direction.
        q: Tensor with shape [3]. A point on the main axis.
        num_icp_init: Number of initial transforms.

    Returns:
        R: Tensor with shape [num_icp_init, 3, 3]. Reflection matrices.
        t: Tensor with shape [num_icp_init, 3].
    """
    device = axis.device
    dtype = axis.dtype

    axis = axis / axis.norm()

    # Build an orthonormal basis e1, e2 of axis^\perp.
    tmp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    if torch.abs((tmp * axis).sum()) > 0.9:
        tmp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    e1 = tmp - (tmp * axis).sum() * axis
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(axis, e1)
    e2 = e2 / e2.norm()

    # Unique mirror planes correspond to theta in [0, pi).
    theta = torch.linspace(
        0.0,
        math.pi,
        steps=num_icp_init + 1,
        device=device,
        dtype=dtype,
    )[:-1]

    # Plane normals in axis^\perp.
    n = torch.cos(theta)[:, None] * e1[None, :] + torch.sin(theta)[:, None] * e2[None, :]  # [K, 3]

    # Reflection matrix: R = I - 2 n n^T
    I = torch.eye(3, device=device, dtype=dtype)[None, :, :]
    R = I - 2.0 * (n[:, :, None] @ n[:, None, :])  # [K, 3, 3]

    # Keep q fixed under x' = x @ R.T + t:
    # q = q @ R.T + t  =>  t = q - q @ R.T
    t = (
        q[None, :]
        - torch.bmm(
            q[None, None, :].expand(num_icp_init, -1, -1),
            R.transpose(1, 2),
        )[:, 0, :]
    )  # [K, 3]

    return R, t


def sample_axis_perp_reflections(
    verts: torch.Tensor,
    axis: torch.Tensor,
    q: torch.Tensor,
    num_icp_init: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample reflection planes perpendicular to a main axis.

    The plane normal is `axis`; plane positions are uniformly sampled between
    the min and max projection of `verts` onto the line defined by (`axis`,
    `q`).

    Args:
        verts: Tensor with shape [N, 3]. Mesh vertices or sampled points.
        axis: Tensor with shape [3]. Main axis direction and plane normal.
        q: Tensor with shape [3]. A point on the main axis.
        num_icp_init: Number of initial transforms.

    Returns:
        R: Tensor with shape [num_icp_init, 3, 3]. Reflection matrices.
        t: Tensor with shape [num_icp_init, 3].
    """
    device = verts.device
    dtype = verts.dtype

    axis = axis / axis.norm()

    # Scalar coordinates of verts along the axis line (`axis`, `q`).
    s = (verts - q[None, :]) @ axis
    s_min = s.min()
    s_max = s.max()

    # Uniformly sample plane positions in [s_min, s_max].
    s_grid = torch.linspace(
        s_min,
        s_max,
        steps=num_icp_init,
        device=device,
        dtype=dtype,
    )  # [K]

    # Each plane is perpendicular to axis and passes through c_k.
    c = q[None, :] + s_grid[:, None] * axis[None, :]  # [K, 3]

    # Reflection matrix with plane normal = axis.
    I = torch.eye(3, device=device, dtype=dtype)
    R_single = I - 2.0 * torch.outer(axis, axis)  # [3, 3]
    R = R_single[None, :, :].expand(num_icp_init, -1, -1).contiguous()  # [K, 3, 3]

    # Keep c_k fixed under x' = x @ R.T + t:
    # c_k = c_k @ R.T + t  =>  t = c_k - c_k @ R.T
    t = c - torch.bmm(c[:, None, :], R.transpose(1, 2))[:, 0, :]  # [K, 3]

    return R, t
