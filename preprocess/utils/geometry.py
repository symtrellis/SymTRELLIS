import math
from typing import Tuple

import torch
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix


def approx_miniball_radius(
    points: torch.Tensor,
    num_iters: int = 32,
) -> torch.Tensor:
    """Return an upper-bounded approximation of the minimum bounding-ball radius."""
    assert points.ndim == 2 and points.shape[1] == 3

    center = points[0]
    _, first_index = ((points - center) ** 2).sum(1).max(0)
    _, second_index = ((points - points[first_index]) ** 2).sum(1).max(0)
    center = 0.5 * (points[first_index] + points[second_index])

    max_distance_squared, farthest_index = ((points - center) ** 2).sum(1).max(0)
    for iteration in range(2, int(num_iters) + 1):
        center = center + (points[farthest_index] - center) / float(iteration)
        max_distance_squared, farthest_index = ((points - center) ** 2).sum(1).max(0)

    return max_distance_squared.sqrt()


def sobol_rotation_samples(
    n: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample uniformly distributed rotations from a scrambled Sobol sequence."""
    scramble_seed = int(
        torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (),
            device=device,
            generator=generator,
        ).item()
    )
    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=scramble_seed)
    u = sobol.draw(n).to(device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    r1 = torch.sqrt(1.0 - u1)
    r2 = torch.sqrt(u1)
    t1 = 2.0 * math.pi * u2
    t2 = 2.0 * math.pi * u3

    qx = r1 * torch.sin(t1)
    qy = r1 * torch.cos(t1)
    qz = r2 * torch.sin(t2)
    qw = r2 * torch.cos(t2)
    quaternions = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion_to_matrix(quaternions)  # [n, 3, 3]


def small_rotation_perturbation_samples(
    n: int,
    std_rad: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample isotropic Gaussian axis-angle perturbations."""
    rotvec = torch.randn(n, 3, device=device, generator=generator) * std_rad
    return axis_angle_to_matrix(rotvec)  # [n, 3, 3]


def sample_mesh_srt(
    vertices: torch.Tensor,
    num_scale: int,
    min_scale: float,
    num_rots: int,
    num_perts: int,
    perturbation_rad_std: float,
    shape_latent_resolution: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize vertices and sample scale, rotation, inversion, and translation."""
    if not torch.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")

    device = vertices.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    center = (vertices.amin(dim=0) + vertices.amax(dim=0)) / 2
    diameter = 2 * approx_miniball_radius(vertices).clamp_min(0.5).item()
    base_scale = 0.999998 * (1 - 4 / shape_latent_resolution) / diameter
    vertices = (vertices - center) * base_scale

    # Sample scale first.
    scales = torch.linspace(0.0, -math.log(min_scale), num_scale, device=device).neg().exp()
    scales = scales.reshape(num_scale, 1).expand(num_scale, num_rots)

    # Then sample rotation.
    rotations = sobol_rotation_samples(
        num_scale * num_rots,
        device,
        generator,
    ).reshape(num_scale, num_rots, 1, 3, 3)
    rotation_scales = scales[..., None, None, None] * rotations
    rotation_scales = rotation_scales.expand(num_scale, num_rots, num_perts, 3, 3)

    # Then add small rotation perturbations.
    perturbations = small_rotation_perturbation_samples(
        num_scale * num_rots * num_perts,
        perturbation_rad_std,
        device,
        generator,
    ).reshape(num_scale, num_rots, num_perts, 3, 3)
    rotation_scales = torch.einsum("srpik,srpkj->srpij", perturbations, rotation_scales)

    # Then apply random inversion.
    signs = torch.randint(
        0,
        2,
        size=(num_scale, num_rots, num_perts, 1, 1),
        device=device,
        generator=generator,
    ).to(vertices.dtype)
    rotation_scales = rotation_scales * (2 * signs - 1)

    # Then sample a feasible translation.
    transformed_vertices = torch.einsum("srpij,nj->srpni", rotation_scales, vertices)
    vertex_min = transformed_vertices.amin(dim=-2)
    vertex_max = transformed_vertices.amax(dim=-2)
    extent = vertex_max - vertex_min
    if (extent > 1.0 - 2e-6).any():
        raise ValueError(f"no feasible translation: extent_max={extent.max().item()}")

    translation_min = -vertex_min - 0.5 + 1e-6
    translation_max = -vertex_max + 0.5 - 1e-6
    translations = torch.rand(
        translation_min.shape,
        dtype=translation_min.dtype,
        device=device,
        generator=generator,
    )
    translations = translations * (translation_max - translation_min) + translation_min
    transformed_vertices = transformed_vertices + translations[..., None, :]
    assert transformed_vertices.amax() <= 0.5 - 1e-6
    assert transformed_vertices.amin() >= -0.5 + 1e-6

    # Prepare the transformation matrices.
    transforms = torch.zeros(
        (num_scale, num_rots, num_perts, 4, 4),
        dtype=transformed_vertices.dtype,
        device=device,
    )
    transforms[..., :3, :3] = rotation_scales
    transforms[..., :3, 3] = translations
    transforms[..., 3, 3] = 1.0
    return transformed_vertices, transforms
