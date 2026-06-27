"""Reference pipeline for coarse symmetry detection on generated meshes.

This module is a compact template for generated-mesh detection. It mirrors the
basic evaluation logic: sample the mesh, build the intrinsic context once, then
run unconstrained rotation-axis and reflection-plane detectors. It is not the
full symmetry-specific analysis path.

If a coarse symmetry has already been identified, use the more targeted
detectors in `detectors.py`:
`detect_reflection_planes_containing_axis`,
`detect_reflection_planes_perpendicular_to_axis`, and
`detect_c2_axes_perpendicular_to_axis`. Final candidates can then be refined
with `icp_refine_rotation` or `icp_refine_reflection`.
"""

from typing import Dict, List

import torch
from pytorch3d.structures import Meshes

from .detectors import detect_reflection_planes, detect_rotation_axes
from .intrinsic import build_intrinsic_basis
from .sampling import sample_mesh_farthest_points


@torch.no_grad()
def detect_mesh_symmetry(
    verts: torch.Tensor,
    faces: torch.Tensor,
    num_samples: int = 4096,
    intrinsic_dim: int = 64,
    num_icp_iter: int = 128,
    num_icp_init: int = 512,
    max_fold: int = 30,
) -> Dict[str, List[Dict]]:
    """Run the coarse generated-mesh symmetry detection template.

    The function computes the reusable context once:
    `samples` are farthest surface samples, while `phi` and `Gphi` are the
    intrinsic basis tensors used to score candidate transforms. The same
    context is then passed to both coarse detectors.

    Args:
        verts: Mesh vertex tensor with shape [num_verts, 3].
        faces: Mesh face index tensor with shape [num_faces, 3].
        num_samples: Number of surface samples used by the coarse detector.
        intrinsic_dim: Number of intrinsic basis channels.
        num_icp_iter: ICP iterations used for each candidate transform.
        num_icp_init: Number of ICP initializations for each detector.
        max_fold: Maximum rotation fold tested by the rotation detector.

    Returns:
        A dictionary with the same top-level fields as the evaluation script:
        `rotational_symmetry` and `reflectional_symmetry`.
    """
    # Build one sampled surface context for all coarse detectors.
    mesh = Meshes(verts[None], faces[None])
    samples = sample_mesh_farthest_points(mesh, num_points=num_samples)
    samples, phi, Gphi = build_intrinsic_basis(
        verts=verts,
        faces=faces,
        samples=samples,
        intrinsic_dim=intrinsic_dim,
    )

    # Coarse rotation detection finds unconstrained global rotation axes.
    rotational_symmetry = detect_rotation_axes(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        num_icp_iter=num_icp_iter,
        num_icp_init=num_icp_init,
        max_fold=max_fold,
    )

    # Coarse reflection detection finds unconstrained global mirror planes.
    reflectional_symmetry = detect_reflection_planes(
        samples=samples,
        phi=phi,
        Gphi=Gphi,
        num_icp_iter=num_icp_iter,
        num_icp_init=num_icp_init,
    )

    symmetry = {
        "rotational_symmetry": rotational_symmetry,
        "reflectional_symmetry": reflectional_symmetry,
    }

    return symmetry
