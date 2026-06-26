import math
import re
from typing import Optional, Tuple

import torch

from .TOI import ROTATION_ANGLES_BY_ORDER, TOI_GROUP_ELEMENT_FAMILIES
from .transforms import axis_point_half_turn_transform, axis_point_rotation_transform, plane_reflection_transform

"""Point-group parsing tables and axial construction recipes.

These constants define the small grammar accepted by `get_3d_point_group`.
`POINT_GROUP_RE` parses concrete Schoenflies labels into a prefix (`C`, `D`,
or `S`), an integer fold, and an optional suffix. The suffix tables map valid
`C*` and `D*` suffixes to canonical axial families, while `S_SPECIAL_ORDERS`
handles the two special improper groups that do not use a fold in the axial
construction.

`AXIAL_RECIPES` describes how each axial family is generated after parsing:
which rotation mode to use, which axes are required, and which optional mirror
or dihedral operations should be appended. Polyhedral groups are not described
here; they use `TOI_GROUP_ELEMENT_FAMILIES` instead.
"""

POLYHEDRAL_GROUPS = {"T", "Td", "Th", "O", "Oh", "I", "Ih"}
POINT_GROUP_RE = re.compile(r"(?P<prefix>[CDS])(?P<fold>[1-9]\d*)(?P<suffix>[hvd]?)")
C_SUFFIX_TO_FAMILY = {"": "Cn", "h": "Cnh", "v": "Cnv"}
D_SUFFIX_TO_FAMILY = {"": "Dn", "d": "Dnd", "h": "Dnh"}
S_SPECIAL_ORDERS = {1: ("S1", None), 2: ("S2", None)}
AXIAL_RECIPES = {
    "C1": {"rotation": "none", "needs_major": False, "needs_minor": False, "horizontal_mirror": False, "dihedral_c2": False, "vertical_mirror": None},
    "S1": {"rotation": "none", "needs_major": True, "needs_minor": False, "horizontal_mirror": True, "dihedral_c2": False, "vertical_mirror": None},
    "S2": {"rotation": "inversion", "needs_major": False, "needs_minor": False, "horizontal_mirror": False, "dihedral_c2": False, "vertical_mirror": None},
    "S2n": {"rotation": "improper", "needs_major": True, "needs_minor": False, "horizontal_mirror": False, "dihedral_c2": False, "vertical_mirror": None},
    "Cn": {"rotation": "proper", "needs_major": True, "needs_minor": False, "horizontal_mirror": False, "dihedral_c2": False, "vertical_mirror": None},
    "Cnh": {"rotation": "proper", "needs_major": True, "needs_minor": False, "horizontal_mirror": True, "dihedral_c2": False, "vertical_mirror": None},
    "Cnv": {"rotation": "proper", "needs_major": True, "needs_minor": False, "horizontal_mirror": False, "dihedral_c2": False, "vertical_mirror": "vertical"},
    "Dn": {"rotation": "proper", "needs_major": True, "needs_minor": True, "horizontal_mirror": False, "dihedral_c2": True, "vertical_mirror": None},
    "Dnd": {"rotation": "proper", "needs_major": True, "needs_minor": True, "horizontal_mirror": False, "dihedral_c2": True, "vertical_mirror": "diagonal"},
    "Dnh": {"rotation": "proper", "needs_major": True, "needs_minor": True, "horizontal_mirror": True, "dihedral_c2": True, "vertical_mirror": "vertical"},
}


def get_3d_point_group(
    label: str,
    center: torch.Tensor,
    major_axis: Optional[torch.Tensor] = None,
    minor_axis: Optional[torch.Tensor] = None,
    include_identity: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Euclidean transforms for a concrete 3D Schoenflies point group.

    The returned transforms use the row-vector convention:

        x_new = x @ O.T + t

    The implementation has two stages:

    1. Parse `label` into a construction spec. Axial groups (`C*`, `D*`,
       `S*`) are mapped to a canonical family, a fold value, and an axial
       recipe. Polyhedral groups (`T*`, `O*`, `I*`) use precomputed geometric
       element families in a canonical local frame.
    2. Generate transforms from that spec. Axial groups are built from
       rotations around `major_axis`, optional mirror planes, and optional
       dihedral half-turn axes. Polyhedral groups map canonical local axes and
       mirror normals into the world frame defined by `major_axis` and
       `minor_axis`.

    Family conventions:
        `C1`: identity-only group. It requires `include_identity=True`.
        `Cn`: proper rotations around `major_axis`.
        `Cnh`: `Cn` plus a mirror plane normal to `major_axis`.
        `Cnv`: `Cn` plus vertical mirror planes containing `major_axis`.
        `Dn`: `Cn` plus in-plane C2 axes. `minor_axis` fixes the azimuth.
        `Dnd`: `Dn` plus diagonal mirror planes.
        `Dnh`: `Dn` plus horizontal and vertical mirror planes.
        `S1`: a single mirror reflection; `major_axis` is the plane normal.
        `S2`: inversion through `center`; no axis is required.
        `S2n`: improper rotations around `major_axis`.
        `T`, `Td`, `Th`, `O`, `Oh`, `I`, `Ih`: polyhedral point groups.

    Args:
        label: Strict canonical Schoenflies label, such as `C3`, `C3v`,
            `D2h`, `S4`, `T`, `Td`, `O`, `Oh`, `I`, or `Ih`. `Ci` and `Cs`
            are intentionally unsupported; use `S2` and `S1`.
        center: Tensor with shape [3]. Point on the symmetry axis or mirror
            plane, and the inversion center for `S2`.
        major_axis: Optional tensor with shape [3]. Required by every
            nontrivial axial/polyhedral label except `S2`. For `S1`, this is
            the mirror-plane normal.
        minor_axis: Optional tensor with shape [3]. Required by `D*`,
            `D*d`, `D*h`, `T*`, `O*`, and `I*` labels. It fixes the azimuth
            around `major_axis`.
        include_identity: Whether to include the identity transform.

    Returns:
        O_list: Tensor with shape [num_transforms, 3, 3]. Orthogonal linear
            parts of the Euclidean transforms.
        t_list: Tensor with shape [num_transforms, 3]. Translation parts.
        s_list: Int64 tensor with shape [num_transforms]. `1` marks
            orientation-preserving transforms with `det(O) > 0`; `0` marks
            orientation-reversing transforms with `det(O) < 0`.
    """
    # Reject labels with implicit normalization; the public API expects strict canonical labels.
    if label != label.strip():
        raise ValueError(f"Invalid point-group label {label!r}: leading/trailing spaces are not allowed.")

    # `Ci` and `Cs` are valid Schoenflies aliases, but this API exposes their equivalent concrete forms as `S2` and `S1`.
    if label in {"Ci", "Cs"}:
        raise ValueError(f"Unsupported point-group label {label!r}. Use S2 for Ci and S1 for Cs.")

    # Parse the label into a construction type, canonical family, fold, and optional axial recipe.
    fold: Optional[int] = None
    if label in POLYHEDRAL_GROUPS:
        # Polyhedral groups use precomputed canonical local axes and mirror normals from `TOI_GROUP_ELEMENT_FAMILIES`.
        group_type, family, fold, recipe = "polyhedral", label, None, None
        needs_major, needs_minor = True, True

    else:
        # Axial labels have prefix `C`, `D`, or `S`, an integer fold, and an optional mirror suffix.
        m = POINT_GROUP_RE.fullmatch(label)
        if m is None:
            raise ValueError(f"Unsupported point-group label: {label!r}")

        prefix, fold, suffix, group_type = m["prefix"], int(m["fold"]), m["suffix"] or "", "axial"

        if prefix == "C":
            # Cyclic labels map to one principal axis; suffix `h` adds a horizontal mirror and `v` adds vertical mirrors.
            if suffix not in C_SUFFIX_TO_FAMILY:
                raise ValueError(f"Unsupported point-group label: {label!r}")

            if fold == 1:
                # `C1` is identity-only; mirror-like `C1h` and `C1v` are rejected in favor of explicit `S1`.
                if suffix == "":
                    family = "C1"
                elif suffix == "h":
                    raise ValueError("Unsupported point-group label 'C1h'. Use S1.")
                elif suffix == "v":
                    raise ValueError("Unsupported point-group label 'C1v'. Use S1.")
                else:
                    raise ValueError(f"Unsupported point-group label: {label!r}")
            else:
                family = C_SUFFIX_TO_FAMILY[suffix]
        elif prefix == "D":
            # Dihedral labels require fold >= 2; the suffix table selects pure, diagonal-mirror, or horizontal-mirror families.
            if fold == 1 or suffix not in D_SUFFIX_TO_FAMILY:
                raise ValueError(f"Unsupported point-group label: {label!r}")
            family = D_SUFFIX_TO_FAMILY[suffix]
        else:
            # Improper rotation labels do not accept suffixes; `S1` and `S2` are special cases, and higher orders must be even.
            if suffix:
                raise ValueError(f"Unsupported point-group label: {label!r}")

            if fold in S_SPECIAL_ORDERS:
                family, fold = S_SPECIAL_ORDERS[fold]
            elif fold % 2 == 0:
                family, fold = "S2n", fold // 2
            else:
                raise ValueError(f"Unsupported point-group label {label!r}. Use S1, S2, or even-order S labels.")

        recipe = AXIAL_RECIPES[family]
        needs_major, needs_minor = recipe["needs_major"], recipe["needs_minor"]

    if needs_major and major_axis is None:
        raise ValueError(f"`major_axis` is required for point-group label {label!r}.")
    if needs_minor and minor_axis is None:
        raise ValueError(f"`minor_axis` is required for point-group label {label!r}.")
    if family == "C1" and not include_identity:
        raise ValueError("C1 contains only the identity transform. Set include_identity=True or use a nontrivial point group.")

    O_list, t_list = [], []
    ez, ex, ey, frame = None, None, None, None

    if include_identity:
        O = torch.eye(3, device=center.device, dtype=center.dtype)
        t = torch.zeros_like(center)
        O_list.append(O)
        t_list.append(t)

    if needs_major:
        if major_axis is None:
            raise ValueError(f"`major_axis` is required for point-group label {label!r}.")
        ez = major_axis / major_axis.norm().clamp_min(1e-12)

        if minor_axis is None:
            ex = torch.tensor([1.0, 0.0, 0.0], device=center.device, dtype=center.dtype)
            if torch.abs((ex * ez).sum()) > 0.9:
                ex = torch.tensor([0.0, 1.0, 0.0], device=center.device, dtype=center.dtype)
        else:
            ex = minor_axis
        ex = ex - (ex * ez).sum() * ez
        ex = ex / ex.norm().clamp_min(1e-12)
        ey = torch.linalg.cross(ez, ex)
        ey = ey / ey.norm().clamp_min(1e-12)
        frame = torch.stack((ex, ey, ez), dim=0)

    if group_type == "polyhedral":
        # Polyhedral generation maps canonical local axes and mirrors into the world frame; the table is not full closure.
        assert frame is not None
        spec = TOI_GROUP_ELEMENT_FAMILIES[family]
        rotation_axes = spec["rotation_axes"]
        mirror_normals = spec["mirror_normals"]
        has_inversion = spec["has_inversion"]

        for order, axis_local_list in rotation_axes.items():
            # Convert all local axes of the same rotation order into world axes in one batched matrix multiply.
            axis_local = torch.as_tensor(axis_local_list, device=center.device, dtype=center.dtype)
            axis_world_list = axis_local @ frame

            for axis_world in axis_world_list:
                if order == 2:
                    # Half-turns use the closed-form C2 transform.
                    O, t = axis_point_half_turn_transform(axis=axis_world, q=center)
                    O_list.append(O)
                    t_list.append(t)
                else:
                    # Higher-order axes append every non-identity rotation angle listed for that order.
                    for angle in ROTATION_ANGLES_BY_ORDER[order]:
                        O, t = axis_point_rotation_transform(axis=axis_world, q=center, angle=angle)
                        O_list.append(O)
                        t_list.append(t)

        if len(mirror_normals) > 0:
            # Add all reflection planes associated with this polyhedral family.
            normal_local = torch.as_tensor(mirror_normals, device=center.device, dtype=center.dtype)
            normal_world_list = normal_local @ frame

            for normal_world in normal_world_list:
                O, t = plane_reflection_transform(normal=normal_world, q=center)
                O_list.append(O)
                t_list.append(t)

        if has_inversion:
            # Centrosymmetric polyhedral families additionally include inversion through `center`.
            O = -torch.eye(3, device=center.device, dtype=center.dtype)
            t = center - center @ O.T
            O_list.append(O)
            t_list.append(t)
    else:
        # Axial generation follows the parsed recipe instead of checking the raw family name in several places.
        assert recipe is not None
        rotation = recipe["rotation"]
        horizontal_mirror = recipe["horizontal_mirror"]
        dihedral_c2 = recipe["dihedral_c2"]
        vertical_mirror = recipe["vertical_mirror"]

        if rotation == "none":
            # `C1` and `S1` have no rotational component; `S1` is represented by the horizontal mirror below.
            pass
        elif rotation == "inversion":
            # `S2` is inversion through `center` and does not require an axis.
            O = -torch.eye(3, device=center.device, dtype=center.dtype)
            t = center - center @ O.T
            O_list.append(O)
            t_list.append(t)
        elif rotation == "proper":
            # Proper axial groups add all non-identity rotations around `ez`.
            assert fold is not None
            assert ez is not None
            for i in range(1, fold):
                O, t = axis_point_rotation_transform(axis=ez, q=center, angle=2 * i * torch.pi / fold)
                O_list.append(O)
                t_list.append(t)
        elif rotation == "improper":
            # `S2n` alternates rotations around `ez` with reflection across the normal plane; compute reflection once.
            assert fold is not None
            assert ez is not None
            Op, tp = plane_reflection_transform(normal=ez, q=center)

            for i in range(1, 2 * fold):
                O, t = axis_point_rotation_transform(axis=ez, q=center, angle=i * torch.pi / fold)

                if i % 2 == 1:
                    t = t @ Op.T + tp
                    O = Op @ O

                O_list.append(O)
                t_list.append(t)
        else:
            raise AssertionError(f"Unsupported axial rotation recipe: {rotation!r}")

        if horizontal_mirror:
            # Horizontal mirror plane is perpendicular to the principal axis, so its normal is `ez`.
            assert ez is not None
            O, t = plane_reflection_transform(normal=ez, q=center)
            O_list.append(O)
            t_list.append(t)

        if dihedral_c2:
            # Dihedral groups add `fold` half-turn axes in the local xy plane.
            assert fold is not None
            assert ex is not None and ey is not None

            for i in range(fold):
                angle = i * math.pi / fold
                axis_c2 = math.cos(angle) * ex + math.sin(angle) * ey
                O, t = axis_point_half_turn_transform(axis=axis_c2, q=center)
                O_list.append(O)
                t_list.append(t)

        if vertical_mirror is not None:
            # Vertical mirrors contain the principal axis; diagonal mirrors use the Dnd half-step angular offset.
            assert fold is not None
            assert ex is not None and ey is not None
            for i in range(fold):
                angle = (i / fold + 0.5) * math.pi

                if vertical_mirror == "diagonal":
                    angle += math.pi / (2 * fold)

                normal_refl = math.cos(angle) * ex + math.sin(angle) * ey
                O, t = plane_reflection_transform(normal=normal_refl, q=center)
                O_list.append(O)
                t_list.append(t)

    O_out = torch.stack(O_list, dim=0)
    t_out = torch.stack(t_list, dim=0)
    s_out = (torch.linalg.det(O_out) > 0).to(dtype=torch.int64)

    return O_out, t_out, s_out
