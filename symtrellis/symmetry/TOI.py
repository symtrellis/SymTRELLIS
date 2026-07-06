import math

# ---------------------------------------------------------------------
# Canonical local-frame constants for T/Td/Th, O/Oh, I/Ih.
#
# Frame convention:
#   ex = (1, 0, 0)
#   ey = (0, 1, 0)
#   ez = (0, 0, 1)
#
# major_axis in the future API corresponds to local +z.
# minor_axis is only used to fix the azimuth around +z.
#
# Axis convention:
#   only one direction is stored per geometric axis.
#   This is sufficient because:
#     C2: angle = pi
#     C3: angles = 2pi/3, 4pi/3
#     C4: angles = pi/2, pi, 3pi/2
#     C5: angles = 2pi/5, 4pi/5, 6pi/5, 8pi/5
# ---------------------------------------------------------------------

PI = math.pi
TAU = 2.0 * math.pi

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT5 = math.sqrt(5.0)

PHI = (1.0 + SQRT5) / 2.0

COS36 = (1.0 + SQRT5) / 4.0
SIN36 = math.sqrt(10.0 - 2.0 * SQRT5) / 4.0

ROTATION_ANGLES_BY_ORDER = {
    2: (PI,),
    3: (2.0 * PI / 3.0, 4.0 * PI / 3.0),
    4: (0.5 * PI, PI, 1.5 * PI),
    5: (2.0 * PI / 5.0, 4.0 * PI / 5.0, 6.0 * PI / 5.0, 8.0 * PI / 5.0),
}


def ring(count: int, r_xy: float, z: float, phase: float):
    return tuple(
        (
            r_xy * math.cos(phase + TAU * k / count),
            r_xy * math.sin(phase + TAU * k / count),
            z,
        )
        for k in range(count)
    )


# =====================================================================
# T / Td / Th
# Canonical choice:
#   ez is one C3 axis.
# =====================================================================

T_C3_RING_R = 2.0 * SQRT2 / 3.0
T_C3_RING_Z = -1.0 / 3.0

T_C2_RING_R = math.sqrt(2.0 / 3.0)
T_C2_RING_Z = 1.0 / SQRT3

TD_MIRROR_EQ_R = 1.0
TD_MIRROR_EQ_Z = 0.0
TD_MIRROR_EQ_PHASE = PI / 6.0

TD_MIRROR_TILT_R = 1.0 / SQRT3
TD_MIRROR_TILT_Z = math.sqrt(2.0 / 3.0)
TD_MIRROR_TILT_PHASE = PI / 3.0

T_C3_AXES = (
    (0.0, 0.0, 1.0),
    *ring(3, T_C3_RING_R, T_C3_RING_Z, 0.0),
)

T_C2_AXES = ring(3, T_C2_RING_R, T_C2_RING_Z, 0.0)

TD_MIRROR_NORMALS = (
    *ring(3, TD_MIRROR_EQ_R, TD_MIRROR_EQ_Z, TD_MIRROR_EQ_PHASE),
    *ring(3, TD_MIRROR_TILT_R, TD_MIRROR_TILT_Z, TD_MIRROR_TILT_PHASE),
)

# In this canonical frame, Th mirror normals coincide with the 3 C2-axis directions.
TH_MIRROR_NORMALS = T_C2_AXES


# =====================================================================
# O / Oh
# Canonical choice:
#   ez is one C4 axis.
# =====================================================================

O_C4_AXES = (
    (0.0, 0.0, 1.0),  # principal axis first
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

O_C3_AXES = (
    (1.0 / SQRT3, 1.0 / SQRT3, 1.0 / SQRT3),
    (-1.0 / SQRT3, 1.0 / SQRT3, 1.0 / SQRT3),
    (-1.0 / SQRT3, -1.0 / SQRT3, 1.0 / SQRT3),
    (1.0 / SQRT3, -1.0 / SQRT3, 1.0 / SQRT3),
)

# These are the 6 genuine C2 axes of O/Oh, not counting the 180-degree
# rotations that also exist on the 3 C4 axes.
O_C2_AXES = (
    (1.0 / SQRT2, 1.0 / SQRT2, 0.0),
    (1.0 / SQRT2, -1.0 / SQRT2, 0.0),
    (1.0 / SQRT2, 0.0, 1.0 / SQRT2),
    (1.0 / SQRT2, 0.0, -1.0 / SQRT2),
    (0.0, 1.0 / SQRT2, 1.0 / SQRT2),
    (0.0, 1.0 / SQRT2, -1.0 / SQRT2),
)

OH_MIRROR_NORMALS = (
    *O_C4_AXES,
    *O_C2_AXES,
)


# =====================================================================
# I / Ih
# Canonical choice:
#   ez is one C5 axis.
#   the ez/ex plane contains one additional C5 axis.
# =====================================================================

# 6 C5 axes: one pole + one 5-ring
I_C5_RING_R = 2.0 / SQRT5
I_C5_RING_Z = 1.0 / SQRT5
I_C5_RING_PHASE = 0.0

I_C5_AXES = (
    (0.0, 0.0, 1.0),
    *ring(5, I_C5_RING_R, I_C5_RING_Z, I_C5_RING_PHASE),
)

# 10 C3 axes: two 5-rings
I_C3_Z_LO = math.sqrt((5.0 - 2.0 * SQRT5) / 15.0)
I_C3_Z_HI = math.sqrt((5.0 + 2.0 * SQRT5) / 15.0)
I_C3_R_LO = math.sqrt((10.0 + 2.0 * SQRT5) / 15.0)
I_C3_R_HI = math.sqrt((10.0 - 2.0 * SQRT5) / 15.0)
I_C3_PHASE = PI / 5.0

I_C3_AXES = (
    *ring(5, I_C3_R_LO, I_C3_Z_LO, I_C3_PHASE),
    *ring(5, I_C3_R_HI, I_C3_Z_HI, I_C3_PHASE),
)

# 15 C2 axes: one equatorial 5-ring + two tilted 5-rings
#
# Correct values:
#   sqrt((5 ± sqrt(5)) / 10)
# not the previous sin(36) / cos(36) pair.
I_C2_EQ_R = 1.0
I_C2_EQ_Z = 0.0
I_C2_EQ_PHASE = -PI / 10.0

I_C2_RING_A_R = math.sqrt((5.0 + SQRT5) / 10.0)
I_C2_RING_A_Z = math.sqrt((5.0 - SQRT5) / 10.0)
I_C2_RING_A_PHASE = PI / 5.0

I_C2_RING_B_R = math.sqrt((5.0 - SQRT5) / 10.0)
I_C2_RING_B_Z = math.sqrt((5.0 + SQRT5) / 10.0)
I_C2_RING_B_PHASE = 0.0

I_C2_AXES = (
    *ring(5, I_C2_EQ_R, I_C2_EQ_Z, I_C2_EQ_PHASE),
    *ring(5, I_C2_RING_A_R, I_C2_RING_A_Z, I_C2_RING_A_PHASE),
    *ring(5, I_C2_RING_B_R, I_C2_RING_B_Z, I_C2_RING_B_PHASE),
)

# In this canonical frame, Ih mirror normals coincide with the 15 C2-axis directions.
IH_MIRROR_NORMALS = I_C2_AXES


# =====================================================================
# Group-level lookup tables
# This stores geometric element families, not a full element enumeration
# for Td/Th/Oh/Ih.
# =====================================================================

TOI_GROUP_ELEMENT_FAMILIES = {
    "T": {
        "major_order": 3,
        "rotation_axes": {
            3: T_C3_AXES,
            2: T_C2_AXES,
        },
        "mirror_normals": (),
        "has_inversion": False,
    },
    "Td": {
        "major_order": 3,
        "rotation_axes": {
            3: T_C3_AXES,
            2: T_C2_AXES,
        },
        "mirror_normals": TD_MIRROR_NORMALS,
        "has_inversion": False,
    },
    "Th": {
        "major_order": 3,
        "rotation_axes": {
            3: T_C3_AXES,
            2: T_C2_AXES,
        },
        "mirror_normals": TH_MIRROR_NORMALS,
        "has_inversion": True,
    },
    "O": {
        "major_order": 4,
        "rotation_axes": {
            4: O_C4_AXES,
            3: O_C3_AXES,
            2: O_C2_AXES,  # genuine C2 axes only
        },
        "mirror_normals": (),
        "has_inversion": False,
    },
    "Oh": {
        "major_order": 4,
        "rotation_axes": {
            4: O_C4_AXES,
            3: O_C3_AXES,
            2: O_C2_AXES,  # genuine C2 axes only
        },
        "mirror_normals": OH_MIRROR_NORMALS,
        "has_inversion": True,
    },
    "I": {
        "major_order": 5,
        "rotation_axes": {
            5: I_C5_AXES,
            3: I_C3_AXES,
            2: I_C2_AXES,
        },
        "mirror_normals": (),
        "has_inversion": False,
    },
    "Ih": {
        "major_order": 5,
        "rotation_axes": {
            5: I_C5_AXES,
            3: I_C3_AXES,
            2: I_C2_AXES,
        },
        "mirror_normals": IH_MIRROR_NORMALS,
        "has_inversion": True,
    },
}


TOI_GROUP_COUNTS = {
    "T": {"C2_axes": 3, "C3_axes": 4, "C4_axes": 0, "C5_axes": 0, "mirrors": 0, "inversion": 0},
    "Td": {"C2_axes": 3, "C3_axes": 4, "C4_axes": 0, "C5_axes": 0, "mirrors": 6, "inversion": 0},
    "Th": {"C2_axes": 3, "C3_axes": 4, "C4_axes": 0, "C5_axes": 0, "mirrors": 3, "inversion": 1},
    "O": {"C2_axes": 6, "C3_axes": 4, "C4_axes": 3, "C5_axes": 0, "mirrors": 0, "inversion": 0},
    "Oh": {"C2_axes": 6, "C3_axes": 4, "C4_axes": 3, "C5_axes": 0, "mirrors": 9, "inversion": 1},
    "I": {"C2_axes": 15, "C3_axes": 10, "C4_axes": 0, "C5_axes": 6, "mirrors": 0, "inversion": 0},
    "Ih": {"C2_axes": 15, "C3_axes": 10, "C4_axes": 0, "C5_axes": 6, "mirrors": 15, "inversion": 1},
}
