from collections.abc import Sequence

import numpy as np

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


def radical_inverse(base: int, index: int) -> float:
    value = 0.0
    inverse_base = 1.0 / base
    inverse_power = inverse_base
    while index > 0:
        digit = index % base
        value += digit * inverse_power
        index //= base
        inverse_power *= inverse_base
    return value


def halton_sequence(dim: int, index: int) -> list[float]:
    return [radical_inverse(PRIMES[axis], index) for axis in range(dim)]


def hammersley_sequence(dim: int, index: int, num_samples: int) -> list[float]:
    return [index / num_samples, *halton_sequence(dim - 1, index)]


def fibonacci_sphere_samples(num_samples: int) -> np.ndarray:
    """Return evenly distributed unit directions on a sphere."""
    indices = np.arange(num_samples, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / num_samples
    radius = np.sqrt(1.0 - z * z)
    azimuth = np.pi * (3.0 - np.sqrt(5.0)) * indices
    return np.stack(
        (radius * np.cos(azimuth), radius * np.sin(azimuth), z),
        axis=1,
    )


def sphere_hammersley_sequence(
    index: int,
    num_samples: int,
    offset: Sequence[float] = (0.0, 0.0),
) -> list[float]:
    u, v = hammersley_sequence(2, index, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]
