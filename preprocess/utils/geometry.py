import torch


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
