import torch
import torch.nn.functional as F


def compute_feature_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # feature_l2 is the old training "mse": mean squared error over latent channels,
    # then averaged over the task-selected destination rows.
    prediction = prediction[mask].float()
    target = target[mask].float()

    diff = prediction - target
    squared_error = diff.square()
    absolute_error = diff.abs()
    target_squared = target.square()

    per_point_l2 = squared_error.mean(dim=1)
    per_point_l1 = absolute_error.mean(dim=1)
    per_point_relative_l2 = per_point_l2 / target_squared.mean(dim=1).clamp_min(1e-12)
    target_norm = target.norm(dim=1)
    target_norm_sum = target_norm.sum().clamp_min(1e-12)
    per_point_cosine = F.cosine_similarity(prediction, target, dim=1)

    feature_l2 = per_point_l2.mean()
    feature_l1 = per_point_l1.mean()
    feature_cosine = per_point_cosine.mean()

    return {
        "feature_l2": feature_l2,
        "feature_l1": feature_l1,
        "feature_relative_l2": per_point_relative_l2.mean(),
        "feature_norm_weighted_l2": (per_point_l2 * target_norm).sum() / target_norm_sum,
        "feature_norm_weighted_l1": (per_point_l1 * target_norm).sum() / target_norm_sum,
        "feature_cosine": feature_cosine,
        "feature_cosine_distance": 1.0 - feature_cosine,
        "feature_norm_weighted_cosine": (per_point_cosine * target_norm).sum() / target_norm_sum,
        "mask_keep_ratio": mask.float().mean(),
    }
