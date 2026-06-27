"""TRELLIS-specific inference adapters."""

import torch

from symtrellis.flow import AffineFlowStep, BaseFlowPredictor

from .trellis2 import TRELLIS2SparseStructureLatentNoiseSampler as TRELLISSparseStructureLatentNoiseSampler
from .trellis2 import trellis2_dense_grid_coords as trellis_dense_grid_coords
from .trellis2 import trellis2_sparse_structure_latent_to_sparse_view as trellis_sparse_structure_latent_to_sparse_view
from .trellis2 import trellis2_sparse_structure_logits_to_coords as trellis_sparse_structure_logits_to_coords
from .trellis2 import trellis2_sparse_view_to_sparse_structure_latent as trellis_sparse_view_to_sparse_structure_latent

# Official TRELLIS sampler defaults expressed in SymTRELLIS flow/CFG convention.
TRELLIS_SPARSE_STRUCTURE_STEPS = 25
TRELLIS_SPARSE_STRUCTURE_RESCALE_T = 3.0
TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH = 6.0
TRELLIS_SPARSE_STRUCTURE_CFG_INTERVAL = (0.0, 0.5)
TRELLIS_SPARSE_STRUCTURE_CFG_RESCALE = 0.0

TRELLIS_SLAT_STEPS = 25
TRELLIS_SLAT_RESCALE_T = 3.0
TRELLIS_SLAT_CFG_STRENGTH = 6.0
TRELLIS_SLAT_CFG_INTERVAL = (0.0, 0.5)
TRELLIS_SLAT_CFG_RESCALE = 0.0


class TRELLISFlowPredictor(BaseFlowPredictor):
    """Adapt a TRELLIS flow model to the repository's velocity convention.

    TRELLIS models use the official sampler convention where `t = 1` is noise
    and `t = 0` is data. `AffineFlowStep` uses the opposite convention, so this
    adapter flips both time and velocity sign.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the TRELLIS model module."""
        self.model = model

    def predict_velocity(self, step: AffineFlowStep, cond=None, **kwargs):
        """Predict velocity in the `AffineFlowStep` convention.

        Args:
            step: Current flow state.
            cond: TRELLIS conditioning tensor passed to the wrapped model.
            **kwargs: Extra TRELLIS model arguments.

        Returns:
            Velocity with the same layout as `step.x_t`.
        """

        # Convert from SymTRELLIS flow time to TRELLIS model time.
        t = 1 - step.t
        x_t = step.x_t

        # TRELLIS flow models expect timestep values scaled to [0, 1000].
        t = torch.tensor(
            [1000 * t] * x_t.shape[0],
            device=x_t.device,
            dtype=torch.float32,
        )

        # TRELLIS conditioning can be shared across generated samples.
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(
                x_t.shape[0],
                *([1] * (len(cond.shape) - 1)),
            )

        # Flip the sign to convert TRELLIS velocity to SymTRELLIS velocity.
        v_pred = -self.model(x_t, t, cond, **kwargs)

        return v_pred


__all__ = [
    "TRELLISFlowPredictor",
    "TRELLIS_SLAT_CFG_INTERVAL",
    "TRELLIS_SLAT_CFG_RESCALE",
    "TRELLIS_SLAT_CFG_STRENGTH",
    "TRELLIS_SLAT_RESCALE_T",
    "TRELLIS_SLAT_STEPS",
    "TRELLIS_SPARSE_STRUCTURE_CFG_INTERVAL",
    "TRELLIS_SPARSE_STRUCTURE_CFG_RESCALE",
    "TRELLIS_SPARSE_STRUCTURE_CFG_STRENGTH",
    "TRELLIS_SPARSE_STRUCTURE_RESCALE_T",
    "TRELLIS_SPARSE_STRUCTURE_STEPS",
    "TRELLISSparseStructureLatentNoiseSampler",
    "trellis_dense_grid_coords",
    "trellis_sparse_structure_latent_to_sparse_view",
    "trellis_sparse_structure_logits_to_coords",
    "trellis_sparse_view_to_sparse_structure_latent",
]
