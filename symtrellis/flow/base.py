from dataclasses import dataclass
from typing import Any, List

"""Base interfaces for affine-flow sampling and velocity adapters.

This module defines the common step convention used by solvers, model
wrappers, classifier-free guidance, and symmetry projection guidance. The
actual tensors can be dense tensors, sparse-view tensors, or model-specific
containers as long as the concrete predictor and solver agree on the format.
"""


@dataclass
class AffineFlowStep:
    """State and parameterization conversions for one affine-flow step.

    The flow runs from the noise endpoint to the data endpoint:

        x_0 = eps
        x_1 = x_start

    with the affine interpolation:

        x_t = (1 - (1 - sigma_min) * t) * eps + t * x_start
        v = x_start - (1 - sigma_min) * eps

    `sigma_min` is the residual noise scale at the data endpoint. `t` is the
    current flow time and must be broadcastable to `x_t` for the conversion
    formulas below.
    """

    sigma_min: float

    t: Any
    x_t: Any

    def eps_to_xstart(self, eps: Any):
        """Convert a noise prediction `eps` at this step into `x_start`."""
        return (self.x_t - (1 - (1 - self.sigma_min) * self.t) * eps) / self.t

    def xstart_to_eps(self, xstart: Any):
        """Convert a data prediction `x_start` at this step into `eps`."""
        return (self.x_t - self.t * xstart) / (1 - (1 - self.sigma_min) * self.t)

    def v_to_xstart(self, v: Any):
        """Convert a velocity prediction `v` at this step into `x_start`."""
        return (1 - self.sigma_min) * self.x_t + (1 - (1 - self.sigma_min) * self.t) * v

    def v_to_eps(self, v: Any):
        """Convert a velocity prediction `v` at this step into `eps`."""
        return self.x_t - self.t * v

    def xstart_to_v(self, xstart: Any):
        """Convert a data prediction `x_start` at this step into velocity `v`."""
        return (xstart - (1 - self.sigma_min) * self.x_t) / (1 - (1 - self.sigma_min) * self.t)


class BaseFlowPredictor:
    """Interface for objects that produce flow velocity predictions.

    Concrete implementations can be raw model wrappers or velocity adapters
    such as CFG and symmetry projection guidance. The solver only assumes
    that `predict_velocity` follows the `AffineFlowStep` velocity convention.
    """

    def predict_velocity(self, step: AffineFlowStep, *args, **kwargs):
        """Return a velocity prediction with the same sample layout as `step.x_t`."""
        raise NotImplementedError


class BaseSolver:
    """Interface for samplers that integrate a flow predictor over time."""

    def sample(self, noise, *args, **kwargs) -> List[AffineFlowStep]:
        """Run sampling from initial noise and return the visited flow steps."""
        raise NotImplementedError


class BaseInitialNoiseSampler:
    """Interface for constructing the initial noise input for a solver."""

    def sample(self, *args, **kawargs):
        """Return an initial noise sample in the model-specific sample format."""
        raise NotImplementedError
