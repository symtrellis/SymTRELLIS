from typing import Tuple

from .base import AffineFlowStep, BaseFlowPredictor

"""Classifier-free guidance as a velocity-prediction adapter."""


class ClassifierFreeGuidanceWrapper(BaseFlowPredictor):
    """Wrap a flow predictor and apply classifier-free guidance.

    The wrapped predictor is called with either `cond` or `neg_cond`. When the
    guidance strength is active, the output velocity is:

        v_cfg = strength * v_cond + (1 - strength) * v_neg

    `strength = 1` gives the conditional prediction, and `strength = 0` gives
    the negative/unconditional prediction. The wrapper keeps the same
    `predict_velocity` interface as a normal predictor so it can be composed
    with other velocity adapters.
    """

    def __init__(
        self,
        predictor: BaseFlowPredictor,
        strength: float,
        interval: Tuple[float, float],
        rescale: float = 0.0,
    ) -> None:
        """Create a CFG adapter.

        Args:
            predictor: Base velocity predictor to wrap.
            strength: CFG interpolation/extrapolation strength.
            interval: Inclusive time interval where `strength` is applied.
                Outside the interval, the wrapper uses strength `1.0`.
            rescale: Optional CFG rescale factor applied in `x_start` space.
        """
        self.predictor = predictor

        self.strength = strength
        self.interval = interval
        self.rescale = rescale

    def predict_velocity(
        self,
        step: AffineFlowStep,
        cond,
        neg_cond,
        **kwargs,
    ):
        """Predict guided velocity for the current flow step.

        Args:
            step: Current affine-flow state.
            cond: Positive/conditional model conditioning.
            neg_cond: Negative or unconditional model conditioning.
            **kwargs: Additional arguments forwarded to the wrapped predictor.

        Returns:
            Guided velocity in the same layout as the wrapped predictor output.
        """

        # Use CFG only inside the requested flow-time interval.
        if self.interval[0] <= step.t <= self.interval[1]:
            strength = self.strength
        else:
            strength = 1.0

        # Avoid the second model call when the guidance degenerates to one side.
        if strength == 1.0:
            v_pred = self.predictor.predict_velocity(
                step=step,
                cond=cond,
                **kwargs,
            )
        elif strength == 0.0:
            v_pred = self.predictor.predict_velocity(
                step=step,
                cond=neg_cond,
                **kwargs,
            )
        else:
            v_pos = self.predictor.predict_velocity(
                step=step,
                cond=cond,
                **kwargs,
            )
            v_neg = self.predictor.predict_velocity(
                step=step,
                cond=neg_cond,
                **kwargs,
            )
            v_pred = strength * v_pos + (1 - strength) * v_neg

            # CFG rescale matches the guided x_start statistics to cond x_start.
            if self.rescale > 0:
                xstart_pos = step.v_to_xstart(v_pos)
                xstart_cfg = step.v_to_xstart(v_pred)

                std_pos = xstart_pos.std(dim=list(range(1, xstart_pos.ndim)), keepdim=True)
                std_cfg = xstart_cfg.std(dim=list(range(1, xstart_cfg.ndim)), keepdim=True)
                xstart_cfg_rescaled = xstart_cfg * (std_pos / std_cfg)

                xstart = self.rescale * xstart_cfg_rescaled + (1 - self.rescale) * xstart_cfg
                v_pred = step.xstart_to_v(xstart)

        return v_pred
