from typing import Callable, Tuple

from ..mapper.operator import SymmetryProjector
from .base import AffineFlowStep, BaseFlowPredictor, BaseInitialNoiseSampler


class SymmetryProjectionGuidanceWrapper(BaseFlowPredictor):

    def __init__(
        self,
        predictor: BaseFlowPredictor,
        strength: float,
        interval: Tuple[float, float],
        symmetrize_target: str = "x_start",
        rescale: float = 0.0,
    ) -> None:
        """
        Apply symmetry projection guidance to the predicted velocity.

        if symmetrize_target is velocity, then the symmetrized velocity is P
            v = P(v)
        if symmetrize_target is x_start, then
            v = [P((1-t)*v + x_t) - x_t] / (1-t)

        strength determines the mix factor between original velocity and the symmetrized velocity
        inside the specified flow-time interval. Outside this interval, no symmetry projection is applied.

        rescale is to mix the x_start or new velocity and the std rescaled velocity

        """
        assert symmetrize_target in ["velocity", "x_start"]

        self.predictor = predictor

        self.strength = strength
        self.interval = interval
        self.symmetrize_target = symmetrize_target
        self.rescale = rescale

    def predict_velocity(
        self,
        step: AffineFlowStep,
        projector: SymmetryProjector,
        to_sparse_view: Callable,
        to_original_view: Callable,
        self_include: bool = False,
        **kwargs,
    ):
        # Apply symmetry projection guidance only inside the requested flow-time interval.
        if self.interval[0] <= step.t <= self.interval[1]:
            symm_strength = self.strength
        else:
            symm_strength = 0.0

        v_native = self.predictor.predict_velocity(
            step=step,
            **kwargs,
        )

        if symm_strength == 0.0:
            return v_native

        if self.symmetrize_target == "x_start":
            target = step.v_to_xstart(v_native)
        else:
            target = v_native

        target_sp = to_sparse_view(target)

        projected_sp = projector.forward_project(
            feats=target_sp,
            self_include=self_include,
        )

        projected = to_original_view(projected_sp)

        if self.symmetrize_target == "x_start":
            v_symm = step.xstart_to_v(projected)
        else:
            v_symm = projected

        v_pred = symm_strength * v_symm + (1 - symm_strength) * v_native

        if self.rescale > 0:
            xstart_symm = step.v_to_xstart(v_pred)
            xstart_ref = step.v_to_xstart(v_native)

            std_symm = xstart_symm.std(dim=list(range(1, xstart_symm.ndim)), keepdim=True)
            std_ref = xstart_ref.std(dim=list(range(1, xstart_ref.ndim)), keepdim=True)
            xstart_symm_rescaled = xstart_symm * (std_ref / std_symm)

            xstart = self.rescale * xstart_symm_rescaled + (1 - self.rescale) * xstart_symm
            v_pred = step.xstart_to_v(xstart)

        return v_pred


class SymmetryProjectionNoiseSampler(BaseInitialNoiseSampler):

    def __init__(
        self,
        sampler: BaseInitialNoiseSampler,
        symmetry_strength: float = 1.0,
    ) -> None:

        self.sampler = sampler
        self.symmetry_strength = symmetry_strength

    def sample(
        self,
        **kwargs,
    ):
        projector: SymmetryProjector = kwargs["projector"]
        to_sparse_view: Callable = kwargs["to_sparse_view"]
        to_original_view: Callable = kwargs["to_original_view"]
        self_include: bool = kwargs["self_include"]

        noise_native = self.sampler.sample(**kwargs)
        if self.symmetry_strength == 0.0:
            return noise_native

        noise_sp = to_sparse_view(noise_native)

        projected_sp = projector.forward_project(
            feats=noise_sp,
            self_include=self_include,
        )

        noise_symm = to_original_view(projected_sp)

        if self.symmetry_strength == 1.0:
            return noise_symm

        noise = noise_symm * self.symmetry_strength + (1 - self.symmetry_strength) * noise_native

        return noise
