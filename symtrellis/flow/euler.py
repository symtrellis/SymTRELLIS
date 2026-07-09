from typing import Dict, Iterator, List

import numpy as np
import torch
from tqdm import tqdm

from .base import AffineFlowStep, BaseFlowPredictor, BaseSolver

"""Euler sampler for affine-flow velocity prediction."""


class EulerSolver(BaseSolver):
    """Integrate a flow predictor with explicit Euler steps.

    The predictor is expected to return velocity `v` under the convention
    defined by `AffineFlowStep`. The solver updates the current sample by:

        x_{t_next} = x_t + (t_next - t) * v_t

    and stores every visited state as an `AffineFlowStep`.
    """

    @torch.no_grad()
    def iter_steps(
        self,
        noise,
        predictor: BaseFlowPredictor,
        steps: int,
        predictor_args: Dict,
        sigma_min: float = 1e-5,
        rescale_t: float = 1.0,
    ) -> Iterator[AffineFlowStep]:
        """Yield every Euler state from initial noise to final sample."""

        timesteps = np.linspace(0, 1, steps + 1)
        timesteps = timesteps / (rescale_t - (rescale_t - 1) * timesteps)
        timesteps = timesteps.tolist()

        current = AffineFlowStep(
            sigma_min=sigma_min,
            t=timesteps[0],
            x_t=noise,
        )
        yield current

        for i in range(steps):
            pred_v = predictor.predict_velocity(
                step=current,
                **predictor_args,
            )
            x_t = current.x_t + (timesteps[i + 1] - timesteps[i]) * pred_v
            current = AffineFlowStep(
                sigma_min=sigma_min,
                t=timesteps[i + 1],
                x_t=x_t,
            )
            yield current

    def sample(
        self,
        noise,
        predictor: BaseFlowPredictor,
        steps: int,
        predictor_args: Dict,
        sigma_min: float = 1e-5,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs,
    ) -> List[AffineFlowStep]:
        """Sample from initial `noise` and return the full Euler trajectory.

        Args:
            noise: Initial sample at `t = 0`, in the model-specific format.
            predictor: Object that predicts flow velocity for each step.
            steps: Number of Euler integration steps from `t = 0` to `t = 1`.
            predictor_args: Extra keyword arguments passed to the predictor.
            sigma_min: Residual noise scale used by `AffineFlowStep`.
            rescale_t: Optional time reparameterization factor. `1.0` keeps
                uniformly spaced timesteps.
            verbose: Whether to show a progress bar.
            tqdm_desc: Progress-bar label.

        Returns:
            A list of `steps + 1` states, including the initial noise state.
        """

        step_iter = self.iter_steps(
            noise=noise,
            predictor=predictor,
            steps=steps,
            predictor_args=predictor_args,
            sigma_min=sigma_min,
            rescale_t=rescale_t,
        )
        trajectories = [next(step_iter)]
        for step in tqdm(
            step_iter,
            total=steps,
            desc=tqdm_desc,
            disable=not verbose,
        ):
            trajectories.append(step)

        return trajectories
