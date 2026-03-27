from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
import xarray as xr

from hobj.learning_models import BinaryLearningModel
from hobj.stats.ci import estimate_basic_bootstrap_CI

ResultT = TypeVar("ResultT")


class SessionSubtask(Protocol[ResultT]):
    """Protocol for benchmark subtasks that can simulate one learner session."""

    def simulate_session(
        self,
        learner: BinaryLearningModel,
        seed: int | None,
    ) -> ResultT:
        """Simulate a single session of learner behavior."""


@dataclass(frozen=True)
class BootstrapSummary:
    """Summary of a bootstrap distribution for a scalar benchmark score.

    Attributes:
        sigma: Sample standard deviation of the bootstrap distribution.
        ci95: Basic bootstrap 95% confidence interval.
    """

    sigma: float
    ci95: tuple[float, float]


def simulate_sessions(
    subtask: SessionSubtask[ResultT],
    learner: BinaryLearningModel,
    nsimulations: int,
) -> list[ResultT]:
    """Simulate repeated learner sessions for a single benchmark subtask.

    Args:
        subtask: Benchmark subtask to simulate.
        learner: Learning model to evaluate.
        nsimulations: Number of independent sessions to run.

    Returns:
        A list of session results in simulation order.
    """

    results = []
    for _ in range(nsimulations):
        learner.reset_state(seed=None)
        results.append(
            subtask.simulate_session(
                learner=learner,
                seed=None,
            )
        )

    return results


def compare_msen(
    model_phat: xr.DataArray,
    model_varhat_phat: xr.DataArray,
    target_phat: xr.DataArray,
    target_varhat_phat: xr.DataArray,
    condition_dims: tuple[str, ...],
) -> xr.DataArray:
    """Compute the variance-corrected mean squared error normalized over dims.

    Args:
        model_phat: Mean model performance estimate.
        model_varhat_phat: Estimated variance of the model mean.
        target_phat: Mean target performance estimate.
        target_varhat_phat: Estimated variance of the target mean.
        condition_dims: Dimensions across which to average the squared error.

    Returns:
        The normalized mean squared error with variance penalties removed.
    """

    msen = (
        np.square(model_phat - target_phat).mean(condition_dims)
        - model_varhat_phat.mean(condition_dims)
        - target_varhat_phat.mean(condition_dims)
    )
    return msen


def summarize_bootstrap_score(
    point_estimate: Any,
    bootstrapped_point_estimates: Any,
) -> BootstrapSummary:
    """Summarize a bootstrap distribution for a benchmark score.

    Args:
        point_estimate: Point estimate corresponding to the bootstrap samples.
        bootstrapped_point_estimates: Bootstrap replicates of the score.

    Returns:
        The bootstrap standard deviation and 95% confidence interval.
    """

    boot_array = np.asarray(bootstrapped_point_estimates)
    return BootstrapSummary(
        sigma=float(np.std(boot_array, ddof=1)),
        ci95=estimate_basic_bootstrap_CI(
            alpha=0.05,
            point_estimate=point_estimate,
            bootstrapped_point_estimates=boot_array,
        ),
    )
