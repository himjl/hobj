from dataclasses import dataclass
import numpy as np
from typing import List

from hobj.learning_models import BinaryLearningModel


class LearningCurveBenchmark:

    """
    This benchmark measures the divergence between a human and model learning curve across a variety of binary learning subtasks.
    """

    @dataclass
    class LearningCurveBenchmarkResult:

        # Benchmark results:
        msen: float
        msen_sigma: float
        msen_null_samples: List[float]

        # Point estimates that may be used for further downstream analysis:
        model_phat: np.ndarray  # [trial, subtask]
        model_varhat_phat: np.ndarray  # [trial, subtask]
        model_lapse_rate: float

        target_phat: np.ndarray  # [trial, subtask]
        target_varhat_phat: np.ndarray  # [trial, subtask]
        target_phat_bootstrap_resamples: np.ndarray  # [nboot, trial, subtask] - bootstrap resamples of these statistics over worker resamples.

        # Legacy version of msen, where variance correction from the human was not subtracted.
        _legacy_msen: float
        _legacy_msen_sigma: float

    def __init__(
            self,

    ):
        pass

    def __call__(
            self,
            learner: BinaryLearningModel
    ) -> LearningCurveBenchmarkResult:

        pass


    @dataclass
    class SimulateExperimentResult:
        """
        The result of simulating an experiment on this battery of experiments.
        """
        subtasks: List[str]
        k: np.ndarray  # [trial, subtask]
        n: np.ndarray  # [trial, subtask]


    def _simulate_experiment(
            self,
            learner: BinaryLearningModel,
            seed: int,
    ):