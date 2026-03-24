from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pydantic
import xarray as xr
from tqdm import tqdm

from hobj.benchmarks.generalization.estimator import GeneralizationStatistics
from hobj.benchmarks.generalization.simulator import GeneralizationSessionResult, GeneralizationSubtask
from hobj.learning_models import BinaryLearningModel
from hobj.stats.ci import estimate_basic_bootstrap_CI


# %% Models for configuring a LearningCurveBenchmark:
class GeneralizationBenchmarkConfig(pydantic.BaseModel):
    subtasks: List[GeneralizationSubtask]
    results: List[GeneralizationSessionResult]
    num_simulations_per_subtask: int = pydantic.Field(ge=2)
    num_bootstrap_samples: int = pydantic.Field(ge=2)
    bootstrap_target_by_worker: bool


# %%
class GeneralizationBenchmark:

    def __init__(
            self,
            config: GeneralizationBenchmarkConfig,
    ):
        self.config = config
        self._generalization_statistics = GeneralizationStatistics(
            results=self.config.results,
            perform_lapse_rate_correction=True,
            n_bootstrap_iterations=self.config.num_bootstrap_samples,
            bootstrap_by_worker=self.config.bootstrap_target_by_worker,
        )

    @property
    def target_statistics(self) -> GeneralizationStatistics:
        """
        Returns the target data for the benchmark, which is a dictionary of subtask_name -> [session, trial] boolean matrix of performance.
        :return:
        """
        return self._generalization_statistics

    @dataclass
    class GeneralizationBenchmarkResult:
        msen: float
        msen_sigma: float
        msen_CI95: Tuple[float, float]
        model_statistics: GeneralizationStatistics

    def __call__(
            self,
            learner: BinaryLearningModel,
            show_pbar: bool = False
    ) -> GeneralizationBenchmarkResult:
        """
        :param learner: LearningModel
        :return:
        """

        # Get model generalization behaviors:
        results: List[GeneralizationSessionResult] = []
        for subtask in tqdm(self.config.subtasks, desc='Subtask simulations:', disable=not show_pbar):
            # Get [simulation, trial] boolean performance matrix for the model
            subtask_results = self.simulate_model_behavior(
                subtask=subtask,
                learner=learner,
                nsimulations=self.config.num_simulations_per_subtask,
            )
            results.extend(subtask_results)

        # Assemble into a single matrix
        model_statistics = GeneralizationStatistics(
            results=results,
            perform_lapse_rate_correction=False,
            n_bootstrap_iterations=self.config.num_bootstrap_samples,
            bootstrap_by_worker=False, # bootstrap simulations
        )

        # Get target transformation
        model_statistics = model_statistics.sel(transformation = self.target_statistics.transformation)

        # Calculate comparison statistics between target and model learning curves:
        msen_point = self._compare_generalization_patterns(
            model_phat=model_statistics.phat,
            model_varhat_phat=model_statistics.varhat_phat,
            target_phat=self.target_statistics.phat,
            target_varhat_phat=self.target_statistics.varhat_phat,
            condition_dims=('transformation',),
        )

        # Calculate bootstrap replicates:
        msen_boot = self._compare_generalization_patterns(
            model_phat=model_statistics.boot_phat,
            model_varhat_phat=model_statistics.boot_varhat_phat,
            target_phat=self.target_statistics.boot_phat,
            target_varhat_phat=self.target_statistics.boot_varhat_phat,
            condition_dims=('transformation',),
        )

        msen_sigma = np.std(msen_boot, ddof=1)
        msen_CI95 = estimate_basic_bootstrap_CI(
            alpha=0.05,
            point_estimate=msen_point,
            bootstrapped_point_estimates=np.array(msen_boot),
        )

        # Return result
        return self.GeneralizationBenchmarkResult(
            msen=float(msen_point),
            msen_sigma=float(msen_sigma),
            msen_CI95=msen_CI95,
            model_statistics=model_statistics,
        )

    @staticmethod
    def simulate_model_behavior(
            subtask: GeneralizationSubtask,
            learner: BinaryLearningModel,
            nsimulations: int,
    ) -> List[GeneralizationSessionResult]:
        """
        Returns a [nsimulations, ntrials] matrix of model performance on the subtask.
        :param subtask:
        :param learner:
        :param nsimulations:
        :return: a [nsimulations, ntrials] matrix of model performance on the subtask.
        """

        results = []

        # Perform simulations for this subtask:
        for i_simulation in range(nsimulations):
            # Reset learner internal state
            learner.reset_state(seed=None)

            # Simulate session:
            results.append(subtask.simulate_session(learner=learner, seed=None, ))

        return results

    @classmethod
    def _compare_generalization_patterns(
            cls,
            model_phat: xr.DataArray,
            model_varhat_phat: xr.DataArray,
            target_phat: xr.DataArray,
            target_varhat_phat: xr.DataArray,
            condition_dims: Tuple[str, ...],
    ):

        msen = np.square(model_phat - target_phat).mean(condition_dims) - model_varhat_phat.mean(condition_dims) - target_varhat_phat.mean(condition_dims)
        return msen
