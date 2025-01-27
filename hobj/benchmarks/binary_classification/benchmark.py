from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional, Iterator

import numpy as np
import pydantic
import xarray as xr
from tqdm import tqdm

import hobj.data.schema as schema
from hobj.benchmarks.binary_classification.estimator import LearningCurveStatistics
from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtask, BinaryClassificationSubtaskResult
from hobj.learning_models import BinaryLearningModel
from hobj.stats.ci import estimate_basic_bootstrap_CI


# %% Models for configuring a LearningCurveBenchmark:
class TargetSubtaskData(pydantic.BaseModel):
    subtask: BinaryClassificationSubtask  # The subtask which generated the associated results
    results: List[BinaryClassificationSubtaskResult]  # [session, trial] boolean matrix of performance

    model_config = dict(
        arbitrary_types_allowed=True
    )

    @pydantic.model_validator(mode='after')
    def validate_results(self) -> 'TargetSubtaskData':
        # Check shape
        for result in self.results:
            if self.subtask.ntrials != len(result.perf_seq):
                raise ValueError(f"Expected {self.subtask.ntrials} trials, but got {result.perf_seq} trials")

        return self


class LearningCurveBenchmarkConfig(pydantic.BaseModel):
    subtask_name_to_data: Dict[str, 'TargetSubtaskData'] = pydantic.Field(default_factory=dict, description="A dictionary of subtask_name -> TargetSubtaskConfig")
    num_simulations_per_subtask: int = pydantic.Field(ge=2)
    num_bootstrap_samples: int = pydantic.Field(ge=2)
    bootstrap_by_worker: bool
    ntrials: Optional[int] = pydantic.Field(default=None)

    @pydantic.model_validator(mode='after')
    def ensure_rectangular(self) -> 'LearningCurveBenchmarkConfig':

        ntrials_observed = set()
        for name, data in self.subtask_name_to_data.items():
            ntrials_observed.add(data.subtask.ntrials)

        if not len(ntrials_observed) == 1:
            raise ValueError(f"Expected all subtasks to have the same number of trials, but got {ntrials_observed}")

        if self.ntrials is not None:
            if self.ntrials != ntrials_observed.pop():
                raise ValueError(f"Expected ntrials to be {ntrials_observed.pop()}, but got {self.ntrials}")
        else:
            self.ntrials = ntrials_observed.pop()
        return self


# %%
class LearningCurveBenchmark:

    def __init__(
            self,
            config: LearningCurveBenchmarkConfig,
    ):
        self.config = config

        # Attach properties
        self.subtask_names = sorted(config.subtask_name_to_data.keys())
        self.subtask_name_to_subtask: Dict[str, BinaryClassificationSubtask] = {
            name: config.subtask_name_to_data[name].subtask for name in self.subtask_names
        }

        self.subtask_name_to_results: Dict[str, List[BinaryClassificationSubtaskResult]] = {}
        self._target_data = {}

        for name in self.subtask_names:
            results = config.subtask_name_to_data[name].results
            self.subtask_name_to_results[name] = results

            self._target_data[name] = {}
            for result in results:
                worker_id = result.worker_id
                if worker_id in self._target_data[name]:
                    raise ValueError(f"Worker {worker_id} has already been seen for subtask {name}")
                self._target_data[name][result.worker_id] = list([bool(v) for v in result.perf_seq])

        self._target_statistics = LearningCurveStatistics(
            subtask_name_to_results=self.subtask_name_to_results,
            nbootstrap_samples=self.config.num_bootstrap_samples,
            bootstrap_by_worker=self.config.bootstrap_by_worker,
        )

    @property
    def target_data(self) -> Dict[str, Dict[str, List[bool]]]:
        return self._target_data


    @property
    def target_statistics(self) -> LearningCurveStatistics:
        """
        Returns point estimates of the mean (and an estimate of the variance of that estimator) of the binomial proportion for the target data,
        and bootstrapped resamples of those statistics.
        :return:
        """
        return self._target_statistics

    @dataclass
    class LearningCurveBenchmarkResult:
        msen: float
        msen_sigma: float
        msen_CI95: Tuple[float, float]
        lapse_rate: float

        model_statistics: LearningCurveStatistics

    def __call__(
            self,
            learner: BinaryLearningModel,
            show_pbar: bool = False
    ) -> LearningCurveBenchmarkResult:
        """
        :param learner: LearningModel
        :return:
        """

        # Get model learning curve statistics:
        subtask_name_to_model_results: Dict[str, List[BinaryClassificationSubtaskResult]] = {}
        for i_subtask, subtask_name in enumerate(tqdm(self.subtask_names, desc='Subtask simulations:', disable=not show_pbar)):
            # Get [simulation, trial] boolean performance matrix for the model
            subtask_results = self.simulate_model_behavior(
                subtask=self.subtask_name_to_subtask[subtask_name],
                learner=learner,
                nsimulations=self.config.num_simulations_per_subtask,
            )

            subtask_name_to_model_results[subtask_name] = subtask_results

        # Assemble into a single matrix
        model_statistics = LearningCurveStatistics(
            subtask_name_to_results=subtask_name_to_model_results,
            nbootstrap_samples=self.config.num_bootstrap_samples,
            bootstrap_by_worker=False,
        )

        # Calculate comparison statistics between target and model learning curves:
        msen_point, lapse_rate = self._compare_learning_curves(
            model_phat=model_statistics.phat,
            model_varhat_phat=model_statistics.varhat_phat,
            target_phat=self.target_statistics.phat,
            target_varhat_phat=self.target_statistics.varhat_phat,
            condition_dims=('subtask', 'trial'),
            fit_lapse_rate=True,
        )

        # Calculate bootstrap replicates:
        msen_boot, _ = self._compare_learning_curves(
            model_phat=model_statistics.boot_phat,
            model_varhat_phat=model_statistics.boot_varhat_phat,
            target_phat=self.target_statistics.boot_phat,
            target_varhat_phat=self.target_statistics.boot_varhat_phat,
            condition_dims=('subtask', 'trial'),
            fit_lapse_rate=True,
        )

        msen_sigma = np.std(msen_boot, ddof=1)

        msen_CI95 = estimate_basic_bootstrap_CI(
            alpha=0.05,
            point_estimate=msen_point,
            bootstrapped_point_estimates=np.array(msen_boot),
        )

        # Return result
        return self.LearningCurveBenchmarkResult(
            msen=float(msen_point),
            msen_sigma=float(msen_sigma),
            msen_CI95=msen_CI95,
            model_statistics=model_statistics,
            lapse_rate=lapse_rate,
        )

    @staticmethod
    def simulate_model_behavior(
            subtask: BinaryClassificationSubtask,
            learner: BinaryLearningModel,
            nsimulations: int,
    ) -> List[BinaryClassificationSubtaskResult]:
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
            results.append(
                subtask.simulate_session(
                    learner=learner,
                    seed=None,
                )
            )

        return results

    @classmethod
    def _compare_learning_curves(
            cls,
            model_phat: xr.DataArray,
            model_varhat_phat: xr.DataArray,
            target_phat: xr.DataArray,
            target_varhat_phat: xr.DataArray,
            condition_dims: Tuple[str, ...],
            fit_lapse_rate: bool
    ) -> Tuple[Union[np.ndarray, np.generic], Union[xr.DataArray, None]]:

        if fit_lapse_rate:
            lapse_rate = cls._fit_lapse_rate(
                pmodel=model_phat,
                ptarget=target_phat,
                condition_dims=condition_dims
            )
            model_phat = model_phat * (1 - lapse_rate) + 0.5 * lapse_rate
            model_varhat_phat = model_varhat_phat * (1 - lapse_rate) ** 2
        else:
            lapse_rate = None

        msen = np.square(model_phat - target_phat).mean(condition_dims) - model_varhat_phat.mean(condition_dims) - target_varhat_phat.mean(condition_dims)
        return msen, lapse_rate

    @staticmethod
    def _fit_lapse_rate(
            pmodel: xr.DataArray,
            ptarget: xr.DataArray,
            condition_dims: Tuple[str, ...]
    ) -> Union[np.ndarray, np.generic]:
        """
        Fits a "lapse rate" parameter (gamma), which takes on values between [0, 1]. It may be interpreted
        as the probability that a  uniform random guess is made. The value of gamma which minimizes the
        empirical MSE between pmodel and ptarget is selected.


        pmodel_adjusted := (1 - gamma) * pmodel + 1/2 * gamma
        Loss = ((pmodel_adjusted - ptarget)**2).sum()

        :param pmodel:
        :param ptarget:
        :return:
        """

        nway = 2
        numerator = -(2 * pmodel / nway - 2 * np.square(pmodel) + 2 * pmodel * ptarget - 2 * ptarget / nway).sum(dim=condition_dims)
        denominator = (2 / (nway ** 2) - 4 * pmodel / nway + 2 * (pmodel ** 2)).sum(dim=condition_dims)
        gamma_star = numerator / denominator
        gamma_star = np.clip(gamma_star, 0, 1)
        return gamma_star
