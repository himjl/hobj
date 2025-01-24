from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional, Iterator

import numpy as np
from scipy import stats as ss

from hobj.benchmarks.binary_classification.task import BinaryClassificationSubtask
from hobj.learning_models import BinaryLearningModel
from hobj.statistics.variance_estimates import binomial as binomial_funcs
import pydantic
from tqdm import tqdm
from hobj.stats_new.learning_curve import calculate_learning_curve
from hobj.stats_new.ci import estimate_basic_bootstrap_CI

import hobj.data.schema as schema

import xarray as xr
# %% Models for configuring a LearningCurveBenchmark:

class TargetSubtaskData(pydantic.BaseModel):
    subtask: BinaryClassificationSubtask  # The subtask which generated the associated results
    results: Dict[str, List[bool]]  # worker_id -> perf_seq

    @pydantic.model_validator(mode='after')
    def validate_results(self) -> 'TargetSubtaskData':
        for worker_id, perf_seq in self.results.items():
            if len(perf_seq) != self.subtask.ntrials:
                raise ValueError(f"Expected {self.subtask.ntrials} trials, but got {len(perf_seq)} trials for worker {worker_id}")

        return self


class LearningCurveBenchmarkConfig(pydantic.BaseModel):
    subtask_name_to_data: Dict[str, 'TargetSubtaskData'] = pydantic.Field(default_factory=dict, description="A dictionary of subtask_name -> TargetSubtaskConfig")
    num_simulations_per_subtask: int = pydantic.Field(ge=2)
    num_bootstrap_samples: int = pydantic.Field(ge=2)
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


class LearningCurveStatistics(xr.Dataset):
    def __init__(
            self,
            subtask_names: List[str],
            phat: np.ndarray,
            varhat_phat: np.ndarray,
            boot_phat: np.ndarray,
            boot_varhat_phat: np.ndarray,
    ):
        data_vars = dict(
            phat = (['subtask', 'trial'], phat),
            varhat_phat = (['subtask', 'trial'], varhat_phat),
            boot_phat = (['boot_iter', 'subtask', 'trial'], boot_phat),
            boot_varhat_phat = (['boot_iter', 'subtask', 'trial'], boot_varhat_phat),
        )

        coords = dict(
            subtask = subtask_names,
        )

        super().__init__(
            data_vars = data_vars,
            coords = coords
        )


# %%
class LearningCurveBenchmark:

    @property
    def image_refs(self) -> Iterator[schema.ImageRef]:
        """
        Returns all image references used in the benchmark.
        :return:
        """
        for subtask in self.subtask_name_to_subtask.values():
            for ref in subtask.classA + subtask.classB:
                yield ref

    @staticmethod
    def simulate_model_behavior(
            subtask: BinaryClassificationSubtask,
            learner: BinaryLearningModel,
            nsimulations: int,
    ) -> np.ndarray:
        """
        Returns a [nsimulations, ntrials] matrix of model performance on the subtask.
        :param subtask:
        :param learner:
        :param nsimulations:
        :return: a [nsimulations, ntrials] matrix of model performance on the subtask.
        """

        # Instantiate array to store performance sequences for this subtask:
        perf_mat = np.zeros((nsimulations, subtask.ntrials), dtype=np.bool)

        # Perform simulations for this subtask:
        for i_simulation in range(nsimulations):
            # Reset learner internal state
            learner.reset_state(seed=None)

            # Simulate session:
            perf_mat[i_simulation, :] = subtask.simulate_session(
                learner=learner,
                seed=None,
            )

        return perf_mat

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
        self.subtask_name_to_target_data: Dict[str, Dict[str, List[bool]]] = {
            subtask_name: target_subtask.results for subtask_name, target_subtask in self.config.subtask_name_to_data.items()
        }

        # Cache target learning curve statistics
        phat = np.zeros((len(self.subtask_names), self.config.ntrials))
        varhat_phat = np.zeros((len(self.subtask_names), self.config.ntrials))
        boot_phat = np.zeros((self.config.num_bootstrap_samples, len(self.subtask_names), self.config.ntrials))
        boot_varhat_phat = np.zeros((self.config.num_bootstrap_samples, len(self.subtask_names), self.config.ntrials))

        for i_subtask, subtask_name in enumerate(self.subtask_names):
            subtask = self.subtask_name_to_subtask[subtask_name]

            nworkers = len(self.subtask_name_to_target_data[subtask_name])
            perf_mat = np.zeros((nworkers, subtask.ntrials), dtype=np.bool)

            for i_worker, (worker_id, perf_seq) in enumerate(self.subtask_name_to_target_data[subtask_name].items()):
                perf_mat[i_worker] = perf_seq

            subtask_learning_curve_statistics = calculate_learning_curve(
                perf_matrix=perf_mat,
                bootstrap_seed=i_subtask,
                nbootstrap_samples=self.config.num_bootstrap_samples,
                repetition_axis=0,
            )

            phat[i_subtask] = subtask_learning_curve_statistics.phat
            varhat_phat[i_subtask] = subtask_learning_curve_statistics.varhat_phat
            boot_phat[:, i_subtask] = subtask_learning_curve_statistics.boot_phat
            boot_varhat_phat[:, i_subtask] = subtask_learning_curve_statistics.boot_varhat_phat

        self._target_statistics = LearningCurveStatistics(
            subtask_names=self.subtask_names,
            phat=phat,
            varhat_phat=varhat_phat,
            boot_phat=boot_phat,
            boot_varhat_phat=boot_varhat_phat,
        )


    @property
    def data(self) -> Dict[str, Dict[str, List[bool]]]:
        """
        Returns the target data as a nested dictionary. Example usage

            target_data = self.get_target_as_json()
            print(target_data[subtask_name][worker_id])  # List[bool] of trial performances on {subtask_name} by {worker_id}.

        Useful for performing custom analyses on the target data.
        :return:
        """

        return self.subtask_name_to_target_data

    @property
    def target_statistics(self) -> xr.Dataset:
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
        # Todo: deduplicate with constructor
        phat = np.zeros((len(self.subtask_names), self.config.ntrials))
        varhat_phat = np.zeros((len(self.subtask_names), self.config.ntrials))
        boot_phat = np.zeros((self.config.num_bootstrap_samples, len(self.subtask_names), self.config.ntrials))
        boot_varhat_phat = np.zeros((self.config.num_bootstrap_samples, len(self.subtask_names), self.config.ntrials))

        for i_subtask, subtask_name in enumerate(tqdm(self.subtask_names, desc='Subtask simulations:', disable=not show_pbar)):
            # Get [simulation, trial] boolean performance matrix for the model
            perf_matrix = self.simulate_model_behavior(
                subtask=self.subtask_name_to_subtask[subtask_name],
                learner=learner,
                nsimulations=self.config.num_simulations_per_subtask,
            )

            # Calculate statistics
            subtask_estimate = calculate_learning_curve(
                perf_matrix=perf_matrix,
                bootstrap_seed=None,
                nbootstrap_samples=self.config.num_bootstrap_samples,
                repetition_axis=0,
            )
            phat[i_subtask] = subtask_estimate.phat
            varhat_phat[i_subtask] = subtask_estimate.varhat_phat
            boot_phat[:, i_subtask] = subtask_estimate.boot_phat
            boot_varhat_phat[:, i_subtask] = subtask_estimate.boot_varhat_phat

        # Assemble into a single matrix
        model_statistics = LearningCurveStatistics(
            subtask_names=self.subtask_names,
            phat=phat,
            varhat_phat=varhat_phat,
            boot_phat=boot_phat,
            boot_varhat_phat=boot_varhat_phat,
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
            alpha = 0.05,
            point_estimate = msen_point,
            bootstrapped_point_estimates = np.array(msen_boot),
        )

        # Return result
        return self.LearningCurveBenchmarkResult(
            msen=float(msen_point),
            msen_sigma=float(msen_sigma),
            msen_CI95=msen_CI95,
            model_statistics=model_statistics,
            lapse_rate  = lapse_rate,
        )

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
        else :
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
        print(gamma_star)
        return gamma_star
