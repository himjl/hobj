import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.stats as ss

import hobj.learning_models
import hobj.statistics.lapse_rate as lapse_rate_funcs
import hobj.statistics.variance_estimates.binomial as binomial_funcs
from hobj.data.behavior.template import LearningDataset
from hobj.learning_models import learning_model as lm


class MutatorHighVarBenchmark:
    nboot = 1000

    dataset_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-highvar-human-learning-data.json'

    def __init__(self):

        self.dataset = LearningDataset.from_url(dataset_url = self.dataset_url)


    @dataclass
    class SimulateExperimentResult:
        """
        The result of simulating an experiment on this battery of experiments.
        """
        subtasks: List[str]
        k: np.ndarray  # [trial, subtask]
        n: np.ndarray  # [trial, subtask]

    def _simulate_experiment(self, learner: hobj.learning_models.BinaryLearningModel, seed: int) -> SimulateExperimentResult:
        # Todo
        nreps = 500
        ntrials = 100
        warnings.warn("Returning dummy results.")
        RS = np.random.RandomState(0)

        return self.SimulateExperimentResult(
            subtasks=[],
            k=RS.randint(low=0, high=n, size=(64, 100)),
            n=np.ones((64, 100)) * n
        )

    @dataclass
    class EvaluateModelResult:

        # Benchmark results:
        msen: float
        msen_sigma: float
        msen_null_samples: List[float]

        # Point estimates that may be used for further downstream analysis:
        model_phat: np.ndarray  # [trial, subtask]
        model_varhat_phat: np.ndarray  # [trial, subtask]
        model_lapse_rate: float

        human_phat: np.ndarray  # [trial, subtask]
        human_varhat_phat: np.ndarray  # [trial, subtask]
        human_phat_bootstrap_resamples: np.ndarray  # [nboot, trial, subtask] - bootstrap resamples of these statistics over worker resamples.

        # Legacy version of msen, where variance correction from the human was not subtracted.
        _legacy_msen: float
        _legacy_msen_sigma: float

    def evaluate_model(self, learner: hobj.learning_models.BinaryLearningModel, seed: int = 0) -> EvaluateModelResult:
        """
        :param learner: LearningModel
        :param force_recompute: bool. If True, recompute the model behavior, even if it is already cached.
        :return:
        """

        # Simulate the behavior of this model
        model_behavior = self._simulate_experiment(learner=learner, seed=seed)
        k = model_behavior.k
        n = model_behavior.n

        # Fit a lapse rate which minimizes the MSE between the model and human data
        lapse_rate = lapse_rate_funcs.fit_optimal_lapse_rate(
            phat=model_behavior.k / model_behavior.n,
            p=self.ds_behavioral_statistics.perf.values,
            nway=2,
        )

        # Load human data
        target_point = self.ds_worker_table.sum('worker_id')
        ktarget = target_point.k.values
        ntarget = target_point.n.values
        phat_target = ktarget / ntarget
        varhat_phat_target = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=ktarget, nvec=ntarget)

        # Get estimates of performance
        phat_model = k / n
        varhat_phat_model = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)

        # Get MSEn point estimate on the lapse-rate corrected predictions
        phat_model_lapsed = phat_model * (1 - lapse_rate) + 0.5 * lapse_rate
        varhat_phat_model_lapsed = varhat_phat_model * (1 - lapse_rate) ** 2

        # Get elementwise errors:
        msen_elementwise = np.square(phat_target - phat_model_lapsed) - varhat_phat_model_lapsed - varhat_phat_target
        legacy_msen_elementwise = np.square(phat_target - phat_model) - varhat_phat_model_lapsed

        # Get point estimates:
        msen_point = msen_elementwise.mean()
        msen_sigma = ss.sem(msen_elementwise, ddof = 1)

        # Package return
        return self.EvaluateModelResult(
            msen=msen_point,
            msen_sigma = msen_sigma,
            model_phat=phat_model,
            model_varhat_phat=varhat_phat_model,
            model_lapse_rate=lapse_rate,
            human_phat=phat_target,
            human_varhat_phat=varhat_phat_target,
            msen_null_samples = msen_null_samples,
            human_phat_bootstrap_resamples=phat_target_bootstrap_resamples,
            _legacy_msen=legacy_msen_elementwise.mean(),
            _legacy_msen_sigma=ss.sem(legacy_msen_elementwise, ddof=1),
        )
