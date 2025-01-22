import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.stats as ss
import xarray as xr

import hobj.data.images.depr_imagesets as imagesets
import hobj.statistics.hypothesis_testing.approximated_null_distribution as approximated_null_distribution
import hobj.statistics.hypothesis_testing.bootstrapped_confidence_intervals as bootstrapped_confidence_intervals
import hobj.statistics.lapse_rate as lapse_rate_funcs
import hobj.statistics.score_statistics.MSEn as R2n_funcs
import hobj.statistics.variance_estimates.binomial as binomial_funcs
from hobj.data.behavior import human_data as human_data
from hobj.learning_models import learning_model as lm
from hobj.statistics.resamplers.resamplers import LearningCurveResampler, WorkerResampler
from hobj.utils import stats as stats


class MutatorHighVarBenchmark:

    def __init__(self):

        self.nboot = 1000
        self.tearly = slice(1, 6)
        self.tlate = slice(94, 100)
        self.dataset = human_data.MutatorHighVarDataset()
        self.resampler = LearningCurveResampler(
            ds=self.ds_worker_table,
            condition_dim='subtask',
            rep_dim='worker_id'
        )

        self.worker_resampler = WorkerResampler(
            ds_worker_table=self.ds_worker_table,
            condition_dim='subtask',
            worker_dim='worker_id'
        )

        self.imageset = imagesets.MutatorHighVarDeprImageset()

        # Todo: move/load this manifest elsewhere – e.g. to imageset
        subtask_manifest = Path(os.path.dirname(__file__)) / 'MutatorHighVarSubtasks.json'
        subtask_manifest = json.loads(subtask_manifest.read_text())
        self.subtask_manifest = subtask_manifest

    @dataclass
    class SimulateExperimentResult:
        """
        The result of simulating an experiment on this battery of experiments.
        """
        subtasks: List[str]
        k: np.ndarray  # [trial, subtask]
        n: np.ndarray  # [trial, subtask]

    def _simulate_experiment(self, learner: lm.BinaryLearningModel, seed: int) -> SimulateExperimentResult:
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
        mse_n: float
        subtask_vector_spearmanr: float
        perf_early: float
        perf_late: float

        subtasks: List[str]
        model_phat: np.ndarray  # [trial, subtask]
        model_varhat_phat: np.ndarray  # [trial, subtask]
        model_lapse_rate: float

        human_phat: np.ndarray  # [trial, subtask]
        human_varhat_phat: np.ndarray  # [trial, subtask]

    def evaluate_model(self, learner: lm.BinaryLearningModel, seed: int = 0) -> EvaluateModelResult:
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

        msen_elementwise = np.square(phat_target - phat_model_lapsed) - varhat_phat_model_lapsed
        MSEn_point = msen_elementwise.mean()

        # Get SpearmanR between the subtask difficulty patterns
        predicted_subtask_vector = phat_model.mean(1)  # [subtask]
        actual_subtask_vector = phat_target.mean(1)  # [subtask]

        SpearmanR_point = ss.spearmanr(predicted_subtask_vector, actual_subtask_vector).correlation

        # %% Get the (g)rand (l)earning (c)urve: the average over all subtasks, and the confidence intervals
        glc = phat_model.mean(0)  # [trial]
        glc_lapsed = glc * (1 - lapse_rate) + 0.5 * lapse_rate
        overall_perf = glc_lapsed.mean()  # Prediction of lapse-rate adjusted model

        # %% Get early and late performances
        perf_early = glc[self.tearly].mean()
        perf_late = glc[self.tlate].mean()

        return self.EvaluateModelResult(
            mse_n=MSEn_point,
            subtask_vector_spearmanr=SpearmanR_point,
            subtasks=subtasks,
            model_phat=phat_model,
            model_varhat_phat=varhat_phat_model,
            model_lapse_rate=lapse_rate,
            human_phat=phat_target,
            human_varhat_phat=varhat_phat_target
        )

    @property
    def ds_ceilings(self):
        """
        - The point estimate for the R2n noise ceiling, and associated CI.
        - The approximate null (where the model is equal to humans) distribution of R2n scores, given the number of model simulations.
        - The point estimates for the grand-human learning curve, and associated CI.

        - The expected range of SpearmanRs between the subtask difficulty vector between two repetitions of this experiment

        :return:
        """
        if not hasattr(self, '_ds_ceilings'):
            alpha = 0.05
            ds_ceilings = xr.Dataset()

            # %% Get point estimate of R2 ceiling, and confidence intervals
            point = self.ds_worker_table.sum('worker_id')
            ds_ceilings['MSEn'] = binomial_funcs.estimate_variance_of_binomial_proportion(
                kvec=point.k,
                nvec=point.n
            ).mean(['subtask', 'trial'])

            # Approximate the null distribution of R2n,
            # The null hypothesis is that null samples are drawn from the same distribution as the human data.

            def get_null_MSEn_functional(ds_data: xr.Dataset):
                pvec = ds_data.k / ds_data.n
                pvec_null = ds_data.knull / ds_data.nnull
                var_pvec_null = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=ds_data.knull, nvec=ds_data.nnull)

                MSEn = R2n_funcs.estimate_MSEn(
                    hat_theta_model=pvec_null,
                    hat_var_hat_theta_model=var_pvec_null,
                    hat_theta_target=pvec,
                    condition_dims=('subtask', 'trial')
                )
                return MSEn

            nsessions = self.resampler.nsessions.copy(deep=True)
            n_null_sessions = nsessions - nsessions + 500  # Because 500 simulations per subtask were performed, for the models.

            ds_boot = self.resampler.get_ds_boot(self.nboot, seed=0)
            ds_null = self.resampler.get_ds_resamples(nresamples=self.nboot, nsessions=n_null_sessions, seed=1)
            ds_null = ds_null.rename({'k': 'knull', 'n': 'nnull'})

            ds_data = xr.merge([ds_boot, ds_null])

            ds_ceilings['MSEn_null'] = approximated_null_distribution.estimate_null_distribution(
                statistical_functional=get_null_MSEn_functional,
                ds_null_samples=ds_data,
                hat_mu=ds_ceilings['MSEn'],
                resample_dim='boot_iter'
            )

            # %% Approximate the distribution of SpearmanRs between the subtask difficulty vectors from two independent repetitions of this experiment
            ds_boot1 = self.worker_resampler.get_ds_boot(nboot=self.nboot, seed=1)
            ds_boot2 = self.worker_resampler.get_ds_boot(nboot=self.nboot, seed=2)
            vec1 = ds_boot1.k.sum('trial') / ds_boot1.n.sum('trial')
            vec2 = ds_boot2.k.sum('trial') / ds_boot2.n.sum('trial')
            ds_ceilings['SpearmanR_repeat'] = stats.calc_correlation(obs_dim='subtask', y_pred=vec1, y_actual=vec2, spearman=True)

            # %% Approximate the distribution of overall performances over repetitions of this experiment
            ds_ceilings['overall_perf'] = (self.ds_human.k / self.ds_human.n).mean(['subtask', 'trial'])
            ds_ceilings['overall_perf_repeat'] = (ds_boot1.k / ds_boot1.n).mean(['subtask', 'trial'])

            # %% Get point estimate of the grand-human learning curve, and confidence intervals
            ds_ceilings['glc'] = (self.ds_human.k / self.ds_human.n).mean('subtask')
            ds_ceilings['glc_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds_data: (ds_data.k / ds_data.n).mean('subtask'),
                parameter_functional=None,
            )

            # %% Get point estimates of early and late performances
            ds_ceilings['early_perf'] = ds_ceilings.glc.isel(trial=self.tearly).mean('trial')
            ds_ceilings['early_perf_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds_data: (ds_data.k / ds_data.n).mean('subtask').isel(trial=self.tearly).mean('trial'),
            )
            ds_ceilings['late_perf'] = ds_ceilings.glc.isel(trial=self.tlate).mean('trial')
            ds_ceilings['late_perf_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds_data: (ds_data.k / ds_data.n).mean('subtask').isel(trial=self.tlate).mean('trial'),
            )

            # %% Save
            self._ds_ceilings = ds_ceilings

        return self._ds_ceilings

    @property
    def ds_worker_table(self):
        if not hasattr(self, '_ds_worker_table'):
            self._ds_worker_table = self.dataset.load_ds_worker_table()
        return self._ds_worker_table

    @property
    def ds_human(self):
        if not hasattr(self, '_ds_human'):
            self._ds_human = self.ds_worker_table.sum('worker_id')
        return self._ds_human

    @property
    def ds_behavioral_statistics(self):
        """

        perf: (subtask, trial)
        perf_CI: (subtask, trial)
        perf_std: (subtask, trial)

        glc: (trial)
        glc_CI: (trial)
        glc_std: (subtask, trial)

        subtask_vector: (subtask)
        subtask_vector_CI: (subtask)
        subtask_vector_std: (subtask)

        :return:
        """
        if not hasattr(self, '_ds_behavioral_statistics'):
            alpha = 0.05
            ds_stats = xr.Dataset()

            # Performance point statistics
            ds_stats['perf'] = self.ds_human['k'] / self.ds_human['n']
            ds_stats['perf_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds: ds['k'] / ds['n'],
                parameter_functional=None,
            )
            ds_stats['perf_std'] = (self.ds_boot['k'] / self.ds_boot['n']).std('boot_iter')

            # Overall learning curve
            ds_stats['glc'] = ds_stats['perf'].mean('subtask')
            ds_stats['glc_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds: (ds['k'] / ds['n']).mean('subtask'),
                parameter_functional=None,
            )
            ds_stats['glc_std'] = (self.ds_boot['k'] / self.ds_boot['n']).mean('subtask').std('boot_iter')

            # Subtask difficulty vector
            ds_stats['subtask_vector'] = ds_stats['perf'].mean('trial')
            ds_stats['subtask_vector_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_human,
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds: (ds['k'] / ds['n']).mean('trial'),
                parameter_functional=None,
            )
            ds_stats['subtask_vector_std'] = (self.ds_boot['k'] / self.ds_boot['n']).mean('trial').std('boot_iter')
            self._ds_behavioral_statistics = ds_stats

        return self._ds_behavioral_statistics

    @property
    def ds_boot(self):

        if not hasattr(self, '_ds_boot'):
            self._ds_boot = self.resampler.get_ds_boot(nboot=self.nboot, seed=0)

        return self._ds_boot
