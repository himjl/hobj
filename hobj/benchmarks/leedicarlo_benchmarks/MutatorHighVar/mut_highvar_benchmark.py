from typing import Union

import numpy as np
import xarray as xr

from hobj.behavioral_data import human_data as human_data
from hobj.benchmarks.benchmarks import Benchmark
from hobj.learning_models import learning_model as lm
from hobj.utils import stats as stats
import hobj.benchmarks.leedicarlo_benchmarks.MutatorHighVar.mut_highvar_experiment as mutator

import hobj.statistics.lapse_rate as lapse_rate_funcs
from hobj.statistics.resamplers.resamplers import LearningCurveResampler, WorkerResampler
import hobj.statistics.score_statistics.MSEn as R2n_funcs
import hobj.statistics.variance_estimates.binomial as binomial_funcs
import hobj.statistics.hypothesis_testing.bootstrapped_confidence_intervals as bootstrapped_confidence_intervals
import hobj.statistics.hypothesis_testing.approximated_null_distribution as approximated_null_distribution


class MutatorHighVarBenchmark(Benchmark):

    def __init__(self):
        super().__init__()
        self.nboot = 1000
        self.tearly = slice(1, 6)
        self.tlate = slice(94, 100)
        self.dataset = human_data.MutatorHighVarDataset()
        self.experiment = mutator.MutatorHighVarExperiment()
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

    def evaluate_model(self, learner: lm.LearningModel, force_recompute: bool = False):
        """
        :param learner: LearningModel
        :param force_recompute: bool. If True, recompute the model behavior, even if it is already cached.
        :return:
        """

        # Get behavior of this model
        ds_behavior = self.experiment.run(learner=learner, seed=0, force_recompute=force_recompute)

        k = ds_behavior.k.isel(stimulus_category=1) + (ds_behavior.n.isel(stimulus_category=0) - ds_behavior.k.isel(stimulus_category=0))
        n = ds_behavior.n.isel(stimulus_category=1) + ds_behavior.n.isel(stimulus_category=0)

        # Fit optimal lapse rate
        lapse_rate = lapse_rate_funcs.fit_optimal_lapse_rate(
            phat=k / n,
            p=(self.ds_behavioral_statistics.perf),
            nway=2,
            condition_dims=('subtask', 'trial')
        )

        # Compute scores
        target_point = self.ds_worker_table.sum('worker_id')
        ds_scores = self.score_predictions(
            k=k,
            n=n,
            ktarget=target_point.k,
            ntarget=target_point.n,
            lapse_rate=lapse_rate,
        ).assign_coords(
            lapse_rate=lapse_rate
        )

        return ds_scores

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
                kvec = point.k,
                nvec = point.n
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

    def score_predictions(
            self,
            k: xr.DataArray,
            n: xr.DataArray,
            ktarget: xr.DataArray,
            ntarget: xr.DataArray,
            lapse_rate: Union[float, int, xr.DataArray] = 0.,
    ):

        """
        k: (subtask, trial). The number of correct classifications in that condition.
        n: (subtask, trial). The number of trials for each condition.
        ktarget: (subtask, trial). The number of correct classifications in that condition, for the system aiming to be modeled.
        ntarget: (subtask, trial). The number of trials for each condition, for the system aiming to be modeled
        lapse_rate: a scalar. Adjusts the predictions of the model by simulating a random guess rate. Can only drive the behavior of the model to randomness.

        returns ds_scores, which contains:

            Point estimates, 95% CIs, and standard errors of:
                R2n: () the noise-corrected R^2 score
                glc: (trial) (g)rand (l)earning (c)urve over all subtasks
                overall_perf_raw: () the average performance over all subtasks and trials (before lapse rate correction)
                overall_perf: () the average performance over all subtasks and trials  (after lapse rate correction)
                SpearmanR: () between subtask difficulty vectors
        """

        assert np.all(lapse_rate >= 0) and np.all(lapse_rate <= 1), lapse_rate

        def check_kn(k, n):
            assert isinstance(k, xr.DataArray)
            assert isinstance(n, xr.DataArray)
            assert np.all(k <= n)
            assert np.all(n > 0)
            assert 'worker_id' not in k.dims
            assert 'subtask' in k.dims
            assert 'trial' in k.dims

            # assert set(k.dims) == {'subtask', 'trial'}, (k.dims)

            assert set(k.dims) == set(n.dims), (k.dims, n.dims)

        check_kn(k, n)
        check_kn(ktarget, ntarget)

        alpha = 0.05
        # Get estimates of performance
        hat_prob = k / n
        hat_var_hat_prob = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)

        # %% Get MSEn point estimate on the lapse-rate corrected predictions
        hat_prob_lapse_rate = hat_prob * (1 - lapse_rate) + 0.5 * (lapse_rate)
        hat_var_hat_prob_lapse_rate = hat_var_hat_prob * (1 - lapse_rate) ** 2

        hat_prob_target = ktarget / ntarget

        MSEn_point = R2n_funcs.estimate_MSEn(
            hat_theta_model=hat_prob_lapse_rate,
            hat_var_hat_theta_model=hat_var_hat_prob_lapse_rate,
            hat_theta_target=hat_prob_target,
            condition_dims=('subtask', 'trial'),
        )


        # %% Get SpearmanR between the subtask difficulty patterns
        predicted_subtask_vector = hat_prob.mean("trial")
        actual_subtask_vector = hat_prob_target.mean('trial')

        SpearmanR_point = stats.calc_correlation(
            obs_dim='subtask',
            y_pred=predicted_subtask_vector,
            y_actual=actual_subtask_vector,
            spearman=True
        )

        # %% Get the (g)rand (l)earning (c)urve: the average over all subtasks, and the confidence intervals
        glc = hat_prob.mean('subtask')
        glc_lapsed = glc * (1 - lapse_rate) + 0.5 * (lapse_rate)
        overall_perf = glc_lapsed.mean('trial')  # Prediction of lapse-rate adjusted model

        # %% Get early and late performances
        perf_early = glc.isel(trial=self.tearly).mean('trial')
        perf_late = glc.isel(trial=self.tlate).mean('trial')

        # %% Assemble scores
        ds_scores = xr.Dataset(
            data_vars=dict(
                overall_perf=overall_perf,
                overall_perf_raw=glc.mean('trial'),
                SpearmanR=SpearmanR_point,
                MSEn=MSEn_point,
                glc=glc,
                early_perf=perf_early,
                late_perf=perf_late,
            ),
        )

        return ds_scores

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

