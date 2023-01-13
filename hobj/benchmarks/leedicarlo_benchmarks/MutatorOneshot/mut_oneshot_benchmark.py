import collections

import xarray as xr
import numpy as np
from tqdm import trange

import hobj.learning_models.learning_model as lm
import hobj.images.imagesets as imagesets
import hobj.benchmarks.leedicarlo_benchmarks.MutatorOneshot.mut_oneshot_experiment as mutator


import hobj.benchmarks.benchmarks as benchmarks
import hobj.behavioral_data.human_data as human_data
from hobj.statistics.resamplers.resamplers import Resampler
import hobj.statistics.score_statistics.MSEn as R2n_funcs
import hobj.statistics.variance_estimates.binomial as binomial_funcs
import hobj.statistics.hypothesis_testing.bootstrapped_confidence_intervals as bootstrapped_confidence_intervals
import hobj.statistics.hypothesis_testing.approximated_null_distribution as approximated_null_distribution


class MutatorOneshotBenchmark(benchmarks.Benchmark):
    environment_name_dim = 'session'

    def __init__(self):
        super().__init__()
        self.nboot = 1000
        self.dataset = human_data.MutatorOneShotGeneralizationDataset()
        self.ds_raw = self.dataset.load_ds_data()

        # Remove sessions with less than 100% catch performance
        self.ds_meta = imagesets.MutatorOneshotImageset().ds_meta
        self.url_to_trans_id = {url: trans for (url, trans) in zip(self.ds_meta.image_url.values, self.ds_meta.transformation_id.values)}
        self.url_to_obj = {url: obj for (url, obj) in zip(self.ds_meta.image_url.values, self.ds_meta.obj.values)}
        self.experiment = mutator.MutatorOneShotExperiment()

        self.transformation_ids = ['backgrounds | 0.1',
                                   'backgrounds | 0.215443',
                                   'backgrounds | 0.464159',
                                   'backgrounds | 1.0',
                                   'blur | 0.007812',
                                   'blur | 0.015625',
                                   'blur | 0.03125',
                                   'blur | 0.0625',
                                   'contrast | -0.4',
                                   'contrast | -0.8',
                                   'contrast | 0.4',
                                   'contrast | 0.8',
                                   'delpixels | 0.25',
                                   'delpixels | 0.5',
                                   'delpixels | 0.75',
                                   'delpixels | 0.95',
                                   'inplanerotation | 135.0',
                                   'inplanerotation | 180.0',
                                   'inplanerotation | 45.0',
                                   'inplanerotation | 90.0',
                                   'inplanetranslation | 0.125',
                                   'inplanetranslation | 0.25',
                                   'inplanetranslation | 0.5',
                                   'inplanetranslation | 0.75',
                                   'noise | 0.125',
                                   'noise | 0.25',
                                   'noise | 0.375',
                                   'noise | 0.5',
                                   'outplanerotation | 135.0',
                                   'outplanerotation | 180.0',
                                   'outplanerotation | 45.0',
                                   'outplanerotation | 90.0',
                                   'scale | 0.125',
                                   'scale | 0.25',
                                   'scale | 0.5',
                                   'scale | 1.5']
        self.subtasks = [
            'MutatorB2000_2292,MutatorB2000_2444',
            'MutatorB2000_138,MutatorB2000_2344',
            'MutatorB2000_1251,MutatorB2000_953',
            'MutatorB2000_3043,MutatorB2000_694',
            'MutatorB2000_3496,MutatorB2000_496',
            'MutatorB2000_1219,MutatorB2000_296',
            'MutatorB2000_1825,MutatorB2000_2757',
            'MutatorB2000_3077,MutatorB2000_4703',
            'MutatorB2000_270,MutatorB2000_3615',
            'MutatorB2000_3066,MutatorB2000_3585',
            'MutatorB2000_2139,MutatorB2000_746',
            'MutatorB2000_116,MutatorB2000_2365',
            'MutatorB2000_2130,MutatorB2000_4628',
            'MutatorB2000_462,MutatorB2000_926',
            'MutatorB2000_2304,MutatorB2000_3733',
            'MutatorB2000_1363,MutatorB2000_3278',
            'MutatorB2000_4049,MutatorB2000_663',
            'MutatorB2000_2722,MutatorB2000_3527',
            'MutatorB2000_2832,MutatorB2000_801',
            'MutatorB2000_1258,MutatorB2000_3123',
            'MutatorB2000_1865,MutatorB2000_613',
            'MutatorB2000_1164,MutatorB2000_2106',
            'MutatorB2000_1229,MutatorB2000_1280',
            'MutatorB2000_1767,MutatorB2000_2122',
            'MutatorB2000_2198,MutatorB2000_701',
            'MutatorB2000_3636,MutatorB2000_4305',
            'MutatorB2000_3035,MutatorB2000_46',
            'MutatorB2000_3601,MutatorB2000_4792',
            'MutatorB2000_2092,MutatorB2000_288',
            'MutatorB2000_1424,MutatorB2000_2314',
            'MutatorB2000_3308,MutatorB2000_3525',
            'MutatorB2000_2909,MutatorB2000_4256'
        ]

    def evaluate_model(self, learner: lm.LearningModel, force_recompute: bool = False):
        """
        :param learner: LearningModel
        :param force_recompute: bool. If True, recompute the results even if they are already cached.
        :return:
        """
        ds_model_behavior = self.experiment.run(learner=learner, seed=0, force_recompute=force_recompute)
        ds_model_behavior = ds_model_behavior.sel(transformation_id=self.ds_table.transformation_id)
        ds_model_behavior = ds_model_behavior.sum('subtask')

        # Compute scores
        ds_score = self.score_predictions(
            k=ds_model_behavior.k,
            n=ds_model_behavior.n,
        )

        return ds_score

    def get_lapse_rate_corrected_performance(
            self,
            k,
            n,
            kcatch,
            ncatch,
    ):
        """

        Returns an estimate of the performance that would be expected if the system had a lapse rate of 0.
        The values kcatch and ncatch are assumed to be drawn from a Bernoulli distribution with parameter p = (1 - lapse rate).
        This function also returns an estimate of the variance associated with that estimate, using bootstrapping.

        We assume
        kcatch ~ Binomial(ncatch, (1 - lapse_rate) * 1 + lapse_rate * 0.5)
        k ~ Binomial(n, (1 - lapse_rate) * p + (lapse_rate) * 0.5)

        Denote
        pcatch = (1 - lapse_rate) * 1 + lapse_rate * 0.5)

        Then,
        lapse_rate = 2 - 2 * pcatch
        praw =  (1 - lapse_rate) * p + (lapse_rate) * 0.5 (the empirically observed probability of a correct response)

        We have the relation:
        p = (praw - lapse_rate / 2) / (1 - lapse_rate)

        Here, we estimate p using the plug-in estimator for p.

        :param k:
        :param n:
        :param kcatch:
        :param ncatch:
        :return:
        """

        assert np.all(kcatch <= ncatch)
        assert np.all(kcatch >= 0)

        hat_praw = k / n
        hat_pcatch = kcatch / ncatch
        hat_lapse_rate = 2 - 2 * hat_pcatch
        hat_p_adjusted = (hat_praw - hat_lapse_rate / 2) / (1 - hat_lapse_rate)

        return hat_p_adjusted

    def score_predictions(
            self,
            k: xr.DataArray,
            n: xr.DataArray,
    ):

        """
        k: (transformation_id)
        n: (transformation_id)
        kcatch: int
        ncatch: int
        returns ds_scores
        """

        # Compute R^2 score
        assert isinstance(k, xr.DataArray)
        assert isinstance(n, xr.DataArray)
        assert set(k.dims) == {'transformation_id'}, (k.dims)
        assert set(k.dims) == set(n.dims), (k.dims, n.dims)

        hat_prob = k / n
        hat_var_hat_prob = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)

        # %% Get MSEn point estimate
        MSEn_point = R2n_funcs.estimate_MSEn(
            hat_theta_model=hat_prob,
            hat_var_hat_theta_model=hat_var_hat_prob,
            hat_theta_target=self.ds_behavioral_statistics['perf'],
            condition_dims=('transformation_id',),
        )

        # %% Assemble scores
        ds_scores = xr.Dataset(
            data_vars=dict(
                MSEn = MSEn_point,
                perf=hat_prob,
                perf_by_type=hat_prob.groupby('transformation').mean(),
            ),
        )

        return ds_scores

    @property
    def ds_behavioral_statistics(self):
        if not hasattr(self, '_ds_behavioral_statistics'):
            p = self.ds_table.k.sum('session') / self.ds_table.n.sum('session')
            pvec_boot = self.get_lapse_rate_corrected_performance(
                k=self.ds_boot.k,
                n=self.ds_boot.n,
                kcatch=self.ds_boot.k_catch.sum(['catch_trial']),
                ncatch=self.ds_boot.n_catch.sum(['catch_trial']),
            )

            var_pvec = pvec_boot.var('boot_iter', ddof=1)

            ds_behavioral_statistics = xr.Dataset(
                {
                    'perf_raw': p,
                    'perf': self.get_lapse_rate_corrected_performance(
                        k=self.ds_table.k.sum('session'),
                        n=self.ds_table.n.sum('session'),
                        kcatch=int(self.ds_table.k_catch.sum(['session', 'catch_trial'])),
                        ncatch=int(self.ds_table.n_catch.sum(['session', 'catch_trial'])),
                    ),
                    'hat_var_perf': var_pvec,
                }
            )

            self._ds_behavioral_statistics = ds_behavioral_statistics
        return self._ds_behavioral_statistics

    def _process_raw_experiment(self, da_perf: xr.DataArray):
        """

        :param da_perf:  (session, trial)
        :return: ds_oneshot
            perf: (transformation_id, stimulus_category)
            catch_perf: ()
        """

        """
        ktraining: (session, trial)
        ntraining: (session, trial)
        k: (session, condition)
        n: (session, condition)
        kprobe: (session, condition)
        nprobe: (session, condition)
        :return:
        """

        train_trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        catch_trials = [14, 19]  # trials used to estimate the lapse rate (each of these trials presented one of the two support images)
        test_trials = [10, 11, 12, 13, 15, 16, 17, 18, ]  # trials containing a test trial (defined as a randomly sampled, transformed version of one of the two support images)
        assert np.all(np.mod(da_perf, 1) == 0)
        da_perf = da_perf.astype(int)
        k_catch = da_perf.isel(trial=catch_trials)
        n_catch = k_catch - k_catch + 1

        k_training = da_perf.isel(trial=train_trials)
        n_training = k_training - k_training + 1

        da_train = da_perf.isel(trial=train_trials).transpose('session', 'trial')
        da_test = da_perf.isel(trial=test_trials).transpose('session', 'trial')

        nsessions = len(da_perf.session)
        ntransformations = len(self.transformation_ids)
        k = np.zeros((nsessions, ntransformations))
        n = np.zeros((nsessions, ntransformations))

        trans_id_to_i = collections.defaultdict(lambda: np.nan)
        for i, t in enumerate(self.transformation_ids):
            trans_id_to_i[t] = i

        worker_ids = []
        for i_session in range(da_test.session.values.size):
            worker_ids.append(str(da_train.isel(session=i_session).worker_id.values))
            support_ids = sorted(set(da_train.stimulus_url.values[i_session]))
            support_objects = sorted([self.url_to_obj[u] for u in support_ids])
            assert len(support_objects) == 2, support_objects

            url_seq = da_test.stimulus_url.values[i_session]
            perf_seq = da_test.values[i_session]
            trans_id_seq = [self.url_to_trans_id[u] for u in url_seq]

            for i_trial in range(len(trans_id_seq)):
                trans = trans_id_seq[i_trial]
                i_trans = trans_id_to_i[trans]
                if np.isnan(i_trans):
                    continue

                perf = int(perf_seq[i_trial])

                k[i_session, i_trans] += perf
                n[i_session, i_trans] += 1

        ds = xr.Dataset(
            {
                'k': (['session', 'transformation_id'], k),
                'n': (['session', 'transformation_id'], n),
                'k_catch': (['session', 'catch_trial'], k_catch.transpose('session', 'trial').values),
                'n_catch': (['session', 'catch_trial'], n_catch.transpose('session', 'trial').values),
                'k_train': (['session', 'train_trial'], k_training.transpose('session', 'trial').values),
                'n_train': (['session', 'train_trial'], n_training.transpose('session', 'trial').values),

            },
            coords={
                'transformation_id': self.transformation_ids,
                'transformation': (['transformation_id'], [t.split(" | ")[0] for t in self.transformation_ids]),
                'transformation_level': (['transformation_id'], [float(t.split(" | ")[1]) for t in self.transformation_ids]),
                'session': da_perf.session.values,
                'catch_trial': catch_trials,
                'train_trial': train_trials,
                'worker_id': (['session'], worker_ids)
            }
        )

        return ds


    @property
    def ds_table(self):

        """
        ktraining: (session, trial)
        ntraining: (session, trial)
        k: (session, condition)
        n: (session, condition)
        kprobe: (session, condition)
        nprobe: (session, condition)
        :return:
        """
        if not hasattr(self, '_ds_table'):
            self._ds_table = self._process_raw_experiment(da_perf = self.ds_raw.perf)
        return self._ds_table


    @property
    def ds_ceilings(self):
        """
        - The point estimate for the R2n noise ceiling, and associated CI.
        - The approximate null (where the model is equal to humans) distribution of R2n scores, given the number of model simulations.
        - The point estimates for each transformation, associated CIs, and standard errors.
        - The point estimates for each transformation type, and its associate CI, and standard errors.

        :return:
        """
        if not hasattr(self, '_ds_ceilings'):
            alpha = 0.05
            ds_ceilings = xr.Dataset()

            # %% Get point estimate of MSE floor
            pvec_boot = self.get_lapse_rate_corrected_performance(
                k=self.ds_boot.k,
                n=self.ds_boot.n,
                kcatch=self.ds_boot.k_catch.sum(['catch_trial']),
                ncatch=self.ds_boot.n_catch.sum(['catch_trial']),
            )

            var_pvec = pvec_boot.var('boot_iter', ddof=1)

            ds_ceilings['MSEn'] = var_pvec.mean('transformation_id')

            # %% Approximate the null distribution of MSEn, using the null hypothesis that the model is equal to the (lapse-corrected) human performance

            var_pvec_null = self.get_lapse_rate_corrected_performance(
                k=self.ds_null.k,
                n=self.ds_null.n,
                kcatch=self.ds_null.k_catch.sum(['catch_trial']),
                ncatch=self.ds_null.n_catch.sum(['catch_trial']),
            ).var('boot_iter', ddof=1)

            def get_null_MSE_functional(ds_data: xr.Dataset):
                pvec = self.get_lapse_rate_corrected_performance(
                    k=ds_data.k,
                    n=ds_data.n,
                    kcatch=(ds_data.k_catch.sum('catch_trial')),
                    ncatch=(ds_data.n_catch.sum('catch_trial')),
                )

                pvec_null = self.get_lapse_rate_corrected_performance(
                    k=ds_data.knull,
                    n=ds_data.nnull,
                    kcatch=(ds_data.k_catch_null.sum('catch_trial')),
                    ncatch=(ds_data.n_catch_null.sum('catch_trial')),
                )

                MSEn = R2n_funcs.estimate_MSEn(
                    hat_theta_model=pvec_null,
                    hat_var_hat_theta_model=var_pvec_null,
                    hat_theta_target = pvec,
                    condition_dims=('transformation_id',),
                )

                return  MSEn

            ds_boot = self.ds_boot
            ds_null = self.ds_null
            ds_null = ds_null.rename({
                'k': 'knull',
                'n': 'nnull',
                'k_catch': 'k_catch_null',
                'n_catch': 'n_catch_null',
                'k_train': 'k_train_null',
                'n_train': 'n_train_null',
            })
            ds_data = xr.merge([ds_boot, ds_null])

            ds_ceilings['MSEn_null'] = approximated_null_distribution.estimate_null_distribution(
                statistical_functional=get_null_MSE_functional,
                ds_null_samples=ds_data,
                hat_mu=ds_ceilings['MSEn'],
                resample_dim='boot_iter'
            )

            # %% Get point estimate of per-transformation performances, confidence intervals, and standard errors (lapse-rate adjusted)
            perf_lapsed_functional = lambda ds_data: self.get_lapse_rate_corrected_performance(
                k=ds_data.k,
                n=ds_data.n,
                kcatch=ds_data.k_catch.sum('catch_trial'),
                ncatch=ds_data.n_catch.sum('catch_trial'),
            )

            ds_data = self.ds_table.sum('session')
            perf_lapse = perf_lapsed_functional(ds_data)

            perf_lapsed_boot = perf_lapsed_functional(self.ds_boot)
            ds_ceilings['perf'] = perf_lapse
            ds_ceilings['perf_std'] = perf_lapsed_boot.std('boot_iter', ddof=1)  # Get via bootstrapping
            ds_ceilings['perf_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=ds_data,
                ds_boot=self.ds_boot,
                statistic_functional=perf_lapsed_functional,
                parameter_functional=None,
            )

            # %% Get point estimate of per-transformation performances, confidence intervals, and standard errors
            point = self.ds_table.sum('session')
            ds_ceilings['perf_raw'] = point.k / point.n
            ds_ceilings['perf_raw_std'] = (self.ds_boot.k / self.ds_boot.n).std('boot_iter', ddof=1)
            ds_ceilings['perf_raw_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_table.sum('session')[['k', 'n']],
                ds_boot=self.ds_boot,
                statistic_functional=lambda ds_data: ds_data.k / ds_data.n,
                parameter_functional=None,
            )

            # %% Get point estimate of per-transformation-type performances, confidence intervals, and standard errors

            def perf_by_type_functional(ds_data):
                pvec = self.get_lapse_rate_corrected_performance(
                    k=ds_data.k,
                    n=ds_data.n,
                    kcatch=ds_data.k_catch.sum('catch_trial'),
                    ncatch=ds_data.n_catch.sum('catch_trial'),
                )
                perf_by_type = pvec.groupby("transformation").mean()
                perf_by_type = perf_by_type.rename({'transformation': 'transformation_type'})
                return perf_by_type

            perf_by_type = perf_by_type_functional(self.ds_table.sum('session'))
            perf_by_type_boot = perf_by_type_functional(self.ds_boot)

            ds_ceilings['perf_by_type'] = perf_by_type
            ds_ceilings['perf_by_type_std'] = perf_by_type_boot.std('boot_iter', ddof=1)
            ds_ceilings['perf_by_type_CI'] = bootstrapped_confidence_intervals.estimate_basic_confidence_interval(
                alpha=alpha,
                ds_data=self.ds_table.sum('session'),
                ds_boot=self.ds_boot,
                statistic_functional=perf_by_type_functional,
                parameter_functional=None,
            )

            self._ds_ceilings = ds_ceilings

        return self._ds_ceilings

    @property
    def worker_resampler(self):
        if not hasattr(self, '_worker_resampler'):
            self._worker_resampler = ExperimentResampler(
                ds_session=self.ds_table,
            )
        return self._worker_resampler

    @property
    def ds_boot(self):
        if not hasattr(self, '_ds_boot'):
            dlist = []
            RS = np.random.RandomState(seed=0)
            for i_boot in trange(self.nboot, desc='resampling experiment'):
                ds_resampled = self.worker_resampler.get_ds_resampled(RS=RS).sum('session')
                dlist.append(ds_resampled)

            self._ds_boot = xr.concat(dlist, dim='boot_iter').assign_coords(
                transformation=(['transformation_id'], self.ds_table.transformation.values),
                transformation_level=(['transformation_id'], self.ds_table.transformation_level.values),
            )

        return self._ds_boot

    @property
    def ds_null(self):
        if not hasattr(self, '_ds_null'):
            RS = np.random.RandomState(seed=1)
            dlist = []
            for i_boot in trange(self.nboot, desc='resampling experiment'):
                ds_resampled = self.worker_resampler.get_ds_resampled(RS=RS, nsessions=16000).sum('session')
                dlist.append(ds_resampled)

            self._ds_null = xr.concat(dlist, dim='boot_iter').assign_coords(
                transformation=(['transformation_id'], self.ds_table.transformation.values),
                transformation_level=(['transformation_id'], self.ds_table.transformation_level),
            )

        return self._ds_null


class ExperimentResampler(Resampler):
    """
    Resamples the experiment, simulating the data-generating process:
        1) First, re-samples a subject randomly
         2) Then for that re-sampled subject, resamples a session associated with that subject.

    The probability of re-sampling a particular session is then given by:

    Prob(session) = Prob(worker) * Prob(session | worker)
    Prob(worker) = 1 /nworkers
    Prob(session | worker) = 1 / nsessions[i_worker]
    Prob(session) = 1 / (nworkers * nsessions[i_worker])

    """

    def __init__(
            self,
            ds_session: xr.Dataset,
    ):

        super().__init__()
        mandatory_vars = ['k_catch', 'n_catch', 'k_train', 'n_train', 'k', 'n']
        for var in mandatory_vars:
            assert var in ds_session.data_vars, ds_session.data_vars

        all_dims = ds_session.dims
        assert 'session' in all_dims
        assert 'worker_id' in ds_session.coords

        prob_session = []
        nworkers = len(set(ds_session['worker_id'].values))
        worker_to_nsessions = collections.defaultdict(lambda: 0)
        for worker_id in ds_session['worker_id'].values:
            worker_to_nsessions[worker_id] += 1

        for worker_id in ds_session['worker_id'].values:
            prob_session.append(1 / (nworkers * worker_to_nsessions[worker_id]))

        assert np.isclose(np.sum(prob_session), 1)

        self.ds_session = ds_session.transpose('session', 'transformation_id', 'catch_trial', 'train_trial')
        self.transformation_ids = self.ds_session['transformation_id'].values
        self.catch_trial = self.ds_session['catch_trial'].values
        self.train_trial = self.ds_session['train_trial'].values

        self.prob_session = np.array(prob_session)
        self.nsessions_empirical = len(self.prob_session)
        self.worker_id_seq = self.ds_session['worker_id'].values

    def get_ds_resampled(self, RS: np.random.RandomState, nsessions=None, ):
        """
        Returns a single bootstrap re-sample
        :param seed:
        :return:
        """

        if nsessions is None:
            # Take a bootstrap resample
            nsessions = len(self.ds_session['session'])

        i_sessions = RS.choice(self.nsessions_empirical, size=nsessions, p=self.prob_session, replace=True)

        kbootstrapped_data = self.ds_session['k'].values[i_sessions]
        nbootstrapped_data = self.ds_session['n'].values[i_sessions]
        kcatch_resampled_data = self.ds_session['k_catch'].values[i_sessions]
        ncatch_resampled_data = self.ds_session['n_catch'].values[i_sessions]
        ktrain_resampled_data = self.ds_session['k_train'].values[i_sessions]
        ntrain_resampled_data = self.ds_session['n_train'].values[i_sessions]
        worker_ids = self.worker_id_seq[i_sessions]

        ds_resampled = xr.Dataset(
            data_vars={
                'k': (['session', 'transformation_id'], kbootstrapped_data),
                'n': (['session', 'transformation_id'], nbootstrapped_data),
                'k_catch': (['session', 'catch_trial'], kcatch_resampled_data),
                'n_catch': (['session', 'catch_trial'], ncatch_resampled_data),
                'k_train': (['session', 'train_trial'], ktrain_resampled_data),
                'n_train': (['session', 'train_trial'], ntrain_resampled_data),
            },
            coords={
                'transformation_id': self.transformation_ids,
                'catch_trial': self.catch_trial,
                'train_trial': self.train_trial,
                'session': np.arange(nsessions),
                'worker_id': (['session'], worker_ids),
            }
        )

        return ds_resampled
