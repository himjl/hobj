import numpy as np
import xarray as xr
from tqdm import trange


class Resampler(object):
    def __init__(self):
        return


class WorkerResampler(Resampler):
    def __init__(self, ds_worker_table: xr.Dataset, condition_dim='subtask', worker_dim='worker_id'):
        """
        Bootstrap resamples workers.
        ds:
        :param k, n: (worker_id, subtask, trial)
        """
        super().__init__()
        assert 'k' in ds_worker_table.data_vars
        assert 'n' in ds_worker_table.data_vars
        assert 'trial' in ds_worker_table.dims
        assert worker_dim in ds_worker_table.dims, ds_worker_table.dims
        assert condition_dim in ds_worker_table.dims, ds_worker_table.dims
        assert np.all(ds_worker_table.n <= 1)
        assert np.all(ds_worker_table.k <= ds_worker_table.n), ds_worker_table
        assert np.all(ds_worker_table.k >= 0)

        self.ds_worker_table = ds_worker_table.transpose('worker_id', 'trial', 'subtask')
        self.worker_dim = worker_dim
        self.condition_dim = condition_dim

        return

    def get_ds_boot(self, nboot: int, seed: int):
        RS = np.random.RandomState(seed=seed)
        kbootstrapped = np.zeros((nboot, len(self.ds_worker_table[self.condition_dim]), len(self.ds_worker_table.trial),))
        nbootstrapped = np.zeros((nboot, len(self.ds_worker_table[self.condition_dim]), len(self.ds_worker_table.trial),))

        kdat = self.ds_worker_table.k.transpose('worker_id', 'subtask', 'trial').values
        ndat = self.ds_worker_table.n.transpose('worker_id', 'subtask', 'trial').values
        for i_resample in trange(nboot, desc='bootstrap resampling workers'):
            i_workers = RS.choice(len(self.ds_worker_table[self.worker_dim]), size=len(self.ds_worker_table[self.worker_dim]), replace=True).astype(int)
            kbootstrapped[i_resample] = kdat[i_workers].sum(0)
            nbootstrapped[i_resample] = ndat[i_workers].sum(0)

        ds_boot = xr.Dataset(
            data_vars={
                'k': (('boot_iter', self.condition_dim, 'trial',), kbootstrapped),
                'n': (('boot_iter', self.condition_dim, 'trial',), nbootstrapped),
            },
            coords={
                'boot_iter': np.arange(nboot),
                'trial': self.ds_worker_table.trial,
                self.condition_dim: self.ds_worker_table[self.condition_dim],
            }
        )
        return ds_boot


class LearningCurveResampler(Resampler):

    def __init__(self, ds: xr.Dataset, condition_dim='subtask', rep_dim='worker_id'):
        """
        For each condition, bootstraps sequences of Bernoulli trials.
        ds:
        :param k, n: ({condition_dim}, trial, {rep_dim})
        """
        super().__init__()
        assert 'k' in ds.data_vars
        assert 'n' in ds.data_vars
        assert 'trial' in ds.dims
        assert rep_dim in ds.dims, ds.dims
        assert condition_dim in ds.dims, ds.dims
        assert np.all(ds.n <= 1)
        assert np.all(ds.k <= ds.n), ds
        assert np.all(ds.k >= 0)

        self.ds = ds
        self.rep_dim = rep_dim

        condition_to_kn_dat = {}
        ntrials = len(ds.trial)
        for subtask, ds_subtask in ds.groupby(condition_dim):
            n = ds_subtask.n
            nsessions_with_data = ((n > 0).sum('trial') > 0).sum(rep_dim)
            valid_sessions = (n > 0).sum('trial') == ntrials
            assert valid_sessions.sum() == nsessions_with_data
            ds_valid = ds_subtask.sel({rep_dim: valid_sessions})
            ds_valid = ds_valid.transpose(rep_dim, 'trial')
            condition_to_kn_dat[subtask] = ds_valid['k'].values, ds_valid['n'].values  # [session, trial]

        self.condition_dim = condition_dim
        self.condition_to_kn_dat = condition_to_kn_dat
        self.conditions = sorted(condition_to_kn_dat.keys())

        # Number of sessions for each condition
        self.nsessions = xr.DataArray([self.condition_to_kn_dat[condition][0].shape[0] for condition in self.conditions],
                                      dims=self.condition_dim,
                                      coords={self.condition_dim: self.conditions}
                                      )

        return

    def get_ds_resamples(self, nresamples: int, nsessions: xr.DataArray, seed: int):
        """
        Simulate taking nsessions samples for each condition.
        :param nsessions:
        :return:
        """

        assert nresamples >= 1
        assert nsessions.dims == (self.condition_dim,)
        nsessions = nsessions.sel({self.condition_dim: self.conditions})
        nsessions = np.array(nsessions.values)
        RS = np.random.RandomState(seed=seed)
        kresamp = np.zeros((nresamples, len(self.ds.trial), len(self.ds[self.condition_dim])))
        nresamp = np.zeros((nresamples, len(self.ds.trial), len(self.ds[self.condition_dim])))

        for i_resample in trange(nresamples, desc='resampling learning curves'):

            for i_condition, condition in enumerate(self.conditions):
                kdat, ndat = self.condition_to_kn_dat[condition]  # [session, trial]
                nsessions_cur = nsessions[i_condition]
                navailable_lcs = kdat.shape[0]

                i_sessions = RS.choice(navailable_lcs, size=nsessions_cur, replace=True)
                kboot_samp = kdat[i_sessions].sum(0)
                nboot_samp = ndat[i_sessions].sum(0)

                kresamp[i_resample, :, i_condition] = kboot_samp
                nresamp[i_resample, :, i_condition] = nboot_samp

        ds_resamp = xr.Dataset(
            data_vars={
                'k': (('boot_iter', 'trial', self.condition_dim), kresamp),
                'n': (('boot_iter', 'trial', self.condition_dim), nresamp),
            },
            coords={
                'boot_iter': np.arange(nresamples),
                'trial': self.ds.trial,
                self.condition_dim: self.conditions,
            }
        )

        return ds_resamp

    def get_ds_resamples_path2(self, nresamples: int, nsessions: xr.DataArray, seed: int):
        """
        Simulate taking nsessions samples for each condition.
        :param nsessions:
        :return:
        """

        assert nresamples >= 1
        assert nsessions.dims == (self.condition_dim,)
        nsessions = nsessions.sel({self.condition_dim: self.conditions})
        nsessions = np.array(nsessions.values)
        RS = np.random.RandomState(seed=seed)
        kresamp = np.zeros((nresamples, len(self.ds.trial), len(self.ds[self.condition_dim])))
        nresamp = np.zeros((nresamples, len(self.ds.trial), len(self.ds[self.condition_dim])))

        for i_resample in trange(nresamples, desc='resampling learning curves'):

            for i_condition, condition in enumerate(self.conditions):
                kdat, ndat = self.condition_to_kn_dat[condition]  # [session, trial]
                nsessions_cur = nsessions[i_condition]
                navailable_lcs = kdat.shape[0]

                ntimes_resampled = RS.multinomial(n=nsessions_cur, pvals=1 / navailable_lcs * np.ones(navailable_lcs), )  # [navailable_lcs]
                kboot_samp = (kdat * ntimes_resampled[:, None]).sum(0)
                nboot_samp = (ndat * ntimes_resampled[:, None]).sum(0)

                kresamp[i_resample, :, i_condition] = kboot_samp
                nresamp[i_resample, :, i_condition] = nboot_samp

        ds_resamp = xr.Dataset(
            data_vars={
                'k': (('boot_iter', 'trial', self.condition_dim), kresamp),
                'n': (('boot_iter', 'trial', self.condition_dim), nresamp),
            },
            coords={
                'boot_iter': np.arange(nresamples),
                'trial': self.ds.trial,
                self.condition_dim: self.conditions,
            }
        )

        return ds_resamp

    def get_ds_boot(self, nboot: int, seed: int):
        ds_boot = self.get_ds_resamples_path2(nresamples=nboot, nsessions=self.nsessions, seed=seed)
        return ds_boot
