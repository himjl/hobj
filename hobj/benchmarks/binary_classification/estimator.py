from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtaskResult
from hobj.stats import binomial as binomial_funcs


# %%
class LearningCurveStatistics(xr.Dataset):
    __slots__ = ()

    def __init__(
            self,
            subtask_name_to_results: Dict[str, List[BinaryClassificationSubtaskResult]],
            nbootstrap_samples: int,
            bootstrap_by_worker: bool,  # versus by session
    ):
        # Get canonical order of subtask names
        subtask_names = sorted(subtask_name_to_results.keys())
        subtask_name_to_i = {name: i for i, name in enumerate(subtask_names)}
        nsubtasks = len(subtask_names)

        # Ensure all subtasks have the same number of trials
        ntrials_observed = set()
        all_workers = set()
        nsessions = 0
        for subtask, results in subtask_name_to_results.items():
            nsessions += len(results)
            for r in results:
                ntrials_observed.add(len(r.perf_seq))
                all_workers.add(r.worker_id)

        if not len(ntrials_observed) == 1:
            raise ValueError(f"Expected all subtasks to have the same number of trials, but got {ntrials_observed}")
        ntrials = ntrials_observed.pop()
        all_workers = sorted(all_workers)
        worker_to_i = {worker: i for i, worker in enumerate(all_workers)}

        # Reshape data into a [session, trial] matrix and associated [session] meta vectors
        i_subtasks = []
        i_workers = []
        i_session = 0
        perf_mat = np.zeros((nsessions, ntrials), dtype=np.bool)
        for subtask, results in subtask_name_to_results.items():
            i_subtask = subtask_name_to_i[subtask]
            for result in results:
                i_worker = worker_to_i[result.worker_id]
                perf_mat[i_session] = result.perf_seq
                i_session += 1
                i_subtasks.append(i_subtask)
                i_workers.append(i_worker)

        # Collate data into [subtask, trial] matrices
        k_mat = np.zeros(shape=(nsubtasks, ntrials), dtype=int)
        n_mat = np.zeros(shape=nsubtasks, dtype=int)

        for i_subtask, perf in zip(i_subtasks, perf_mat):
            k_mat[i_subtask] += perf
            n_mat[i_subtask] += 1

        # Get point estimate of the [subtask, trial] of learning curve first and second moments
        phat, varhat_phat = self.get_point_estimates(k=k_mat, n=n_mat[:, None])

        # Perform bootstrap resampling of data
        if bootstrap_by_worker:
            bootstrap_resamples = self._get_bootstrap_resamples_by_worker(
                perf_mat=perf_mat,
                i_subtasks = i_subtasks,
                i_workers = i_workers,
                nbootstrap_samples=nbootstrap_samples,
            )
        else:
            bootstrap_resamples = self._get_bootstrap_resamples_by_session(
                perf_mat=perf_mat,
                i_subtasks = i_subtasks,
                nbootstrap_samples=nbootstrap_samples,
            )

        # Calculate bootstrapped point estimates
        boot_phat, boot_varhat_phat = self.get_point_estimates(
            k=bootstrap_resamples.boot_k,  # [boot_iter, subtask, trial]
            n=bootstrap_resamples.boot_n[..., None],  # [boot_iter, subtask]
        )

        super().__init__(
            data_vars=dict(
                phat=(['subtask', 'trial'], phat),
                varhat_phat=(['subtask', 'trial'], varhat_phat),
                boot_phat=(['boot_iter', 'subtask', 'trial'], boot_phat),
                boot_varhat_phat=(['boot_iter', 'subtask', 'trial'], boot_varhat_phat),
            ),
            coords=dict(
                subtask=subtask_names,
            )
        )

    @staticmethod
    def get_point_estimates(
            k: np.ndarray,
            n: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        phat = k / n
        varhat_phat = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)
        return phat, varhat_phat

    @dataclass
    class BootstrapSamples:
        boot_k: np.ndarray  # [boot_iter, subtask, trial]
        boot_n: np.ndarray  # [boot_iter, subtask]

    @staticmethod
    def _get_bootstrap_resamples_by_worker(
            perf_mat: np.ndarray, # [session, trial]
            i_subtasks: List[int], # [session]
            i_workers: List[int], # [session]
            nbootstrap_samples: int,
    ) -> BootstrapSamples:

        if not perf_mat.dtype == np.bool:
            raise ValueError(f"Expected perf_mat to have dtype bool, but got {perf_mat.dtype}")

        ntrials = perf_mat.shape[1]
        nworkers = max(i_workers) + 1
        nsubtasks = max(i_subtasks) + 1

        if not len(set(i_subtasks)) == nsubtasks:
            raise ValueError(f"Expected i_subtasks to be a contiguous range, but got {set(i_subtasks)}")

        if not len(set(i_workers)) == nworkers:
            raise ValueError(f"Expected i_workers to be a contiguous range, but got {set(i_workers)}")

        # Reshape data
        perf_table = np.zeros(shape=(nworkers, nsubtasks, ntrials), dtype=np.bool)
        was_recorded_table = np.zeros(shape=(nworkers, nsubtasks), dtype=np.bool)

        for i_subtask, i_worker, perf in zip(i_subtasks, i_workers, perf_mat):
            perf_table[i_worker, i_subtask] = perf
            was_recorded_table[i_worker, i_subtask] = True

        # Perform bootstrapping
        gen = np.random.default_rng()
        boot_k = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials), dtype=int)
        boot_n = np.zeros(shape=(nbootstrap_samples, nsubtasks), dtype=int)
        for i_bootstrap in range(nbootstrap_samples):
            # Bootstrap resample workers
            i_workers_boot = gen.integers(low=0, high=nworkers, size=nworkers)
            boot_k[i_bootstrap] = perf_table[i_workers_boot].sum(axis=0)
            boot_n[i_bootstrap] = was_recorded_table[i_workers_boot].sum(axis=0)

        return LearningCurveStatistics.BootstrapSamples(
            boot_k=boot_k,
            boot_n=boot_n,
        )

    @staticmethod
    def _get_bootstrap_resamples_by_session(
            perf_mat: np.ndarray,  # [session, trial]
            i_subtasks: List[int], # [session]
            nbootstrap_samples: int,
    ) -> BootstrapSamples:
        # Bootstrap by session, within each subtask

        ntrials = perf_mat.shape[1]
        nsubtasks = max(i_subtasks) + 1

        if not len(set(i_subtasks)) == nsubtasks:
            raise ValueError(f"Expected i_subtasks to be a contiguous range, but got {set(i_subtasks)}")

        # Reshape data to {i_subtask: [session, trial]}
        i_subtask_to_perf_data = {}
        for i_subtask, perf in zip(i_subtasks, perf_mat):
            if i_subtask not in i_subtask_to_perf_data:
                i_subtask_to_perf_data[i_subtask] = []
            i_subtask_to_perf_data[i_subtask].append(perf)

        # Cast to np arrays
        for i_subtask in i_subtask_to_perf_data:
            i_subtask_to_perf_data[i_subtask] = np.array(i_subtask_to_perf_data[i_subtask], dtype = np.bool)

        # Perform bootstrapping
        gen = np.random.default_rng()
        boot_k = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials), dtype=int)
        boot_n = np.zeros(shape=(nbootstrap_samples, nsubtasks), dtype=int)
        for i_bootstrap in range(nbootstrap_samples):
            for i_subtask in range(nsubtasks):
                nsessions_for_subtask = len(i_subtask_to_perf_data[i_subtask])
                i_sessions = gen.integers(low=0, high=nsessions_for_subtask, size=nsessions_for_subtask)
                boot_k[i_bootstrap, i_subtask] = i_subtask_to_perf_data[i_subtask][i_sessions].sum(axis=0)
                boot_n[i_bootstrap, i_subtask] = len(i_sessions)

        return LearningCurveStatistics.BootstrapSamples(
            boot_k=boot_k,
            boot_n=boot_n,
        )
