from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from hobj.benchmarks.binary_classification.simulation import (
    BinaryClassificationSubtaskResult,
)
from hobj.stats import binomial as binomial_funcs


@dataclass(frozen=True)
class _BootstrapSamples:
    """Bootstrap resamples for binary-classification benchmark statistics."""

    boot_k: np.ndarray
    boot_n: np.ndarray


def build_learning_curve_statistics(
    subtask_name_to_results: Dict[str, List[BinaryClassificationSubtaskResult]],
    nbootstrap_samples: int,
    bootstrap_by_worker: bool,
) -> xr.Dataset:
    """Build learning-curve summary statistics as a plain xarray dataset.

    Args:
        subtask_name_to_results: Mapping from subtask name to session results.
        nbootstrap_samples: Number of bootstrap replicates to generate.
        bootstrap_by_worker: Whether to resample workers instead of sessions.

    Returns:
        A dataset containing point estimates and bootstrap replicates.
    """

    subtask_names = sorted(subtask_name_to_results.keys())
    subtask_name_to_i = {name: i for i, name in enumerate(subtask_names)}
    nsubtasks = len(subtask_names)

    ntrials_observed = set()
    all_workers = set()
    nsessions = 0
    for results in subtask_name_to_results.values():
        nsessions += len(results)
        for result in results:
            ntrials_observed.add(len(result.perf_seq))
            all_workers.add(result.worker_id)

    if len(ntrials_observed) != 1:
        raise ValueError(
            f"Expected all subtasks to have the same number of trials, but got {ntrials_observed}"
        )

    ntrials = ntrials_observed.pop()
    all_workers = sorted(all_workers)
    worker_to_i = {worker: i for i, worker in enumerate(all_workers)}

    i_subtasks: List[int] = []
    i_workers: List[int] = []
    i_session = 0
    perf_mat = np.zeros((nsessions, ntrials), dtype=bool)
    for subtask_name, results in subtask_name_to_results.items():
        i_subtask = subtask_name_to_i[subtask_name]
        for result in results:
            i_worker = worker_to_i[result.worker_id]
            perf_mat[i_session] = result.perf_seq
            i_session += 1
            i_subtasks.append(i_subtask)
            i_workers.append(i_worker)

    k_mat = np.zeros(shape=(nsubtasks, ntrials), dtype=int)
    n_mat = np.zeros(shape=nsubtasks, dtype=int)
    for i_subtask, perf in zip(i_subtasks, perf_mat):
        k_mat[i_subtask] += perf
        n_mat[i_subtask] += 1

    phat, varhat_phat = _get_point_estimates(k=k_mat, n=n_mat[:, None])

    if bootstrap_by_worker:
        bootstrap_resamples = _get_bootstrap_resamples_by_worker(
            perf_mat=perf_mat,
            i_subtasks=i_subtasks,
            i_workers=i_workers,
            nbootstrap_samples=nbootstrap_samples,
        )
    else:
        bootstrap_resamples = _get_bootstrap_resamples_by_session(
            perf_mat=perf_mat,
            i_subtasks=i_subtasks,
            nbootstrap_samples=nbootstrap_samples,
        )

    boot_phat, boot_varhat_phat = _get_point_estimates(
        k=bootstrap_resamples.boot_k,
        n=bootstrap_resamples.boot_n[..., None],
    )

    return xr.Dataset(
        data_vars=dict(
            phat=(["subtask", "trial"], phat),
            varhat_phat=(["subtask", "trial"], varhat_phat),
            boot_phat=(["boot_iter", "subtask", "trial"], boot_phat),
            boot_varhat_phat=(["boot_iter", "subtask", "trial"], boot_varhat_phat),
        ),
        coords=dict(subtask=subtask_names),
    )


def _get_point_estimates(
    k: np.ndarray,
    n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate binomial means and variances from count matrices."""

    phat = k / n
    varhat_phat = binomial_funcs.estimate_variance_of_binomial_proportion(
        kvec=k, nvec=n
    )
    return phat, varhat_phat


def _get_bootstrap_resamples_by_worker(
    perf_mat: np.ndarray,
    i_subtasks: List[int],
    i_workers: List[int],
    nbootstrap_samples: int,
) -> _BootstrapSamples:
    """Bootstrap binary-classification sessions by worker."""

    if perf_mat.dtype != np.dtype(bool):
        raise ValueError(
            f"Expected perf_mat to have dtype bool, but got {perf_mat.dtype}"
        )

    ntrials = perf_mat.shape[1]
    nworkers = max(i_workers) + 1
    nsubtasks = max(i_subtasks) + 1

    if len(set(i_subtasks)) != nsubtasks:
        raise ValueError(
            f"Expected i_subtasks to be a contiguous range, but got {set(i_subtasks)}"
        )

    if len(set(i_workers)) != nworkers:
        raise ValueError(
            f"Expected i_workers to be a contiguous range, but got {set(i_workers)}"
        )

    perf_table = np.zeros(shape=(nworkers, nsubtasks, ntrials), dtype=bool)
    was_recorded_table = np.zeros(shape=(nworkers, nsubtasks), dtype=bool)
    for i_subtask, i_worker, perf in zip(i_subtasks, i_workers, perf_mat):
        perf_table[i_worker, i_subtask] = perf
        was_recorded_table[i_worker, i_subtask] = True

    gen = np.random.default_rng()
    boot_k = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials), dtype=int)
    boot_n = np.zeros(shape=(nbootstrap_samples, nsubtasks), dtype=int)
    for i_bootstrap in range(nbootstrap_samples):
        i_workers_boot = gen.integers(low=0, high=nworkers, size=nworkers)
        boot_k[i_bootstrap] = perf_table[i_workers_boot].sum(axis=0)
        boot_n[i_bootstrap] = was_recorded_table[i_workers_boot].sum(axis=0)

    return _BootstrapSamples(
        boot_k=boot_k,
        boot_n=boot_n,
    )


def _get_bootstrap_resamples_by_session(
    perf_mat: np.ndarray,
    i_subtasks: List[int],
    nbootstrap_samples: int,
) -> _BootstrapSamples:
    """Bootstrap binary-classification sessions within each subtask."""

    ntrials = perf_mat.shape[1]
    nsubtasks = max(i_subtasks) + 1

    if len(set(i_subtasks)) != nsubtasks:
        raise ValueError(
            f"Expected i_subtasks to be a contiguous range, but got {set(i_subtasks)}"
        )

    i_subtask_to_perf_data: Dict[int, np.ndarray] = {}
    for i_subtask, perf in zip(i_subtasks, perf_mat):
        if i_subtask not in i_subtask_to_perf_data:
            i_subtask_to_perf_data[i_subtask] = np.empty((0, ntrials), dtype=bool)
        i_subtask_to_perf_data[i_subtask] = np.vstack(
            [i_subtask_to_perf_data[i_subtask], perf]
        )

    gen = np.random.default_rng()
    boot_k = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials), dtype=int)
    boot_n = np.zeros(shape=(nbootstrap_samples, nsubtasks), dtype=int)
    for i_bootstrap in range(nbootstrap_samples):
        for i_subtask in range(nsubtasks):
            perf_data = i_subtask_to_perf_data[i_subtask]
            nsessions_for_subtask = len(perf_data)
            i_sessions = gen.integers(
                low=0,
                high=nsessions_for_subtask,
                size=nsessions_for_subtask,
            )
            boot_k[i_bootstrap, i_subtask] = perf_data[i_sessions].sum(axis=0)
            boot_n[i_bootstrap, i_subtask] = len(i_sessions)

    return _BootstrapSamples(
        boot_k=boot_k,
        boot_n=boot_n,
    )
