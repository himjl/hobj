import warnings
from typing import List, Tuple

import numpy as np
import xarray as xr

from hobj.benchmarks.generalization.simulator import GeneralizationSessionResult
from hobj.stats import binomial as binomial_funcs


def build_generalization_statistics(
    results: List[GeneralizationSessionResult],
    perform_lapse_rate_correction: bool,
    n_bootstrap_iterations: int,
    bootstrap_by_worker: bool,
) -> xr.Dataset:
    """Build one-shot generalization statistics as a plain xarray dataset.

    Args:
        results: Session-level generalization results.
        perform_lapse_rate_correction: Whether to correct point estimates using
            catch-trial lapse rates.
        n_bootstrap_iterations: Number of bootstrap replicates to generate.
        bootstrap_by_worker: Whether to resample workers instead of sessions.

    Returns:
        A dataset containing point estimates and bootstrap replicates.
    """

    all_transformations = sorted(
        {
            transformation
            for result in results
            for transformation in result.transformation_to_kn.keys()
        }
    )
    transformation_to_i = {
        transformation: i for i, transformation in enumerate(all_transformations)
    }

    all_workers = sorted({result.worker_id for result in results})
    nworkers = len(all_workers)
    worker_to_i = {worker: i for i, worker in enumerate(all_workers)}

    nsessions = len(results)
    ntransformations = len(all_transformations)

    if bootstrap_by_worker:
        if nworkers == 1:
            raise ValueError(
                f"Only one worker {all_workers}, cannot perform valid bootstrap by worker."
            )
        if nworkers < 20:
            warnings.warn(
                (
                    f"Only {nworkers} unique workers, which is less than 20. "
                    "Bootstrapping by worker may not be reliable."
                ),
                stacklevel=2,
            )

        kmat = np.zeros(shape=(nworkers, ntransformations))
        nmat = np.zeros(shape=(nworkers, ntransformations))
        kcatch = np.zeros(shape=nworkers)
        ncatch = np.zeros(shape=nworkers)
    else:
        kmat = np.zeros(shape=(nsessions, ntransformations))
        nmat = np.zeros(shape=(nsessions, ntransformations))
        kcatch = np.zeros(shape=nsessions)
        ncatch = np.zeros(shape=nsessions)

    for i_session, result in enumerate(results):
        i_row = worker_to_i[result.worker_id] if bootstrap_by_worker else i_session
        kcatch[i_row] = result.kcatch
        ncatch[i_row] = result.ncatch

        for transformation, (k, n) in result.transformation_to_kn.items():
            i_transformation = transformation_to_i[transformation]
            kmat[i_row, i_transformation] += k
            nmat[i_row, i_transformation] += n

    phat, varhat_phat = _get_point_estimates(
        k=kmat.sum(0),
        n=nmat.sum(0),
        kcatch=kcatch.sum(),
        ncatch=ncatch.sum(),
        perform_lapse_rate_correction=perform_lapse_rate_correction,
    )

    gen = np.random.default_rng()
    boot_phat = np.zeros(shape=(n_bootstrap_iterations, ntransformations))
    boot_varhat_phat = np.zeros(shape=(n_bootstrap_iterations, ntransformations))
    for i_boot_iter in range(n_bootstrap_iterations):
        if bootstrap_by_worker:
            i_boot = gen.integers(low=0, high=nworkers, size=nworkers)
        else:
            i_boot = gen.integers(low=0, high=nsessions, size=nsessions)

        kboot = kmat[i_boot].sum(0)
        nboot = nmat[i_boot].sum(0)
        kcatch_boot = kcatch[i_boot].sum()
        ncatch_boot = ncatch[i_boot].sum()

        phat_cur, varhat_phat_cur = _get_point_estimates(
            k=kboot,
            n=nboot,
            kcatch=kcatch_boot,
            ncatch=ncatch_boot,
            perform_lapse_rate_correction=perform_lapse_rate_correction,
        )

        boot_phat[i_boot_iter] = phat_cur
        boot_varhat_phat[i_boot_iter] = varhat_phat_cur

    return xr.Dataset(
        data_vars=dict(
            phat=(["transformation"], phat),
            varhat_phat=(["transformation"], varhat_phat),
            boot_phat=(["boot_iter", "transformation"], boot_phat),
            boot_varhat_phat=(["boot_iter", "transformation"], boot_varhat_phat),
        ),
        coords=dict(transformation=all_transformations),
    )


def _get_point_estimates(
    k: np.ndarray,
    n: np.ndarray,
    kcatch: np.ndarray,
    ncatch: np.ndarray,
    perform_lapse_rate_correction: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate generalization means and variances from binomial counts."""

    phat = k / n
    varhat_phat = binomial_funcs.estimate_variance_of_binomial_proportion(
        kvec=k, nvec=n
    )

    if perform_lapse_rate_correction:
        hat_lapse_rate = 2 - 2 * (kcatch / ncatch)
        phat = (phat - hat_lapse_rate / 2) / (1 - hat_lapse_rate)
        varhat_phat = varhat_phat / (1 - hat_lapse_rate) ** 2

    phat = np.clip(phat, 0, 1)
    return phat, varhat_phat
