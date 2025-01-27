from typing import List

import xarray as xr
import numpy as np
from hobj.benchmarks.generalization.simulator import GeneralizationSessionResult

from hobj.stats import binomial as binomial_funcs
import warnings


class GeneralizationStatistics(xr.Dataset):
    __slots__ = ()

    def __init__(
            self,
            results: List[GeneralizationSessionResult],
            perform_lapse_rate_correction: bool,
            n_bootstrap_iterations: int,
            bootstrap_by_worker: bool, # versus by session
    ):

        # Get all transformations
        all_transformations = set()
        for result in results:
            all_transformations.update(result.transformation_to_kn.keys())

        all_transformations = sorted(all_transformations)
        transformation_to_i = {transformation: i for i, transformation in enumerate(all_transformations)}

        # Get all workers
        all_workers = sorted({result.worker_id for result in results})
        print(len(all_workers))
        nworkers = len(all_workers)
        worker_to_i = {worker: i for i, worker in enumerate(all_workers)}

        # Package data into [session, transformation] kn matrices
        nsessions = len(results)
        ntransformations = len(all_transformations)

        if bootstrap_by_worker:
            if nworkers == 1:
                raise ValueError(f"Only one worker {all_workers}, cannot perform valid bootstrap by worker.")
            elif nworkers < 20:
                warnings.warn(f"Only {nworkers} unique workers, which is less than 20. Bootstrapping by worker may not be reliable.")

            kmat = np.zeros(shape=(nworkers, ntransformations))
            nmat = np.zeros(shape=(nworkers, ntransformations))
            kcatch = np.zeros(shape=nworkers)
            ncatch = np.zeros(shape=nworkers)
        else:
            # Will be bootstrapping by session
            kmat = np.zeros(shape=(nsessions, ntransformations))
            nmat = np.zeros(shape=(nsessions, ntransformations))
            kcatch = np.zeros(shape=nsessions)
            ncatch = np.zeros(shape=nsessions)

        for i_session, result in enumerate(results):
            i = worker_to_i[result.worker_id] if bootstrap_by_worker else i_session
            kcatch[i] = result.kcatch
            ncatch[i] = result.ncatch

            for transformation, (k, n) in result.transformation_to_kn.items():
                i_transformation = transformation_to_i[transformation]
                kmat[i, i_transformation] += k
                nmat[i, i_transformation] += n

        # Calculate statistics
        print(kmat.sum(0))
        print(nmat.sum(0))
        phat, varhat_phat = self._get_point_estimates(
            k=kmat.sum(0),
            n=nmat.sum(0),
            kcatch=kcatch.sum(),
            ncatch=ncatch.sum(),
            perform_lapse_rate_correction=perform_lapse_rate_correction,
        )

        # Compute bootstrap replicates by resampling sessions or workers
        gen = np.random.default_rng()
        boot_phat = np.zeros(shape=(n_bootstrap_iterations, ntransformations))
        boot_varhat_phat = np.zeros(shape=(n_bootstrap_iterations, ntransformations))

        for i_boot_iter in range(n_bootstrap_iterations):

            if bootstrap_by_worker:
                # Resample by worker
                i_boot = gen.integers(low=0, high=nworkers, size=nworkers)
            else:
                # Resample by session
                i_boot = gen.integers(low=0, high=nsessions, size=nsessions)
            kboot = kmat[i_boot].sum(0)
            nboot = nmat[i_boot].sum(0)
            kcatch_boot = kcatch[i_boot].sum()
            ncatch_boot = ncatch[i_boot].sum()

            # Compute bootstrapped point estimates
            phat_cur, varhat_phat_cur = self._get_point_estimates(
                k=kboot,
                n=nboot,
                kcatch=kcatch_boot,
                ncatch=ncatch_boot,
                perform_lapse_rate_correction=perform_lapse_rate_correction,
            )

            # Save
            boot_phat[i_boot_iter] = phat_cur
            boot_varhat_phat[i_boot_iter] = varhat_phat_cur

        super().__init__(
            data_vars=dict(
                phat=(['transformation'], phat),
                varhat_phat=(['transformation'], varhat_phat),
                boot_phat=(['boot_iter', 'transformation'], boot_phat),
                boot_varhat_phat=(['boot_iter', 'transformation'], boot_varhat_phat),
            ),
            coords=dict(
                transformation=all_transformations
            )
        )

    @staticmethod
    def _get_point_estimates(
            k: np.ndarray,  # [transformation]
            n: np.ndarray,  # [transformation]
            kcatch: np.ndarray,  # ()
            ncatch: np.ndarray,  # ()
            perform_lapse_rate_correction: bool,
    ):

        phat = k / n
        varhat_phat = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)

        if perform_lapse_rate_correction:
            # Return an estimate of the performance that would be expected if the system had a lapse rate of 0.
            # The lapse rate is estimated using the catch trials.
            hat_lapse_rate = 2 - 2 * (kcatch / ncatch)

            # Estimate the performance that would be expected if the system had a lapse rate of 0, using the plug-in estimator:
            phat = (phat - hat_lapse_rate / 2) / (1 - hat_lapse_rate)

            # Estimate the variance of the corrected performance by treating the lapse rate as a constant:
            varhat_phat = varhat_phat / (1 - hat_lapse_rate) ** 2

        phat = np.clip(phat, 0, 1)

        return phat, varhat_phat
