import numpy as np
from typing import Union
from hobj.statistics.variance_estimates import binomial as binomial_funcs
from dataclasses import dataclass


@dataclass
class CalculateLearningCurveResult:
    phat: np.ndarray # [trial]
    varhat_phat: np.ndarray # [trial]

    boot_phat: np.ndarray # [boot, trial]
    boot_varhat_phat: np.ndarray # [boot, trial]


def calculate_learning_curve(
        perf_matrix: np.ndarray,
        bootstrap_seed: Union[None, int],
        nbootstrap_samples: int,
        repetition_axis: int,
) -> CalculateLearningCurveResult:

    """
    Calculates the learning curve statistics for a given set of performance data for a single subtask.
    Returns the point estimates and bootstrapped estimates of the mean and variance of the binomial proportion.
    :param perf_matrix: a boolean np.ndarray of shape (nrepetitions, ntrials) where True indicates a correct response.
    :param bootstrap_seed:
    :param nbootstrap_samples:
    :return:
    """

    nrepetitions = perf_matrix.shape[repetition_axis]
    ntrials = perf_matrix.shape[1]

    # Calculate statistics:
    k = perf_matrix.sum(axis=repetition_axis)
    n = nrepetitions
    phat = k / n
    varhat_phat = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=k, nvec=n)

    # Perform bootstrapping of the model
    boot_phat = np.zeros((nbootstrap_samples, ntrials))
    boot_varhat_phat = np.zeros((nbootstrap_samples, ntrials))

    # Bootstrap resample model simulations
    gen = np.random.default_rng(seed=bootstrap_seed)

    for i_boot_iter in range(nbootstrap_samples):
        i_boot_sessions = gen.integers(low =0, high = nrepetitions, size = nrepetitions)

        kboot = perf_matrix[i_boot_sessions].sum(repetition_axis)
        boot_phat[i_boot_iter] = kboot / n
        boot_varhat_phat[i_boot_iter] = binomial_funcs.estimate_variance_of_binomial_proportion(kvec=kboot, nvec=n)

    return CalculateLearningCurveResult(
        phat=phat,
        varhat_phat=varhat_phat,
        boot_phat=boot_phat,
        boot_varhat_phat=boot_varhat_phat,
    )

