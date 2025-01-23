from typing import Union

import numpy as np
import xarray as xr


def fit_optimal_lapse_rate(
        phat: Union[np.ndarray, np.generic, float, int],
        p: Union[np.ndarray, np.generic, float, int],
        nway: int,
) -> float:
    """
    Fits a lapse rate (gamma, which is the probability of a uniform random guess) that minimizes the empirical MSE loss between the phat and p:

        z = phat * (1 - gamma) + 1/nway * (gamma)
        Loss = ((z - p)**2).sum()

    :param phat:
    :param p:
    :param nway:
    :return:
    """

    assert nway >= 2
    assert np.all(phat <= 1)
    assert np.all(phat >= 0)
    assert np.all(p <= 1)
    assert np.all(p >= 0)

    if not isinstance(phat, (np.ndarray, xr.DataArray)):
        phat = np.array(phat)

    if not isinstance(p, (np.ndarray, xr.DataArray)):
        p = np.array(p)

    assert p.shape == phat.shape

    numerator = -(2 * phat / nway - 2 * np.square(phat) + 2 * phat * p - 2 * p / nway).sum()
    denominator = (2 / (nway ** 2) - 4 * phat / nway + 2 * (phat ** 2)).sum()
    gamma_star = numerator / denominator
    gamma_star = np.clip(gamma_star, 0, 1)

    if isinstance(gamma_star, np.ndarray):
        if gamma_star.shape == ():
            gamma_star = float(gamma_star)
    return gamma_star

