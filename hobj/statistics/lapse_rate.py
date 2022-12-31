import xarray as xr
import numpy as np
from typing import Union, Tuple, List


def fit_optimal_lapse_rate(
        phat: Union[xr.DataArray, np.ndarray, float, int],
        p: Union[xr.DataArray, np.ndarray, float, int],
        nway: int,
        condition_dims: Union[float, int, tuple, Tuple[str], List[str]] = -1):
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

    numerator = -(2 * phat / nway - 2 * np.square(phat) + 2 * phat * p - 2 * p / nway).sum(condition_dims)
    denominator = (2 / (nway ** 2) - 4 * phat / nway + 2 * (phat ** 2)).sum(condition_dims)
    gamma_star = numerator / denominator
    gamma_star = np.clip(gamma_star, 0, 1)

    if isinstance(gamma_star, np.ndarray):
        if gamma_star.shape == ():
            gamma_star = float(gamma_star)
    return gamma_star


def _test():
    import matplotlib.pyplot as plt

    np.random.seed(0)
    gamma_true = np.random.rand() * 0.3
    phat = np.clip(np.random.rand(500) + gamma_true, 0, 1)
    p = phat - gamma_true

    # Numerical solution
    nway = 8
    gamma_seq = np.linspace(0.01, 1, 1000)
    gamma_star = fit_optimal_lapse_rate(phat, p, nway=nway)
    phat_gamma = phat[None, :] * (1 - gamma_seq[:, None]) + gamma_seq[:, None] * (1 / nway)  # [gamma, cond]
    mse = np.square(phat_gamma - p).mean(1)
    plt.plot(gamma_seq, mse)
    plt.axvline(gamma_star, color='black', ls='-', label='true')
    plt.axvline(gamma_seq[np.argmin(mse)], color='red', ls='-', label='numerical')
    plt.axvline(gamma_star, color='blue', ls=':', label='function')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _test()
