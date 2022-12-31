from scipy.stats import beta
import numpy as np
import xarray as xr


def estimate_variance_of_binomial_proportion(kvec, nvec):

    """
    Returns the *unbiased* estimate of the variance associated with the proportion estimator, phat = k/n:

    phat = kvec / nvec
    \hat var(phat) = (phat * (1 - phat)) / (nvec - 1)
    :param kvec:
    :param nvec:
    :return:
    """
    phat = kvec / nvec
    return (phat * (1 - phat)) / (nvec - 1)


def get_variance_of_binomial_proportion(pvec, nvec):

    """
    Returns the ground-truth variance associated with the proportion estimator associated with this distribution, phat = k/n:

    kvec ~ Binomial(nvec, pvec)
    phat = kvec / nvec
    \hat var(phat) = (phat * (1 - phat)) / (nvec - 1)
    :param kvec:
    :param nvec:
    :return:
    """

    return (pvec * (1 - pvec)) / (nvec)


def get_CI_of_binomial_proportion(kvec, nvec, alpha:float):

        """
        Returns the exact (Clopper-Pearson) confidence interval for the Binomial proportion estimator, phat:

        kvec ~ Binomial(nvec, pvec)
        phat = kvec / nvec

        These confidence intervals may be conservative (wide), due to the discrete nature of the Binomial distribution: they may have higher coverage than alpha.
        :param kvec:
        :param nvec:
        :return:
        """

        assert np.all(kvec <= nvec)
        assert np.all(kvec >= 0)
        assert np.all(nvec >= 2), np.min(nvec)

        template = kvec - kvec
        alpha_lower = (template + 1) * alpha / 2
        alpha_upper = (template + 1) * (1 - alpha / 2)

        p_lower, p_upper = beta.ppf([alpha_lower, alpha_upper], [kvec, kvec + 1], [nvec - kvec + 1, nvec - kvec])

        p_lower = (template) + p_lower
        p_upper = (template) + p_upper

        if isinstance(template, xr.DataArray):
            CI = xr.concat([p_lower, p_upper], dim='CI')
        else:
            CI = p_lower, p_upper
        return CI

