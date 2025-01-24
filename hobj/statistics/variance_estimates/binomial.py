def estimate_variance_of_binomial_proportion(kvec, nvec):

    """
    Returns the *unbiased* estimate of the variance associated with the proportion estimator, phat = k/n:

    phat = kvec / nvec
    \hat var(phat) = (phat * (1 - phat)) / (nvec - 1)
    :param kvec:
    :param nvec:
    :return:
    """
    # todo: add checking on argument
    phat = kvec / nvec
    return (phat * (1 - phat)) / (nvec - 1)
