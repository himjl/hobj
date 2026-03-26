from typing import Tuple, Union

import numpy as np


def estimate_basic_bootstrap_CI(
    alpha: float,
    point_estimate: Union[np.ndarray, np.generic, float, int],  # [*]
    bootstrapped_point_estimates: np.ndarray,  # [n_bootstraps, *]
) -> Tuple[float, float]:
    """
    Estimates the basic confidence interval for a given point estimate(s) using the bootstrap method.
    :param alpha: Sets the width of the confidence interval to be 1 - alpha. Must be in the range (0, 1).
    :param point_estimate: The point estimate(s) for which the confidence interval is to be estimated.
    :param bootstrapped_point_estimates: Bootstrap resamples of the point estimate in question.
    :return: A tuple containing the lower and upper bounds of the confidence interval.
    """

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in the range (0, 1); got {alpha:.2f}")

    # Coerce point_estimate to a numpy array
    point_estimate = np.array(point_estimate)  # [*]

    # Ensure the dimensions are as expected
    if not len(point_estimate.shape) == len(bootstrapped_point_estimates.shape) - 1:
        raise ValueError(
            f"The dimensions of the point estimate and bootstrapped point estimates do not match: {point_estimate.shape} vs {bootstrapped_point_estimates.shape}"
        )

    # Calculate confidence interval using the basic bootstrap method
    low = np.quantile(bootstrapped_point_estimates, alpha / 2, axis=0)  # [*]
    high = np.quantile(bootstrapped_point_estimates, 1 - alpha / 2, axis=0)  # [*]

    empirical_parameter = bootstrapped_point_estimates.mean(0)  # [*]

    delta1 = empirical_parameter - low
    delta2 = high - empirical_parameter

    CI_low = point_estimate - delta2
    CI_high = point_estimate + delta1

    return float(CI_low), float(CI_high)
