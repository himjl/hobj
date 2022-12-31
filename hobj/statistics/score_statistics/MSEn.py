import xarray as xr
import numpy as np
from typing import Union, List



def estimate_MSEn(
        hat_theta_model: Union[xr.DataArray, np.ndarray],
        hat_var_hat_theta_model: Union[xr.DataArray, np.ndarray],
        hat_theta_target: Union[xr.DataArray, np.ndarray],
        condition_dims: tuple,
):
    """
    The expected value of the raw MSE is:

    (E[x] - E[y])^2 + var(x) + var(y)

    The MSEn performs a bias correction using the estimated variance of the model.

    :param hat_theta_model:
    :param hat_var_hat_theta_model:
    :param hat_theta_target:
    :param hat_var_hat_theta_target:
    :param condition_dims:
    :return:
    """
    hat_MSE_raw = np.square(hat_theta_model - hat_theta_target).mean(condition_dims)
    hat_MSE_n = hat_MSE_raw - hat_var_hat_theta_model.mean(condition_dims)
    return hat_MSE_n

