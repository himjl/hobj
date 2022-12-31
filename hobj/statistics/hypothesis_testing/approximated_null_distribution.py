import xarray as xr
from typing import Union, List


def estimate_null_distribution(
        statistical_functional,
        ds_null_samples: xr.Dataset,
        hat_mu: Union[xr.Dataset, xr.DataArray] = None,
        hat_sigma: Union[xr.Dataset, xr.DataArray] = None,
        resample_dim='boot_iter'
):
    null_stats = statistical_functional(ds_null_samples)

    if hat_mu is not None:
        null_stats = null_stats - null_stats.mean(resample_dim) + hat_mu
    if hat_sigma is not None:
        null_stats = (null_stats - null_stats.mean(resample_dim)) * hat_sigma / null_stats.std(resample_dim, ddof=1) + null_stats.mean(resample_dim)
    return null_stats
