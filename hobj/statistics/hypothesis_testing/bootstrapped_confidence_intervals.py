import xarray as xr

def estimate_basic_confidence_interval(
        alpha: float,
        ds_data: xr.Dataset,
        ds_boot: xr.Dataset,
        statistic_functional,
        parameter_functional = None,

):
    assert 'boot_iter' in ds_boot.dims
    for dim in ds_data.dims:
        assert dim in ds_boot.dims, dim

    assert alpha <= 0.1
    assert alpha > 0

    point_estimate = statistic_functional(ds_data)
    ds_boot_stats = statistic_functional(ds_boot)

    if parameter_functional is not None:
        empirical_parameter = parameter_functional(ds_data)
    else:
        empirical_parameter = ds_boot_stats.mean('boot_iter')

    low = ds_boot_stats.quantile(alpha/2,'boot_iter', )
    high = ds_boot_stats.quantile( 1 - alpha / 2,'boot_iter',)

    delta1 = empirical_parameter - low
    delta2 = high - empirical_parameter
    CI = xr.concat([point_estimate - delta2, point_estimate + delta1], 'CI')
    return CI
