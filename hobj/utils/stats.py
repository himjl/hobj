import xarray as xr
from typing import Union, List, Tuple
import numpy as np


def calc_correlation(
        obs_dim:Union[str, List[str]],
        y_pred: xr.DataArray,
        y_actual: Union[xr.DataArray, type(None)] = None,
        spearman=False,
):
    """
    This function returns a DataArray populated by Pearson correlation coefficient values with the same dimensions as x, but reduced along obs_dim.
    """

    if spearman:
        if isinstance(obs_dim, list):
            y_pred = y_pred.stack(spearman_rank_dim = obs_dim)
            y_actual = y_actual.stack(spearman_rank_dim=obs_dim)
            y_actual = y_actual.rank('spearman_rank_dim')
            y_pred = y_pred.rank('spearman_rank_dim')
            obs_dim = 'spearman_rank_dim'
        else:
            y_actual = y_actual.rank(obs_dim)
            y_pred = y_pred.rank(obs_dim)
    assert y_actual is not None, 'If supplying pre-computed ysigma and yresids, do not pass y'

    ymean = y_actual.mean(obs_dim)
    ysigma = np.sqrt(np.square(y_actual - ymean).sum(obs_dim))
    yresids = y_actual - ymean

    xmean = y_pred.mean(obs_dim)
    xsigma = np.sqrt(np.square(y_pred - xmean).sum(obs_dim))
    xresids = y_pred - xmean

    r = (xresids * yresids).sum(obs_dim) / (xsigma * ysigma)
    return r
