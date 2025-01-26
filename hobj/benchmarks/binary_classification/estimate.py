from typing import List, Dict

import numpy as np
import xarray as xr
from hobj.stats_new.learning_curve import calculate_learning_curve


class LearningCurveStatistics(xr.Dataset):

    __slots__ = ()

    def __init__(
            self,
            subtask_name_to_perf: Dict[str, np.ndarray],  # subtask: [session, trial]
            nbootstrap_samples: int,
    ):
        # Ensure rectangular data
        ntrials_observed = set()
        subtask_name_to_nsessions: Dict[str, int] = {}
        for subtask_name, perf_data in subtask_name_to_perf.items():
            ntrials_observed.add(perf_data.shape[1])
            subtask_name_to_nsessions[subtask_name] = perf_data.shape[0]

        if len(ntrials_observed) != 1:
            raise ValueError(f"Expected all subtask data to have the same number of trials, but got ntrials={ntrials_observed}")

        # Preallocate arrays
        subtask_names = sorted(subtask_name_to_perf.keys())
        nsubtasks = len(subtask_names)
        ntrials = ntrials_observed.pop()

        phat = np.zeros(shape=(nsubtasks, ntrials))
        varhat_phat = np.zeros(shape=(nsubtasks, ntrials))
        boot_phat = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials))
        boot_varhat_phat = np.zeros(shape=(nbootstrap_samples, nsubtasks, ntrials))

        # Compute statistics
        for i_subtask, (subtask_name, perf_mat) in enumerate(subtask_name_to_perf.items()):

            # todo: move this function here
            subtask_learning_curve_statistics = calculate_learning_curve(
                perf_matrix=perf_mat,
                bootstrap_seed=i_subtask,
                nbootstrap_samples=nbootstrap_samples,
                repetition_axis=0,
            )

            phat[i_subtask] = subtask_learning_curve_statistics.phat
            varhat_phat[i_subtask] = subtask_learning_curve_statistics.varhat_phat
            boot_phat[:, i_subtask] = subtask_learning_curve_statistics.boot_phat
            boot_varhat_phat[:, i_subtask] = subtask_learning_curve_statistics.boot_varhat_phat

        super().__init__(
            data_vars=dict(
                phat=(['subtask', 'trial'], phat),
                varhat_phat=(['subtask', 'trial'], varhat_phat),
                boot_phat=(['boot_iter', 'subtask', 'trial'], boot_phat),
                boot_varhat_phat=(['boot_iter', 'subtask', 'trial'], boot_varhat_phat),
            ),
            coords=dict(
                subtask=subtask_names,
                nsessions=('subtask', [subtask_name_to_nsessions[name] for name in subtask_names]),
            )
        )
