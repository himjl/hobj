from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from hobj.benchmarks.common import (
    compare_msen,
    simulate_sessions,
    summarize_bootstrap_score,
)
from hobj.benchmarks.binary_classification.estimator import LearningCurveStatistics
from hobj.benchmarks.binary_classification.simulation import (
    BinaryClassificationSubtask,
    BinaryClassificationSubtaskResult,
)
from hobj.data.behavior import load_highvar_behavior
from hobj.data.images import load_imageset_meta_highvar
from hobj.learning_models import BinaryLearningModel


class MutatorHighVarBenchmark:
    num_simulations_per_subtask = 500
    num_bootstrap_samples = 1000
    bootstrap_by_worker = False
    ntrials = 100

    def __init__(self, cachedir: Path | None = None):
        images_df = load_imageset_meta_highvar(cachedir=cachedir)
        df_behavior = load_highvar_behavior(
            remove_probe_trials=True,
            cachedir=cachedir,
        )

        image_id_to_category = dict(
            zip(images_df["image_id"], images_df["category"], strict=True)
        )
        category_to_image_ids = (
            images_df.groupby("category", sort=False)["image_id"]
            .apply(lambda image_ids: sorted(image_ids.tolist()))
            .to_dict()
        )

        subtask_name_to_results: Dict[str, List[BinaryClassificationSubtaskResult]] = {}
        subtask_name_to_subtask: Dict[str, BinaryClassificationSubtask] = {}

        for _, session in df_behavior.groupby("assignment_id", sort=False):
            categories = set()
            perf_seq: List[bool] = []

            session = session.sort_values("trial")
            for perf, image_id in zip(session.perf, session.image_id):
                categories.add(image_id_to_category[image_id])
                perf_seq.append(bool(perf))

            if len(categories) != 2:
                raise ValueError(f"Expected two categories, but got {categories}")

            cat0, cat1 = sorted(categories)
            subtask_name = ",".join((cat0, cat1))

            if subtask_name not in subtask_name_to_subtask:
                subtask_name_to_subtask[subtask_name] = BinaryClassificationSubtask(
                    classA=category_to_image_ids[cat0],
                    classB=category_to_image_ids[cat1],
                    ntrials=self.ntrials,
                    replace=False,
                )
                subtask_name_to_results[subtask_name] = []

            subtask_name_to_results[subtask_name].append(
                BinaryClassificationSubtaskResult(
                    perf_seq=np.array(perf_seq),
                    worker_id=session.worker_id.iloc[0],
                )
            )

        self.subtask_names = sorted(subtask_name_to_subtask)
        self.subtask_name_to_subtask = {
            name: subtask_name_to_subtask[name] for name in self.subtask_names
        }
        self.subtask_name_to_results = {
            name: subtask_name_to_results[name] for name in self.subtask_names
        }

        observed_ntrials = {
            subtask.ntrials for subtask in self.subtask_name_to_subtask.values()
        }
        if len(observed_ntrials) != 1:
            raise ValueError(
                f"Expected all subtasks to have the same number of trials, but got {observed_ntrials}"
            )

        observed_ntrials_value = next(iter(observed_ntrials))
        if observed_ntrials_value != self.ntrials:
            raise ValueError(
                f"Expected ntrials to be {observed_ntrials_value}, but got {self.ntrials}"
            )

        self._target_data: Dict[str, Dict[str, List[bool]]] = {}
        for name in self.subtask_names:
            self._target_data[name] = {}
            for result in self.subtask_name_to_results[name]:
                worker_id = result.worker_id
                if worker_id in self._target_data[name]:
                    raise ValueError(
                        f"Worker {worker_id} has already been seen for subtask {name}"
                    )
                self._target_data[name][worker_id] = [bool(v) for v in result.perf_seq]

        self._target_statistics = LearningCurveStatistics(
            subtask_name_to_results=self.subtask_name_to_results,
            nbootstrap_samples=self.num_bootstrap_samples,
            bootstrap_by_worker=self.bootstrap_by_worker,
        )

    @property
    def target_data(self) -> Dict[str, Dict[str, List[bool]]]:
        return self._target_data

    @property
    def target_statistics(self) -> LearningCurveStatistics:
        return self._target_statistics

    @dataclass
    class LearningCurveBenchmarkResult:
        msen: float
        msen_sigma: float
        msen_CI95: Tuple[float, float]
        lapse_rate: xr.DataArray | None
        model_statistics: LearningCurveStatistics

    def __call__(
        self, learner: BinaryLearningModel, show_pbar: bool = False
    ) -> "MutatorHighVarBenchmark.LearningCurveBenchmarkResult":
        subtask_name_to_model_results: Dict[
            str, List[BinaryClassificationSubtaskResult]
        ] = {}
        for subtask_name in tqdm(
            self.subtask_names,
            desc="Subtask simulations:",
            disable=not show_pbar,
        ):
            subtask_name_to_model_results[subtask_name] = simulate_sessions(
                subtask=self.subtask_name_to_subtask[subtask_name],
                learner=learner,
                nsimulations=self.num_simulations_per_subtask,
            )

        model_statistics = LearningCurveStatistics(
            subtask_name_to_results=subtask_name_to_model_results,
            nbootstrap_samples=self.num_bootstrap_samples,
            bootstrap_by_worker=False,
        )

        msen_point, lapse_rate = self._compare_learning_curves(
            model_phat=model_statistics.phat,
            model_varhat_phat=model_statistics.varhat_phat,
            target_phat=self.target_statistics.phat,
            target_varhat_phat=self.target_statistics.varhat_phat,
            condition_dims=("subtask", "trial"),
            fit_lapse_rate=True,
        )

        msen_boot, _ = self._compare_learning_curves(
            model_phat=model_statistics.boot_phat,
            model_varhat_phat=model_statistics.boot_varhat_phat,
            target_phat=self.target_statistics.boot_phat,
            target_varhat_phat=self.target_statistics.boot_varhat_phat,
            condition_dims=("subtask", "trial"),
            fit_lapse_rate=True,
        )

        summary = summarize_bootstrap_score(
            point_estimate=msen_point,
            bootstrapped_point_estimates=msen_boot,
        )

        return self.LearningCurveBenchmarkResult(
            msen=float(msen_point),
            msen_sigma=summary.sigma,
            msen_CI95=summary.ci95,
            lapse_rate=lapse_rate,
            model_statistics=model_statistics,
        )

    @classmethod
    def _compare_learning_curves(
        cls,
        model_phat: xr.DataArray,
        model_varhat_phat: xr.DataArray,
        target_phat: xr.DataArray,
        target_varhat_phat: xr.DataArray,
        condition_dims: Tuple[str, ...],
        fit_lapse_rate: bool,
    ) -> Tuple[xr.DataArray, xr.DataArray | None]:
        if fit_lapse_rate:
            lapse_rate = cls._fit_lapse_rate(
                pmodel=model_phat,
                ptarget=target_phat,
                condition_dims=condition_dims,
            )
            model_phat = model_phat * (1 - lapse_rate) + 0.5 * lapse_rate
            model_varhat_phat = model_varhat_phat * (1 - lapse_rate) ** 2
        else:
            lapse_rate = None

        msen = compare_msen(
            model_phat=model_phat,
            model_varhat_phat=model_varhat_phat,
            target_phat=target_phat,
            target_varhat_phat=target_varhat_phat,
            condition_dims=condition_dims,
        )
        return msen, lapse_rate

    @staticmethod
    def _fit_lapse_rate(
        pmodel: xr.DataArray,
        ptarget: xr.DataArray,
        condition_dims: Tuple[str, ...],
    ) -> xr.DataArray:
        nway = 2
        numerator = -(
            2 * pmodel / nway
            - 2 * np.square(pmodel)
            + 2 * pmodel * ptarget
            - 2 * ptarget / nway
        ).sum(dim=condition_dims)
        denominator = (2 / (nway**2) - 4 * pmodel / nway + 2 * (pmodel**2)).sum(
            dim=condition_dims
        )
        gamma_star = numerator / denominator
        gamma_star = np.clip(gamma_star, 0, 1)
        return gamma_star


if __name__ == "__main__":
    experiment = MutatorHighVarBenchmark()
