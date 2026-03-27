import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from tqdm import tqdm

from hobj.benchmarks.generalization.estimator import GeneralizationStatistics
from hobj.benchmarks.generalization.simulator import (
    GeneralizationSessionResult,
    GeneralizationSubtask,
)
from hobj.data.behavior import load_oneshot_behavior
from hobj.data.images import load_imageset_meta_oneshot
from hobj.learning_models import BinaryLearningModel
from hobj.stats.ci import estimate_basic_bootstrap_CI
from hobj.types import ImageId


class MutatorOneshotBenchmark:
    num_simulations_per_subtask = 500
    num_bootstrap_samples = 1000
    bootstrap_target_by_worker = True

    subtask_names = [
        "MutatorOneshotObject00,MutatorOneshotObject43",
        "MutatorOneshotObject01,MutatorOneshotObject37",
        "MutatorOneshotObject02,MutatorOneshotObject36",
        "MutatorOneshotObject03,MutatorOneshotObject55",
        "MutatorOneshotObject04,MutatorOneshotObject27",
        "MutatorOneshotObject05,MutatorOneshotObject17",
        "MutatorOneshotObject06,MutatorOneshotObject14",
        "MutatorOneshotObject07,MutatorOneshotObject50",
        "MutatorOneshotObject08,MutatorOneshotObject26",
        "MutatorOneshotObject09,MutatorOneshotObject58",
        "MutatorOneshotObject10,MutatorOneshotObject44",
        "MutatorOneshotObject11,MutatorOneshotObject32",
        "MutatorOneshotObject12,MutatorOneshotObject31",
        "MutatorOneshotObject13,MutatorOneshotObject41",
        "MutatorOneshotObject15,MutatorOneshotObject19",
        "MutatorOneshotObject16,MutatorOneshotObject28",
        "MutatorOneshotObject18,MutatorOneshotObject21",
        "MutatorOneshotObject20,MutatorOneshotObject47",
        "MutatorOneshotObject22,MutatorOneshotObject48",
        "MutatorOneshotObject23,MutatorOneshotObject35",
        "MutatorOneshotObject24,MutatorOneshotObject29",
        "MutatorOneshotObject25,MutatorOneshotObject40",
        "MutatorOneshotObject30,MutatorOneshotObject61",
        "MutatorOneshotObject33,MutatorOneshotObject38",
        "MutatorOneshotObject34,MutatorOneshotObject57",
        "MutatorOneshotObject39,MutatorOneshotObject52",
        "MutatorOneshotObject42,MutatorOneshotObject59",
        "MutatorOneshotObject45,MutatorOneshotObject53",
        "MutatorOneshotObject46,MutatorOneshotObject62",
        "MutatorOneshotObject49,MutatorOneshotObject51",
        "MutatorOneshotObject54,MutatorOneshotObject63",
        "MutatorOneshotObject56,MutatorOneshotObject60",
    ]
    transformation_ids = [
        "backgrounds | 0.1",
        "backgrounds | 0.215443",
        "backgrounds | 0.464159",
        "backgrounds | 1.0",
        "blur | 0.007812",
        "blur | 0.015625",
        "blur | 0.03125",
        "blur | 0.0625",
        "contrast | -0.4",
        "contrast | -0.8",
        "contrast | 0.4",
        "contrast | 0.8",
        "delpixels | 0.25",
        "delpixels | 0.5",
        "delpixels | 0.75",
        "delpixels | 0.95",
        "inplanerotation | 135.0",
        "inplanerotation | 180.0",
        "inplanerotation | 45.0",
        "inplanerotation | 90.0",
        "inplanetranslation | 0.125",
        "inplanetranslation | 0.25",
        "inplanetranslation | 0.5",
        "inplanetranslation | 0.75",
        "noise | 0.125",
        "noise | 0.25",
        "noise | 0.375",
        "noise | 0.5",
        "outplanerotation | 135.0",
        "outplanerotation | 180.0",
        "outplanerotation | 45.0",
        "outplanerotation | 90.0",
        "scale | 0.125",
        "scale | 0.25",
        "scale | 0.5",
        "scale | 1.5",
    ]

    def __init__(self, cachedir: Path | None = None):
        support_trials = set(range(9))
        catch_trials = {9, 14, 19}

        images_df = load_imageset_meta_oneshot(cachedir=cachedir)
        image_id_to_row = images_df.set_index("image_id")

        image_ref_to_transformation_id: Dict[ImageId, str] = {}
        cat_to_support_image: Dict[str, ImageId] = {}
        cat_to_test_images: Dict[str, List[ImageId]] = {}

        for row in images_df.to_dict(orient="records"):
            image_id = row["image_id"]
            transformation_id = (
                f"{row['transformation']} | {row['transformation_level']}"
            )
            image_ref_to_transformation_id[image_id] = transformation_id

            if row["transformation"] == "original":
                if row["category"] in cat_to_support_image:
                    raise ValueError(
                        f"Multiple support images for category {row['category']}"
                    )
                cat_to_support_image[row["category"]] = image_id
            else:
                if row["category"] not in cat_to_test_images:
                    cat_to_test_images[row["category"]] = []

                cat_to_test_images[row["category"]].append(image_id)

        self.subtasks = []
        for subtask_name in self.subtask_names:
            cat_a, cat_b = sorted(subtask_name.split(","))
            test_images_a = cat_to_test_images[cat_a]
            test_images_b = cat_to_test_images[cat_b]

            if len(test_images_a) != 60 or len(test_images_b) != 60:
                raise ValueError(
                    f"Expected 60 test images per category for {subtask_name}, "
                    f"but got {len(test_images_a)} and {len(test_images_b)}."
                )

            self.subtasks.append(
                GeneralizationSubtask(
                    support_imageA=cat_to_support_image[cat_a],
                    support_imageB=cat_to_support_image[cat_b],
                    test_imagesA=test_images_a,
                    test_imagesB=test_images_b,
                    image_ref_to_transformation=image_ref_to_transformation_id,
                )
            )

        oneshot_df = load_oneshot_behavior(cachedir=cachedir)
        self.results: List[GeneralizationSessionResult] = []

        for _, session_df in oneshot_df.groupby(["assignment_id", "slot"], sort=False):
            session_df = session_df.sort_values("trial")
            transformation_to_kn = collections.defaultdict(lambda: [0, 0])
            kcatch = 0
            ncatch = 0
            observed_categories = set(session_df["subtask"].iloc[0].split(","))

            for _, row in session_df.iterrows():
                i_trial = int(row["trial"])
                image_id = row["image_id"]
                annotation = image_id_to_row.loc[image_id]
                perf = bool(row["perf"])

                if i_trial in support_trials:
                    if annotation["transformation"] != "original":
                        raise ValueError(
                            f"Expected support trial {i_trial} to use an original image."
                        )
                elif i_trial in catch_trials:
                    if annotation["transformation"] != "original":
                        raise ValueError(
                            f"Expected catch trial {i_trial} to use an original image."
                        )
                    kcatch += perf
                    ncatch += 1
                else:
                    if annotation["transformation"] == "original":
                        raise ValueError(
                            f"Expected generalization trial {i_trial} to use a transformed image."
                        )

                    transformation_id = image_ref_to_transformation_id[image_id]
                    if transformation_id in self.transformation_ids:
                        transformation_to_kn[transformation_id][0] += perf
                        transformation_to_kn[transformation_id][1] += 1

            if len(observed_categories) != 2:
                raise ValueError(
                    f"Expected two categories for a one-shot session, got {observed_categories}."
                )

            subtask_name = ",".join(sorted(observed_categories))
            if subtask_name not in self.subtask_names:
                raise ValueError(f"Unexpected subtask name: {subtask_name}")

            self.results.append(
                GeneralizationSessionResult(
                    transformation_to_kn=transformation_to_kn,
                    kcatch=kcatch,
                    ncatch=ncatch,
                    worker_id=session_df["worker_id"].iloc[0],
                )
            )

        self._target_statistics = GeneralizationStatistics(
            results=self.results,
            perform_lapse_rate_correction=True,
            n_bootstrap_iterations=self.num_bootstrap_samples,
            bootstrap_by_worker=self.bootstrap_target_by_worker,
        )

    @property
    def target_statistics(self) -> GeneralizationStatistics:
        gen_statistics = self._target_statistics
        return gen_statistics.assign_coords(
            transformation_type=(
                ["transformation"],
                [
                    transformation.split(" | ")[0]
                    for transformation in gen_statistics.transformation.values
                ],
            ),
            transformation_level=(
                ["transformation"],
                [
                    float(transformation.split(" | ")[1])
                    for transformation in gen_statistics.transformation.values
                ],
            ),
        )

    @dataclass
    class GeneralizationBenchmarkResult:
        msen: float
        msen_sigma: float
        msen_CI95: Tuple[float, float]
        model_statistics: GeneralizationStatistics

    def __call__(
        self, learner: BinaryLearningModel, show_pbar: bool = False
    ) -> "MutatorOneshotBenchmark.GeneralizationBenchmarkResult":
        results: List[GeneralizationSessionResult] = []
        for subtask in tqdm(
            self.subtasks,
            desc="Subtask simulations:",
            disable=not show_pbar,
        ):
            results.extend(
                self.simulate_model_behavior(
                    subtask=subtask,
                    learner=learner,
                    nsimulations=self.num_simulations_per_subtask,
                )
            )

        model_statistics = GeneralizationStatistics(
            results=results,
            perform_lapse_rate_correction=False,
            n_bootstrap_iterations=self.num_bootstrap_samples,
            bootstrap_by_worker=False,
        )
        model_statistics = model_statistics.sel(
            transformation=self.target_statistics.transformation
        )

        msen_point = self._compare_generalization_patterns(
            model_phat=model_statistics.phat,
            model_varhat_phat=model_statistics.varhat_phat,
            target_phat=self.target_statistics.phat,
            target_varhat_phat=self.target_statistics.varhat_phat,
            condition_dims=("transformation",),
        )
        msen_boot = self._compare_generalization_patterns(
            model_phat=model_statistics.boot_phat,
            model_varhat_phat=model_statistics.boot_varhat_phat,
            target_phat=self.target_statistics.boot_phat,
            target_varhat_phat=self.target_statistics.boot_varhat_phat,
            condition_dims=("transformation",),
        )

        msen_sigma = np.std(msen_boot, ddof=1)
        msen_CI95 = estimate_basic_bootstrap_CI(
            alpha=0.05,
            point_estimate=msen_point,
            bootstrapped_point_estimates=np.array(msen_boot),
        )

        return self.GeneralizationBenchmarkResult(
            msen=float(msen_point),
            msen_sigma=float(msen_sigma),
            msen_CI95=msen_CI95,
            model_statistics=model_statistics,
        )

    @staticmethod
    def simulate_model_behavior(
        subtask: GeneralizationSubtask,
        learner: BinaryLearningModel,
        nsimulations: int,
    ) -> List[GeneralizationSessionResult]:
        results = []
        for _ in range(nsimulations):
            learner.reset_state(seed=None)
            results.append(
                subtask.simulate_session(
                    learner=learner,
                    seed=None,
                )
            )

        return results

    @staticmethod
    def _compare_generalization_patterns(
        model_phat: xr.DataArray,
        model_varhat_phat: xr.DataArray,
        target_phat: xr.DataArray,
        target_varhat_phat: xr.DataArray,
        condition_dims: Tuple[str, ...],
    ) -> xr.DataArray:
        msen = (
            np.square(model_phat - target_phat).mean(condition_dims)
            - model_varhat_phat.mean(condition_dims)
            - target_varhat_phat.mean(condition_dims)
        )
        return msen


if __name__ == "__main__":
    benchmark = MutatorOneshotBenchmark()
