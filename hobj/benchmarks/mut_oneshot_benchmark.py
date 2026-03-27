import collections
from pathlib import Path
from typing import Dict, List

from hobj.benchmarks.generalization.benchmark import (
    GeneralizationBenchmark,
    GeneralizationBenchmarkConfig,
    GeneralizationSessionResult,
)
from hobj.benchmarks.generalization.estimator import GeneralizationStatistics
from hobj.benchmarks.generalization.simulator import GeneralizationSubtask
from hobj.data.behavior import load_oneshot_behavior
from hobj.data.images import load_imageset_meta_oneshot

from hobj.types import ImageId


# %%
class MutatorOneshotBenchmark(GeneralizationBenchmark):
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
        support_trials = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        catch_trials = {9, 14, 19}

        # Load image manifest
        images_df = load_imageset_meta_oneshot(cachedir=cachedir)
        image_id_to_row = images_df.set_index("image_id")

        # Map image refs to transformation ids
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
                if row["category"] not in cat_to_support_image:
                    cat_to_support_image[row["category"]] = image_id
                else:
                    raise ValueError(
                        f"Multiple support images for category {row['category']}"
                    )
            else:
                if row["category"] not in cat_to_test_images:
                    cat_to_test_images[row["category"]] = []

                cat_to_test_images[row["category"]].append(image_id)

        # Assemble subtask simulators
        subtasks = []
        for subtask_name in self.subtask_names:
            catA, catB = sorted(subtask_name.split(","))

            support_imageA = cat_to_support_image[catA]
            support_imageB = cat_to_support_image[catB]
            test_imagesA = cat_to_test_images[catA]
            test_imagesB = cat_to_test_images[catB]
            assert len(test_imagesA) == len(test_imagesB) == 60

            subtask = GeneralizationSubtask(
                support_imageA=support_imageA,
                support_imageB=support_imageB,
                test_imagesA=test_imagesA,
                test_imagesB=test_imagesB,
                image_ref_to_transformation=image_ref_to_transformation_id,
            )
            subtasks.append(subtask)

        # Package human data into format expected by benchmark
        oneshot_df = load_oneshot_behavior(cachedir=cachedir)

        results = []

        for session, session_df in oneshot_df.groupby(
                ["assignment_id", "slot"], sort=False
        ):
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

                # Record performance in relevant slot
                if i_trial in support_trials:
                    assert annotation["transformation"] == "original"
                elif i_trial in catch_trials:
                    assert annotation["transformation"] == "original"
                    kcatch += perf
                    ncatch += 1
                else:
                    assert annotation["transformation"] != "original"
                    transformation_id = image_ref_to_transformation_id[image_id]

                    # Keep only benchmarked transformations
                    if transformation_id in self.transformation_ids:
                        transformation_to_kn[transformation_id][0] += perf
                        transformation_to_kn[transformation_id][1] += 1

            # Infer subtask name from observed categories
            assert len(observed_categories) == 2
            subtask_name = ",".join(sorted(observed_categories))
            if subtask_name not in self.subtask_names:
                raise ValueError(f"Unexpected subtask name: {subtask_name}")

            # Package result
            result = GeneralizationSessionResult(
                transformation_to_kn=transformation_to_kn,
                kcatch=kcatch,
                ncatch=ncatch,
                worker_id=session_df["worker_id"].iloc[0],
            )
            results.append(result)

        # %% Assemble benchmark config
        config = GeneralizationBenchmarkConfig(
            results=results,
            subtasks=subtasks,
            num_simulations_per_subtask=500,
            num_bootstrap_samples=1000,
            bootstrap_target_by_worker=True,
        )

        super().__init__(config=config)

    @property
    def target_statistics(self) -> GeneralizationStatistics:
        gen_statistics: GeneralizationStatistics = self._generalization_statistics
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


if __name__ == "__main__":
    benchmark = MutatorOneshotBenchmark()
