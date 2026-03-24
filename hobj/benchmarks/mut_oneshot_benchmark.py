import collections
# Coercing human data
from typing import Dict, List

from mref import ImageRef

from hobj.benchmarks.generalization.benchmark import GeneralizationBenchmark, GeneralizationBenchmarkConfig, GeneralizationSessionResult
from hobj.benchmarks.generalization.estimator import GeneralizationStatistics
from hobj.benchmarks.generalization.simulator import GeneralizationSubtask
from hobj.data_loaders.behavior import load_oneshot_behavior
from hobj.data_loaders.images import MutatorOneShotImageset


# %%

class MutatorOneshotBenchmark(GeneralizationBenchmark):

    subtask_names = [
        'MutatorB2000_2292,MutatorB2000_2444',
        'MutatorB2000_138,MutatorB2000_2344',
        'MutatorB2000_1251,MutatorB2000_953',
        'MutatorB2000_3043,MutatorB2000_694',
        'MutatorB2000_3496,MutatorB2000_496',
        'MutatorB2000_1219,MutatorB2000_296',
        'MutatorB2000_1825,MutatorB2000_2757',
        'MutatorB2000_3077,MutatorB2000_4703',
        'MutatorB2000_270,MutatorB2000_3615',
        'MutatorB2000_3066,MutatorB2000_3585',
        'MutatorB2000_2139,MutatorB2000_746',
        'MutatorB2000_116,MutatorB2000_2365',
        'MutatorB2000_2130,MutatorB2000_4628',
        'MutatorB2000_462,MutatorB2000_926',
        'MutatorB2000_2304,MutatorB2000_3733',
        'MutatorB2000_1363,MutatorB2000_3278',
        'MutatorB2000_4049,MutatorB2000_663',
        'MutatorB2000_2722,MutatorB2000_3527',
        'MutatorB2000_2832,MutatorB2000_801',
        'MutatorB2000_1258,MutatorB2000_3123',
        'MutatorB2000_1865,MutatorB2000_613',
        'MutatorB2000_1164,MutatorB2000_2106',
        'MutatorB2000_1229,MutatorB2000_1280',
        'MutatorB2000_1767,MutatorB2000_2122',
        'MutatorB2000_2198,MutatorB2000_701',
        'MutatorB2000_3636,MutatorB2000_4305',
        'MutatorB2000_3035,MutatorB2000_46',
        'MutatorB2000_3601,MutatorB2000_4792',
        'MutatorB2000_2092,MutatorB2000_288',
        'MutatorB2000_1424,MutatorB2000_2314',
        'MutatorB2000_3308,MutatorB2000_3525',
        'MutatorB2000_2909,MutatorB2000_4256'
    ]
    transformation_ids = [
        'backgrounds | 0.1',
        'backgrounds | 0.215443',
        'backgrounds | 0.464159',
        'backgrounds | 1.0',
        'blur | 0.007812',
        'blur | 0.015625',
        'blur | 0.03125',
        'blur | 0.0625',
        'contrast | -0.4',
        'contrast | -0.8',
        'contrast | 0.4',
        'contrast | 0.8',
        'delpixels | 0.25',
        'delpixels | 0.5',
        'delpixels | 0.75',
        'delpixels | 0.95',
        'inplanerotation | 135.0',
        'inplanerotation | 180.0',
        'inplanerotation | 45.0',
        'inplanerotation | 90.0',
        'inplanetranslation | 0.125',
        'inplanetranslation | 0.25',
        'inplanetranslation | 0.5',
        'inplanetranslation | 0.75',
        'noise | 0.125',
        'noise | 0.25',
        'noise | 0.375',
        'noise | 0.5',
        'outplanerotation | 135.0',
        'outplanerotation | 180.0',
        'outplanerotation | 45.0',
        'outplanerotation | 90.0',
        'scale | 0.125',
        'scale | 0.25',
        'scale | 0.5',
        'scale | 1.5'
    ]

    def __init__(self):
        support_trials = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        catch_trials = {9, 14, 19}

        # Load images
        imageset = MutatorOneShotImageset()

        # Map image refs to transformation ids
        image_ref_to_transformation_id = {}
        cat_to_support_image: Dict[str, ImageRef] = {}
        cat_to_test_images: Dict[str, List[ImageRef]] = {}

        for ref in imageset.image_ids:
            annotation = imageset.get_annotation(image_id=ref)
            transformation_id = f"{annotation.transformation} | {annotation.transformation_level}"
            image_ref_to_transformation_id[ref] = transformation_id

            if annotation.transformation == 'original':
                if annotation.category not in cat_to_support_image:
                    cat_to_support_image[annotation.category] = ref
                else:
                    raise ValueError(f"Multiple support images for category {annotation.category}")
            else:
                if annotation.category not in cat_to_test_images:
                    cat_to_test_images[annotation.category] = []

                cat_to_test_images[annotation.category].append(ref)

        # Assemble subtask simulators
        subtasks = []
        for subtask_name in self.subtask_names:
            catA, catB = sorted(subtask_name.split(','))

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
                image_ref_to_transformation=image_ref_to_transformation_id
            )
            subtasks.append(subtask)

        # Package human data into format expected by benchmark
        oneshot_sessions = load_oneshot_behavior()

        results = []

        for session in oneshot_sessions:
            # Parse raw data by worker
            transformation_to_kn = collections.defaultdict(lambda: [0, 0])
            kcatch = 0
            ncatch = 0
            observed_categories = set()

            for i_trial, sha in enumerate(session.stimulus_sha256_seq):
                ref = ImageRef(sha256=sha)
                annotation = imageset.get_annotation(image_id=ref)

                # Add stimulus category to observed categories
                observed_categories.add(annotation.category)

                # Calculate performance
                perf = session.reward_seq[i_trial] > 0

                # Record performance in relevant slot
                if i_trial in support_trials:
                    assert annotation.transformation == 'original'
                elif i_trial in catch_trials:
                    assert annotation.transformation == 'original'
                    kcatch += perf
                    ncatch += 1
                else:
                    assert annotation.transformation != 'original'
                    transformation_id = image_ref_to_transformation_id[ref]

                    # Keep only benchmarked transformations
                    if transformation_id in self.transformation_ids:
                        transformation_to_kn[transformation_id][0] += perf
                        transformation_to_kn[transformation_id][1] += 1

            # Infer subtask name from observed categories
            assert len(observed_categories) == 2
            subtask_name = ','.join(sorted(observed_categories))
            if subtask_name not in self.subtask_names:
                raise ValueError(f"Unexpected subtask name: {subtask_name}")

            # Package result
            result = GeneralizationSessionResult(
                transformation_to_kn=transformation_to_kn,
                kcatch=kcatch,
                ncatch=ncatch,
                worker_id=session.worker_id
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

        gen_statistics: GeneralizationStatistics =  self._generalization_statistics
        return gen_statistics.assign_coords(
            transformation_type=(['transformation'], [transformation.split(' | ')[0] for transformation in gen_statistics.transformation.values]),
            transformation_level=(['transformation'], [float(transformation.split(' | ')[1]) for transformation in gen_statistics.transformation.values])
        )


if __name__ == '__main__':
    benchmark = MutatorOneshotBenchmark()

