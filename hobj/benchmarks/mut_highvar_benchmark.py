import numpy as np
from typing import List, Dict

from hobj.benchmarks.binary_classification.benchmark import LearningCurveBenchmark, LearningCurveBenchmarkConfig, TargetSubtaskData
from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtask, BinaryClassificationSubtaskResult
from hobj.data.behavior import load_highvar_behavior
from hobj.images import MutatorHighVarImageset


# %%
class MutatorHighVarBenchmark(LearningCurveBenchmark):

    def __init__(self):
        # Load data into format expected by benchmark

        # Load imageset:
        imageset = MutatorHighVarImageset()

        # Load raw human session data for benchmark:
        sessions = load_highvar_behavior(remove_probe_trials=True)

        # Normalize data for benchmark:
        sha256_to_category = {
            ref.sha256: imageset.get_annotation(image_ref=ref).category for ref in imageset.image_refs
        }

        subtask_name_to_results = {}
        subtask_name_to_subtask = {}

        # Iterate over worker sessions:
        for session in sessions:

            categories = set()
            perf_seq: List[bool] = []

            # Iterate over trials:
            for i_trial, (reward, sha256) in enumerate(zip(session.reward_seq, session.stimulus_sha256_seq)):
                categories.add(sha256_to_category[sha256])
                perf_seq.append(reward > 0)

            # Infer subtask from the categories observed in the session:
            assert len(categories) == 2, f"Expected two categories, but got {categories}"
            subtask_name = ','.join(sorted(categories))
            cat0, cat1 = sorted(categories)

            # Instantiate the subtask if it does not exist:
            if subtask_name not in subtask_name_to_subtask:
                subtask = BinaryClassificationSubtask(
                    classA=imageset.category_to_image_refs[cat0],
                    classB=imageset.category_to_image_refs[cat1],
                    ntrials=100,
                    replace=False,
                )
                subtask_name_to_subtask[subtask_name] = subtask
                subtask_name_to_results[subtask_name] = []

            subtask_name_to_results[subtask_name].append(
                BinaryClassificationSubtaskResult(
                    perf_seq = np.array(perf_seq),
                    worker_id = session.worker_id
                )
            )

        # Cast as numpy
        for name in subtask_name_to_results:
            subtask_name_to_results[name] = np.array(subtask_name_to_results[name]) # [session, trial]

        subtask_name_to_data: Dict[str, TargetSubtaskData] = {}
        for name in subtask_name_to_results:
            subtask_name_to_data[name] = TargetSubtaskData(
                subtask=subtask_name_to_subtask[name],
                results=subtask_name_to_results[name]
            )

        # Instantiate benchmark config object
        config = LearningCurveBenchmarkConfig(
            subtask_name_to_data=subtask_name_to_data,
            num_simulations_per_subtask=500,
            num_bootstrap_samples=1000,
            bootstrap_by_worker=False,
            ntrials = 100
        )

        super().__init__(
            config=config
        )

if __name__ == '__main__':
    experiment = MutatorHighVarBenchmark()
    print(sorted(experiment.config.subtask_name_to_data.keys()))
