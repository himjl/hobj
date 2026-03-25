from typing import Dict, List

import numpy as np

from hobj.benchmarks.binary_classification.benchmark import LearningCurveBenchmark, LearningCurveBenchmarkConfig, TargetSubtaskData
from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtask, BinaryClassificationSubtaskResult
from hobj.data_loaders.behavior import load_highvar_behavior
from hobj.data_loaders.images import MutatorHighVarImageset


# %%
class MutatorHighVarBenchmark(LearningCurveBenchmark):

    def __init__(self):
        # Load data into format expected by benchmark

        # Load imageset:
        imageset = MutatorHighVarImageset()

        # Load raw human session data for benchmark:
        sessions = load_highvar_behavior(remove_probe_trials=True)

        # Normalize data for benchmark:
        stimulus_id_to_category = {
            image_id: imageset.get_annotation(image_id=image_id).category for image_id in imageset.image_ids
        }

        subtask_name_to_results = {}
        subtask_name_to_subtask = {}

        # Iterate over worker sessions:
        for assignment_id, session in sessions.groupby('assignment_id', sort=False):

            categories = set()
            perf_seq: List[bool] = []

            # Iterate over trials:
            session = session.sort_values('trial')
            for reward, stimulus_id in zip(session.perf, session.stimulus_id):
                categories.add(stimulus_id_to_category[stimulus_id])
                perf_seq.append(bool(reward))

            # Infer subtask from the categories observed in the session:
            assert len(categories) == 2, f"Expected two categories, but got {categories}"
            subtask_name = ','.join(sorted(categories))
            cat0, cat1 = sorted(categories)

            # Instantiate the subtask if it does not exist:
            if subtask_name not in subtask_name_to_subtask:
                subtask = BinaryClassificationSubtask(
                    classA=imageset.category_to_image_ids[cat0],
                    classB=imageset.category_to_image_ids[cat1],
                    ntrials=100,
                    replace=False,
                )
                subtask_name_to_subtask[subtask_name] = subtask
                subtask_name_to_results[subtask_name] = []

            subtask_name_to_results[subtask_name].append(
                BinaryClassificationSubtaskResult(
                    perf_seq = np.array(perf_seq),
                    worker_id = session.worker_id.iloc[0]
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

# %%
if __name__ == '__main__':
    experiment = MutatorHighVarBenchmark()
    print(sorted(experiment.config.subtask_name_to_data.keys()))
