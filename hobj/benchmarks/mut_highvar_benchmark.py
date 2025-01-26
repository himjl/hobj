from typing import List, Dict
import numpy as np

from hobj.benchmarks.binary_classification.scoring import LearningCurveBenchmark, LearningCurveBenchmarkConfig, TargetSubtaskData
from hobj.benchmarks.binary_classification.task import BinaryClassificationSubtask
from hobj.data.behavior import load_highvar_behavior
from hobj.data.images import MutatorHighVarImageset


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

            subtask_name_to_results[subtask_name].append(perf_seq)

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
        )

        super().__init__(
            config=config
        )


# %%
if __name__ == '__main__':
    benchmark = MutatorHighVarBenchmark()

    from hobj.learning_models import DummyBinaryLearner

    learner = DummyBinaryLearner()

    result = benchmark(learner=learner, show_pbar=True)
    print(result.msen_CI95)

    # %%
    import matplotlib.pyplot as plt
    glc = benchmark.target_statistics.phat.mean('subtask')
    glc_sigma = benchmark.target_statistics.boot_phat.mean('subtask').std('boot_iter')
    x = glc.trial.values + 1
    plt.errorbar(x, glc, yerr = glc_sigma, marker = '.')
    plt.xscale('log')
    plt.show()