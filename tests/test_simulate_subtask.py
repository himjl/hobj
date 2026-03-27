from typing import List

import numpy as np

from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtask
from hobj.learning_models.random_guesser import RandomGuesser
from hobj.types import ImageId


def create_image_refs(nimages_per_class: int, seed: int) -> List[ImageId]:
    images = []
    np.random.seed(seed)
    for i in range(nimages_per_class):
        images.append(f"seed{seed}_image{i}")

    return images


def test_simulate_subtask():
    nimages_per_class = 10
    ntrials = nimages_per_class * 2

    subtask = BinaryClassificationSubtask(
        classA=create_image_refs(nimages_per_class=10, seed=0),
        classB=create_image_refs(nimages_per_class=10, seed=1),
        ntrials=ntrials,
        replace=False,
    )

    learner = RandomGuesser(seed=0)

    result = subtask.simulate_session(learner=learner, seed=0)

    assert len(result.perf_seq) == ntrials
