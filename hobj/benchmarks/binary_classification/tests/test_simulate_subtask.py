from hobj.benchmarks.binary_classification.simulation import BinaryClassificationSubtask
from hobj.learning_models import RandomGuesser
from mref import ImageRef
from typing import List
import pytest
import PIL.Image
import numpy as np


def create_image_refs(nimages_per_class: int, seed: int) -> List[ImageRef]:
    images = []
    np.random.seed(seed)
    for _ in range(nimages_per_class):
        image = PIL.Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        image_ref = ImageRef.from_image(image=image)
        images.append(image_ref)

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

    result = subtask.simulate_session(
        learner=learner,
        seed=0
    )

    assert len(result.perf_seq) == ntrials


# Todo: test deterministic