import pytest

import mref.media_references
from hobj.learning_models import RandomGuesser

@pytest.fixture
def dummy_learner() -> RandomGuesser:
    return RandomGuesser(seed=0)

@pytest.fixture
def test_image() -> mref.media_references.ImageRef:
    return mref.media_references.ImageRef(sha256='0' * 64)


def test_dummy_learner_deterministic(dummy_learner, test_image):

    ntests = 10
    actions = []
    expected_actions = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    for i in range(ntests):
        a = dummy_learner.get_response(image=test_image)
        actions.append(a)

    assert actions == expected_actions