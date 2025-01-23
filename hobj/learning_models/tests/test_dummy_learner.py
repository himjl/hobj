import pytest
from hobj.learning_models import DummyBinaryLearner
import hobj.data.schema as schema

@pytest.fixture
def dummy_learner() -> DummyBinaryLearner:
    return DummyBinaryLearner(seed=0)

@pytest.fixture
def test_image() -> schema.ImageRef:
    return schema.ImageRef(sha256='0'*64)


def test_dummy_learner_deterministic(dummy_learner, test_image):

    ntests = 10
    actions = []
    expected_actions = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    for i in range(ntests):
        a = dummy_learner.get_response(image=test_image)
        actions.append(a)

    assert actions == expected_actions