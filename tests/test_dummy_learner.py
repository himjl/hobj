import pytest

from hobj.learning_models.random_guesser import RandomGuesser


@pytest.fixture
def dummy_learner() -> RandomGuesser:
    return RandomGuesser(seed=0)


def test_dummy_learner_deterministic(dummy_learner):
    ntests = 10
    actions = []
    expected_actions = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    for i in range(ntests):
        a = dummy_learner.get_response(image="hi")
        actions.append(a)

    assert actions == expected_actions
