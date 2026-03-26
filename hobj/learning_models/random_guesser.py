import typing

import numpy as np

from hobj.learning_models import BinaryLearningModel
from hobj.types import ImageId


class RandomGuesser(BinaryLearningModel):
    """
    A dummy model which simply returns random outputs, independent of the given image.
    """

    def __init__(self, seed: int = 0):
        self.random_generator = np.random.default_rng(seed=seed)

    def reset_state(self, seed: typing.Union[int, None]) -> None:
        self.random_generator = np.random.default_rng(seed=seed)

    def get_response(
        self,
        image: ImageId,
    ) -> typing.Literal[0, 1]:
        action = self.random_generator.integers(2)
        action = int(action)
        return action

    def give_feedback(self, reward: float) -> None:
        return
