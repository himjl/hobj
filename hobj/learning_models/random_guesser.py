from typing import Literal

import numpy as np

from hobj.learning_models import BinaryLearningModel
from hobj.types import ImageId
from typing import cast


class RandomGuesser(BinaryLearningModel):
    """
    A dummy model which simply returns random outputs, independent of the given image.
    """

    def __init__(self, seed: int | None = None):
        self.random_generator = np.random.default_rng(seed=seed)

    def reset_state(self, seed: int | None) -> None:
        self.random_generator = np.random.default_rng(seed=seed)

    def get_response(
        self,
        image: ImageId,
    ) -> Literal[0, 1]:
        action = self.random_generator.integers(2)
        action = int(action)
        action = cast(Literal[0, 1], action)
        return action

    def give_feedback(self, reward: float) -> None:
        return
