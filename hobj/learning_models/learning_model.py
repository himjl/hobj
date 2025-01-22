from abc import ABC, abstractmethod
import hobj.data.schema as schema
import typing
import PIL.Image
import numpy as np


class BinaryLearningModel(ABC):

    @abstractmethod
    def reset_state(self, seed: int) -> None:
        """
        This function resets the LearningModel to some initial state.
        :param seed: if the LearningModel has any stochastic components, this seed should be used to seed them.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(
            self,
            image: typing.Union[schema.ImageRef, PIL.Image],
    ) -> typing.Literal[0, 1]:
        """
        This function takes the current stimulus image (given either as a PIL.Image or a ImageRef) and returns one of two possible actions (parameterized by an integer).
        :param image: An image (or a reference to an image).
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def give_feedback(self, reward: float) -> None:
        """
        This function takes as input a scalar reward from the environment. The LearningModel may use this reward to update its parameters.
        :return:
        """
        raise NotImplementedError


# %%
class DummyBinaryLearner(BinaryLearningModel):
    """
    A dummy model which simply returns random outputs, independent of the given image.
    """

    def __init__(self, seed: int = 0):
        self.random_generator = np.random.default_rng(seed=seed)

    def reset_state(self, seed: int) -> None:
        self.random_generator = np.random.default_rng(seed=seed)

    def get_response(
            self,
            image: typing.Union[schema.ImageRef, PIL.Image],
    ) -> typing.Literal[0, 1]:

        action = self.random_generator.integers(2)
        action = int(action)
        return action

    def give_feedback(self, reward: float) -> None:
        return
