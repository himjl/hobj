from abc import ABC, abstractmethod
import hobj.data.schema as schema
import typing
import PIL.Image
import numpy as np


class BinaryLearningModel(ABC):

    @abstractmethod
    def reset(self, seed: int) -> None:
        """
        This function resets the LearningModel to some initial state.
        :param seed: if the LearningModel has any stochastic components, this seed should be used to seed them.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def respond(
            self,
            image: typing.Union[schema.ImageRef, PIL.Image],
            offline: bool = False
    ) -> typing.Literal[0, 1]:
        """
        This function takes the current stimulus image (given by its image_url) and returns an action (parameterized by an integer).
        :param image: A public URL to the image.
        :param offline: If True, calling this function should not have any side effects.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, reward: float) -> None:
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

    def reset(self, seed: int) -> None:
        self.random_generator = np.random.default_rng(seed=seed)

    def respond(
            self,
            image: typing.Union[schema.ImageRef, PIL.Image],
            offline: bool = False
    ) -> typing.Literal[0, 1]:
        """
        This function takes the current stimulus image (given by its image_url) and returns an action (parameterized by an integer).
        :param image: An image (or a reference to an image).
        :param offline: If True, calling this function should not have any side effects.
        :return:
        """
        if offline:
            initial_state = self.random_generator.bit_generator.state
            action = self.random_generator.integers(2)
            self.random_generator.bit_generator.state = initial_state
        else:
            action = self.random_generator.integers(2)

        action = int(action)

        return action

    def learn(self, reward: float) -> None:
        """
        This function takes as input a scalar reward from the environment. The LearningModel may use this reward to update its parameters.
        :return:
        """
        return
