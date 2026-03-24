import typing
from abc import ABC, abstractmethod

import numpy as np

from hobj.learning_models.representation import RepresentationalModel
from hobj.learning_models.update_rules import UpdateRule

from hobj.types import ImageId


# %%
class BinaryLearningModel(ABC):

    @abstractmethod
    def reset_state(self, seed: typing.Union[int, None]) -> None:
        """
        This function resets the LearningModel to some initial state.
        :param seed: if the LearningModel has any stochastic components, this seed should be used to seed them.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_response(
            self,
            image: ImageId,
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


# %%
class LinearLearner(BinaryLearningModel):
    def __init__(
            self,
            representational_model: RepresentationalModel,
            update_rule: UpdateRule,
    ):

        self.representational_model = representational_model
        self.update_rule = update_rule

        # State variables
        self.w = None
        self.b = None
        self._f_last = None
        self._logits_last = None
        self._action_last = None
        self._generator: np.random.Generator = np.random.default_rng()

        # Initialize state
        self.reset_state(seed=0)
        return

    def reset_state(self, seed: int) -> None:
        """
        :param seed:
        :return:
        """
        self.update_rule.reset()
        self.w = np.zeros((self.representational_model.d, 2))
        self.b = np.zeros((2,))
        self._f_last = None
        self._logits_last = None
        self._action_last = None
        self._generator = np.random.default_rng(seed=seed)

        return

    def get_response(
            self,
            image: ImageId,
    ) -> typing.Literal[0, 1]:

        f = self.representational_model.get_features(image=image)
        logits = f @ self.w + self.b
        action = self._random_tiebreaking_argmax(logits[0], logits[1])

        # Update internal state with traces
        self._f_last = f
        self._logits_last = logits
        self._action_last = action
        return action

    def give_feedback(self, reward: float) -> None:
        delta_w, delta_b = self.update_rule.get_update(x=self._f_last, w=self.w, b=self.b, logits=self._logits_last, action=self._action_last, reward=reward)  # [action]
        self.w += delta_w
        self.b += delta_b
        return

    def _random_tiebreaking_argmax(self, logit0, logit1) -> typing.Literal[0, 1]:
        if logit0 > logit1:
            return 0
        elif logit0 < logit1:
            return 1
        else:
            return 0 if self._generator.random() < 0.5 else 1
