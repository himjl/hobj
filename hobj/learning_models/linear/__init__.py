import typing

import PIL.Image
import numpy as np

import mref.media_references
from hobj.learning_models import BinaryLearningModel
from hobj.learning_models.linear.representation import RepresentationalModel
from hobj.learning_models.linear.update_rules import UpdateRule


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
            image: typing.Union[mref.media_references.ImageRef, PIL.Image]
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
