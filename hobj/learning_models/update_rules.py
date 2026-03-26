from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import scipy.special


class UpdateRule(ABC):
    def __init__(self, alpha: float):
        """
        Implements an update rule for a linear learner. Assumes that all features have at most norm 1.
        :param alpha: the normalized learning rate. Between 0 and 1.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in the interval [0, 1]")

        self.alpha = alpha
        self.reset()
        return

    def reset(self) -> None:
        """
        Called to reset the update rule to some initial state. Not used by all update rules.
        :return:
        """
        return

    @abstractmethod
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        """
        Returns delta_w and delta_b, where
        w{t+1} = w{t} + delta_w
        b{t+1} = b{t} + delta_b

        :param x: np.ndarray, shape=(d)
        :param w: np.ndarray, shape=(d, actions)
        :param b: np.ndarray, shape=(actions,)
        :param logits: np.ndarray, shape=(action,)
        :param action: int, action taken by the learner
        :param reward: float, reward received by the learner
        :return: delta_w, delta_b, which are np.ndarrays of the same shape as w and b, respectively.
        """
        delta_w = np.zeros(w.shape)
        delta_b = np.zeros(b.shape)
        return delta_w, delta_b


# %%
class Square(UpdateRule):
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        delta = np.zeros(w.shape)

        reward_prediction = logits[action]
        rpe = float(reward) - reward_prediction

        delta[:, action] = self.alpha * rpe * x
        return delta, 0


class Perceptron(UpdateRule):
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        nactions = len(logits)
        gts = np.zeros(nactions)

        yhat = logits[action]
        z = reward * yhat
        if z <= 0:
            gts[action] = self.alpha * reward
        delta = x[:, None] * gts[None, :]
        return delta, 0


class Hinge(UpdateRule):
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        nactions = len(logits)
        gts = np.zeros(nactions)

        yhat = logits[action]
        z = reward * yhat
        if z <= 1:  # and yhat >= 0:
            gts[action] = self.alpha * reward
        delta = x[:, None] * gts[None, :]
        return delta, 0


class MAE(UpdateRule):
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        nactions = len(logits)
        reward_prediction = logits[action]

        gts = np.zeros(nactions)
        if reward_prediction < reward:
            gts[action] = self.alpha
        elif reward_prediction > reward:
            gts[action] = -self.alpha

        delta = x[:, None] * gts[None, :]
        return delta, 0


class Exponential(UpdateRule):
    max_weight_norm = 10.0

    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        nactions = len(logits)

        gts = np.zeros(nactions)
        gts[action] = self.alpha * reward * np.exp(-reward * logits[action])
        delta = x[:, None] * gts[None, :]

        # Bound norm of weights
        wnext = w + delta
        if np.linalg.norm(wnext) > self.max_weight_norm:
            wnext = wnext / np.linalg.norm(wnext) * self.max_weight_norm
            delta = wnext - w
        return delta, 0


class CE(UpdateRule):
    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: np.ndarray,
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        nactions = len(logits)
        gts = np.zeros(nactions)
        reward_prediction = logits[action]

        gts[action] = (
            4 * self.alpha * (reward / (1 + np.exp(reward * reward_prediction)))
        )

        delta = x[:, None] * gts[None, :]
        return delta, 0


class REINFORCE(UpdateRule):
    @staticmethod
    def _compute_trace(
        action_taken: int,
        logits: np.ndarray,
    ):
        """
        Computes the negative derivative of the probability of taking each action with respect to logit[i]
        exps = np.exp(XW)
        denominator = exps.sum()
        probs = exps / denominator
        expression = -probs
        expression[action_taken] = 1 + expression[action_taken]
        return expression
        """

        log_numerator = logits
        log_denominator = scipy.special.logsumexp(logits)
        log_expression = log_numerator - log_denominator
        """
        expression[i] = -exp(XW[i]) / (np.sum(np.exp(XW)) for i!=action_taken
        expression[action_taken] = 1 - -exp(XW[action_taken]) / (np.sum(np.exp(XW))
        """
        expression = np.exp(log_expression)
        expression = -expression
        expression[action_taken] = 1 + expression[action_taken]
        return expression

    def get_update(
        self,
        x: np.ndarray,
        w: np.ndarray,
        b: Union[np.ndarray, np.generic, float],
        logits: np.ndarray,
        action: int,
        reward: float,
    ) -> Tuple[np.ndarray, Union[np.ndarray, np.generic, float]]:
        gt = self.alpha * reward * self._compute_trace(action, logits)

        delta = x[:, None] * gt[None, :]
        return delta, 0
