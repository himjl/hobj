import numpy as np
import scipy.special

class UpdateRule(object):
    def __init__(self, alpha:float):
        """
        Implements an update rule for a linear learner. Assumes that all features have at most norm 1.
        :param alpha: the normalized learning rate. Between 0 and 1.
        """
        assert 0 <= alpha <= 1
        self.alpha = alpha
        self._update_rule_id = None
        self.reset()
        return

    def reset(self):
        """
        Resets the update rule to its initial state. Not used by all update rules.
        :return:
        """
        return

    @property
    def update_rule_id(self):
        return str(self.__class__.__name__) + '_%0.3e'%(self.alpha)

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
        delta_w = np.zeros(w.shape)
        delta_b = np.zeros(b.shape)
        return delta_w, delta_b


class Prototype(UpdateRule):

    """
    Simulates the decision boundary implemented by a prototype learner.
    """
    def reset(self):
        self.ncounts = None

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
        if self.ncounts is None:
            self.ncounts = np.zeros(w.shape[1])


        if reward > 0:
            # Received a positive example of the class associated with the action
            i_updated_class = action
        else:
            # Unknown which class this is associated with. In the two-way case, it could be inferred, but in general, not.
            if w.shape[1] == 2:
                i_updated_class = 1 - action
            else:
                return 0, 0

        mu_cur = w[:, i_updated_class]
        n = self.ncounts[i_updated_class]
        mu_next = (n / (n+1)) * mu_cur + (1 / (n+1)) * x
        norm_next = np.square(np.linalg.norm(mu_next))

        delta_w = np.zeros(w.shape)
        delta_w[:, i_updated_class] = mu_next - mu_cur

        delta_b = np.zeros(b.shape)
        delta_b[i_updated_class] = (-norm_next) - b[i_updated_class]

        return delta_w, delta_b

class Square(UpdateRule):

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
        delta = np.zeros(w.shape)

        reward_prediction = logits[action]
        rpe = float(reward) - reward_prediction

        delta[:, action] = (self.alpha) * rpe * x
        return delta, 0


class Perceptron(UpdateRule):

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """

        nactions = len(logits)
        gts = np.zeros(nactions)

        yhat = logits[action]
        z = reward * yhat
        if z <= 0 :
            gts[action] = self.alpha * reward
        delta = x[:, None] * gts[None, :]
        return delta, 0


class Hinge(UpdateRule):

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """

        nactions = len(logits)
        gts = np.zeros(nactions)

        yhat = logits[action]
        z = reward * yhat
        if z <= 1:# and yhat >= 0:
            gts[action] = self.alpha * reward
        delta = x[:, None] * gts[None, :]
        return delta, 0


class MAE(UpdateRule):

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """

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

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
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

    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
        nactions = len(logits)
        gts = np.zeros(nactions)
        reward_prediction = logits[action]

        gts[action] = 4 * self.alpha * (reward / (1 + np.exp(reward * reward_prediction)))

        delta = x[:, None] * gts[None, :]
        return delta, 0


class REINFORCE(UpdateRule):

    def _compute_trace(
            self,
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


    def get_update(self, x:np.ndarray, w: np.ndarray, b:np.ndarray, logits: np.ndarray, action: int, reward: float):
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
        :return: delta, np.ndarray, shape=(actions)
        """
        gt = self.alpha * reward * self._compute_trace(action, logits)

        delta = x[:, None] * gt[None, :]
        return delta, 0

