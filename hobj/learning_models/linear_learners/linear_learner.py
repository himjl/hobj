import hobj.learning_models.learning_model as lm
import os
import hobj.learning_models.linear_learners.representational_models.representational_model as rms
import hobj.learning_models.linear_learners.update_rules.update_rules as urs
import numpy as np


class LinearLearner(lm.LearningModel):
    """
    A learning model based on a standard cognitive theory of learning.
    """
    def __init__(
            self,
            representational_model: rms.RepresentationalModel,
            update_rule: urs.UpdateRule,
            nactions=2,
    ):
        learner_id = representational_model.representational_model_id + '+' + update_rule.update_rule_id
        super().__init__(learner_id)

        self.representational_model = representational_model
        self.update_rule = update_rule
        self.nactions = nactions
        self.w = np.zeros((self.representational_model.d, self.nactions))
        self.b = np.zeros((self.nactions,))
        self.reset()
        return

    def reset(self):
        self.representational_model.reset()
        self.update_rule.reset()
        self.w = np.zeros((self.representational_model.d, self.nactions))
        self.b = np.zeros((self.nactions,))
        return

    def respond(self, image_url: str):
        f = self.representational_model.get_features(image_url=image_url)
        self.f = f
        self.logits = self.f @ self.w + self.b # [action]
        self.action = int(_random_tiebreaking_argmax(self.logits))
        return self.action

    def learn(self, reward: float):
        delta_w, delta_b = self.update_rule.get_update(x=self.f, w=self.w, b=self.b, logits=self.logits, action=self.action, reward=reward)  # [action]
        self.w += delta_w
        self.b += delta_b
        return


def _random_tiebreaking_argmax(x):
    return np.random.choice(np.flatnonzero(x == x.max()))
