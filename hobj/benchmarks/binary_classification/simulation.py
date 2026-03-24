from typing import List, Optional, Union

import numpy as np
import pydantic
from mref import ImageRef

from hobj.learning_models import BinaryLearningModel


# %%
class BinaryClassificationSubtaskResult(pydantic.BaseModel):
    model_config = dict(
        arbitrary_types_allowed=True
    )

    perf_seq: np.ndarray
    worker_id: Optional[str] = pydantic.Field(default="NO_WORKER")


class BinaryClassificationSubtask(pydantic.BaseModel):
    """
    A representation of a simple binary classification task which samples uniformly (with or without replacement) from a pool of images.
    Feedback consists of +1 or -1 rewards, depending on the class of the image.
    """

    model_config = dict(
        frozen=True
    )

    classA: List[ImageRef]
    classB: List[ImageRef]
    ntrials: int = pydantic.Field(description='The number of trials in the subtask.', gt=0)
    replace: bool = pydantic.Field(description='Whether to show stimulus images with replacement or not.')

    @pydantic.field_validator('classA', 'classB', mode='after')
    @classmethod
    def sort_image_refs(cls, value: List[ImageRef]) -> List[ImageRef]:
        return sorted(value)

    @pydantic.model_validator(mode='after')
    def validate_image_refs(self) -> 'BinaryClassificationSubtask':
        if not self.replace and len(self.classA) + len(self.classB) < self.ntrials:
            raise ValueError(f"Specified replace=False, but only {len(self.classA) + len(self.classB)} images available. (len(classA)={len(self.classA)}; len(classB)={len(self.classB)}")
        return self

    def simulate_session(
            self,
            learner: BinaryLearningModel,
            seed: Union[int, None],
    ) -> BinaryClassificationSubtaskResult:
        """
        Convenience method to simulate a session of the binary classification task on a given BinaryLearningModel.
        :param learner:
        :param seed:
        :return: a [ntrials] boolean np.ndarray vector where True indicates a correct response.
        """

        perf_seq = np.zeros(self.ntrials, dtype=bool)

        # Initialize random state of environment
        gen = np.random.default_rng(seed=seed)

        # Sample the stimulus image sequence
        num_classA = len(self.classA)
        num_classB = len(self.classB)
        n_all_images = num_classA + num_classB
        if self.replace:
            i_image_seq = gen.integers(n_all_images, size=self.ntrials)
        else:
            i_image_seq = gen.permutation(n_all_images)[:self.ntrials]

        # Randomly sample the reward contingency
        classA_correct_action = 0 if gen.random() < 0.5 else 1
        classB_correct_action = 1 - classA_correct_action

        # Iterate over trials
        i_trial = 0
        for i_image in i_image_seq:
            # Retrieve trial information
            image_ref_cur = self.classA[i_image] if i_image < num_classA else self.classB[i_image - num_classA]
            correct_action_cur = classA_correct_action if i_image < num_classA else classB_correct_action

            # Get response from learner
            a = learner.get_response(image=image_ref_cur)

            # Calculate feedback based on response
            feedback = 1. if a == correct_action_cur else -1.

            # Deliver feedback to learner
            learner.give_feedback(feedback)

            # Record results of trial
            perf_seq[i_trial] = feedback > 0

            i_trial += 1

        return BinaryClassificationSubtaskResult(
            perf_seq=perf_seq,
        )
