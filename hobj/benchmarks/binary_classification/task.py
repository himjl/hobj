from typing import List, Optional

import numpy as np
import pydantic

from hobj.data.schema import ImageRef
from hobj.learning_models import BinaryLearningModel


# %%
class BinaryClassificationSubtask(pydantic.BaseModel):
    """
    A representation of a simple binary classification task which samples uniformly (with or without replacement) from a pool of images.
    Feedback consists of +1 or -1 rewards, depending on the class of the image.
    """

    class Config:
        frozen = True

    classA: List[ImageRef]
    classB: List[ImageRef]
    ntrials: int = pydantic.Field(description='The number of trials in the subtask.', gt=0)
    replace: bool = pydantic.Field(description='Whether to show stimulus images with replacement or not.')
    name: Optional[str] = pydantic.Field(description='A human-readable name for the subtask.', default=None)

    @pydantic.field_validator('classA', 'classB', mode='after')
    @classmethod
    def sort_image_refs(cls, value: List[ImageRef]) -> List[ImageRef]:
        return sorted(value)

    @pydantic.model_validator(mode='after')
    def validate_image_refs(self) -> 'BinaryClassificationSubtask':
        if self.replace and len(self.classA) + len(self.classB) < self.ntrials:
            raise ValueError(f"Specified replace=True, but only {len(self.classA) + len(self.classB)} images available. (len(classA)={len(self.classA)}; len(classB)={len(self.classB)}")
        return self

    def simulate_session(
            self,
            learner: BinaryLearningModel,
            seed: int,
    ) -> 'BinaryClassificationSubtaskResult':
        """
        Convenience method to simulate a session of the binary classification task on a given BinaryLearningModel.
        :param learner:
        :param seed:
        :return:
        """

        perf_seq: List[bool] = []
        stimulus_seq: List[ImageRef] = []

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

        # Iterate over trials
        for i_image in i_image_seq:
            # Retrieve trial information
            image_ref_cur = self.classA[i_image] if i_image < num_classA else self.classB[i_image - num_classA]
            correct_action_cur = 0 if i_image < num_classA else 1

            # Get response from learner
            a = learner.get_response(image=image_ref_cur)

            # Calculate feedback based on response
            feedback = 1. if a == correct_action_cur else -1.

            # Deliver feedback to learner
            learner.give_feedback(feedback)

            # Record results of trial
            perf_seq.append(feedback > 0)
            stimulus_seq.append(image_ref_cur)

        return BinaryClassificationSubtaskResult(
            subtask=self,
            perf_seq=perf_seq,
        )


# %%
class BinaryClassificationSubtaskResult(pydantic.BaseModel):
    """
    A class which represents the results of performing a binary classification subtask.
    """

    subtask: BinaryClassificationSubtask

    perf_seq: List[bool] = pydantic.Field(
        description='The performance (correct or incorrect) of the learner on each trial.'
    )

    @pydantic.model_validator(mode='after')
    def validate_perf(self) -> 'BinaryClassificationSubtaskResult':
        if not len(self.perf_seq) == self.subtask.ntrials:
            raise ValueError(f'Performance sequence length does not match number of trials. Got: {len(self.perf_seq)}')
        return self
