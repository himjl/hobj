from dataclasses import dataclass
from typing import List, Set, Union, Dict, Optional

import numpy as np
import pydantic

from hobj.data.schema import ImageRef
from hobj.learning_models import BinaryLearningModel
import collections


# %%
class GeneralizationSessionResult(pydantic.BaseModel):
    transformation_to_kn: Dict[str, List[int]]
    kcatch: int
    ncatch: int

    @pydantic.model_validator(mode='after')
    def validate_results(self) -> 'GeneralizationSessionResult':
        for transformation in self.transformation_to_kn:
            k, n = self.transformation_to_kn[transformation]
            if k < 0 or n < 0 or k > n:
                raise ValueError(f"Got invalid Binomial parameters k={k}, n={n} for transformation {transformation}")

        if self.kcatch < 0 or self.ncatch < 0 or self.kcatch > self.ncatch:
            raise ValueError(f"Got invalid Binomial parameters k={self.kcatch}, n={self.ncatch} for catch trials")

        return self


class GeneralizationSubtask(pydantic.BaseModel):
    model_config = dict(
        frozen=True
    )

    support_imageA: ImageRef
    support_imageB: ImageRef
    test_imagesA: List[ImageRef]
    test_imagesB: List[ImageRef]
    image_ref_to_transformation: Dict[ImageRef, str]

    support_trials: Set[int]  # Trials in which a support image will be given
    catch_trials: Set[int]  # Trials in which a support image will be given, and recorded for analysis
    ntrials: int = pydantic.Field(description='The number of trials in the subtask.', gt=0)
    replace_test_trials: bool

    @pydantic.field_validator('test_imagesA', 'test_imagesB', mode='after')
    @classmethod
    def sort_image_refs(cls, value: List[ImageRef]) -> List[ImageRef]:
        return sorted(value)

    @pydantic.model_validator(mode='after')
    def validate_model(self) -> 'GeneralizationSubtask':

        if len(self.support_trials.intersection(self.catch_trials)) > 0:
            raise ValueError(f"Expected support_trials and catch_trials to be disjoint, but got {self.support_trials.intersection(self.catch_trials)}")

        if max(self.support_trials) >= self.ntrials:
            raise ValueError(f"Expected support_trials to be less than ntrials, but got {max(self.support_trials)} >= {self.ntrials}")
        if min(self.support_trials) < 0:
            raise ValueError(f"Expected support_trials to be non-negative, but got {min(self.support_trials)}")

        if not self.replace_test_trials and len(self.test_imagesA) + len(self.test_imagesB) < self.ntrials:
            raise ValueError(f"Specified replace=True, but only {len(self.test_imagesA) + len(self.test_imagesB)} images available. (len(classA)={len(self.classA)}; len(classB)={len(self.classB)}")

        for ref in self.test_imagesA + self.test_imagesB:
            if not ref in self.image_ref_to_transformation:
                raise ValueError(f"Expected all test images to have an associated transformation, but {ref} was missing.")

        return self

    def simulate_session(
            self,
            learner: BinaryLearningModel,
            seed: Union[int, None],
    ) -> GeneralizationSessionResult:
        """
        Convenience method to simulate a session of the one-shot task on a given BinaryLearningModel.
        :param learner:
        :param seed:
        :return: a [ntrials] boolean np.ndarray vector where True indicates a correct response.
        """
        # Allocate results
        transformation_to_kn = collections.defaultdict(lambda: [0, 0])
        kcatch = 0
        ncatch = 0

        # Initialize random state of environment
        gen = np.random.default_rng(seed=seed)

        # Sample the stimulus image sequence
        num_test_classA = len(self.test_imagesA)
        num_test_classB = len(self.test_imagesB)

        n_all_test_images = num_test_classA + num_test_classB
        ntest_trials = self.ntrials - len(self.support_trials)
        if self.replace_test_trials:
            i_image_seq = gen.integers(n_all_test_images, size=ntest_trials)
        else:
            i_image_seq = gen.permutation(n_all_test_images)[:ntest_trials]
        i_image_seq = list(i_image_seq)

        # Randomly sample the reward contingency
        classA_correct_action = 0 if gen.random() < 0.5 else 1
        classB_correct_action = 1 - classA_correct_action

        # Iterate over trials
        perf_seq = np.zeros(self.ntrials, dtype=bool)

        for i_trial in range(self.ntrials):
            # Retrieve trial information
            stimulus_category_is_A = gen.random() < 0.5
            correct_action_cur = classA_correct_action if stimulus_category_is_A else classB_correct_action

            if i_trial in self.support_trials or i_trial in self.catch_trials:
                # Sample a support image
                image_ref_cur = self.support_imageA if stimulus_category_is_A else self.support_imageB
            else:
                # Sample a test image
                i_test_image = i_image_seq.pop()
                image_ref_cur = self.test_imagesA[i_test_image] if stimulus_category_is_A else self.test_imagesA[i_test_image - num_test_classA]

            # Get response from learner
            a = learner.get_response(image=image_ref_cur)

            # Calculate feedback based on response
            correct = a == correct_action_cur
            feedback = 1. if correct else -1.

            # Deliver feedback to learner
            learner.give_feedback(feedback)

            # Record results of trial:

            if i_trial in self.support_trials or i_trial in self.catch_trials:
                kcatch += correct
                ncatch += 1
            else:
                transformation_cur = self.image_ref_to_transformation[image_ref_cur]
                transformation_to_kn[transformation_cur][0] += correct
                transformation_to_kn[transformation_cur][1] += 1

        return GeneralizationSessionResult(
            transformation_to_kn=transformation_to_kn,
            kcatch=kcatch,
            ncatch=ncatch
        )
