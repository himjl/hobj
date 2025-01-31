from dataclasses import dataclass
from typing import List, Set, Union, Dict, Optional

import numpy as np
import pydantic

from mref import ImageRef
from hobj.learning_models import BinaryLearningModel
import collections


# %%
class GeneralizationSessionResult(pydantic.BaseModel):
    transformation_to_kn: Dict[str, List[int]]
    kcatch: int
    ncatch: int
    worker_id: Optional[str] = pydantic.Field(default="NO_WORKER")

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
    """
    A simulator for a single session of the specific "one-shot" task used in Lee and DiCarlo 2023.
    """

    model_config = dict(
        frozen=True
    )

    support_imageA: ImageRef
    support_imageB: ImageRef
    test_imagesA: List[ImageRef]
    test_imagesB: List[ImageRef]
    image_ref_to_transformation: Dict[ImageRef, str]

    @pydantic.field_validator('test_imagesA', 'test_imagesB', mode='after')
    @classmethod
    def sort_image_refs(cls, value: List[ImageRef]) -> List[ImageRef]:
        return sorted(value)

    @pydantic.model_validator(mode='after')
    def validate_model(self) -> 'GeneralizationSubtask':

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

        # Initialize random state of environment
        gen = np.random.default_rng(seed=seed)

        # Allocate results
        transformation_to_kn = collections.defaultdict(lambda: [0, 0])
        kcatch = 0
        ncatch = 0

        # Sample the sequence of trials using the procedure described in Lee and DiCarlo 2023
        support_trials = set(range(10))
        catch_trials = {9, 14, 19}
        ntrials = 20
        stimulus_category_is_A_seq = []

        # An initial phase of 10 support trials, with classes sampled in equal number (5 each)
        support_sequence = [True] * 5 + [False] * 5
        support_sequence = [support_sequence[i] for i in gen.permutation(len(support_sequence))]
        stimulus_category_is_A_seq.extend(support_sequence)

        # A testing phase of 8 test trials with 2 catch trials, where classes are sampled i.i.d.
        # with 50-50 probability, and test images for each class are sampled without replacement
        test_sequence = [bool(v) for v in gen.random(size=10) < 0.5]
        stimulus_category_is_A_seq.extend(test_sequence)
        assert len(stimulus_category_is_A_seq) == ntrials

        # Randomly sample the reward contingency
        classA_correct_action = 0 if gen.random() < 0.5 else 1
        classB_correct_action = 1 - classA_correct_action

        i_test_seq_A = list(gen.permutation(len(self.test_imagesA)))
        i_test_seq_B = list(gen.permutation(len(self.test_imagesB)))

        # Iterate over trials
        for i_trial in range(ntrials):
            # Retrieve trial information
            stimulus_category_is_A = stimulus_category_is_A_seq[i_trial]
            correct_action_cur = classA_correct_action if stimulus_category_is_A else classB_correct_action

            if i_trial in support_trials or i_trial in catch_trials:
                # Supply a support image
                image_ref_cur = self.support_imageA if stimulus_category_is_A else self.support_imageB

            else:
                # Sample a test image without replacement
                if stimulus_category_is_A:
                    image_ref_cur = self.test_imagesA[i_test_seq_A.pop()]

                else:
                    image_ref_cur = self.test_imagesB[i_test_seq_B.pop()]

            # Get response from learner
            a = learner.get_response(image=image_ref_cur)

            # Calculate feedback based on response
            correct = a == correct_action_cur
            feedback = 1. if correct else -1.

            # Deliver feedback to learner
            learner.give_feedback(feedback)

            # Record results of trial if catch or test:
            if i_trial in catch_trials:
                kcatch += correct
                ncatch += 1
            elif i_trial in support_trials:
                pass
            else:
                transformation_cur = self.image_ref_to_transformation[image_ref_cur]
                transformation_to_kn[transformation_cur][0] += correct
                transformation_to_kn[transformation_cur][1] += 1

        return GeneralizationSessionResult(
            transformation_to_kn=transformation_to_kn,
            kcatch=kcatch,
            ncatch=ncatch
        )
