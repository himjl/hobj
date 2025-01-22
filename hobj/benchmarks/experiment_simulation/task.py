from abc import ABC, abstractmethod
from typing import List, Literal, Union

import PIL.Image
import numpy as np
import pydantic

import hobj.data.schema as schema
from hobj.learning_models.learning_model import BinaryLearningModel


class BinaryTask(ABC):
    class RunSessionResult(pydantic.BaseModel):
        reward_sequence: List[float]
        action_sequence: List[Literal[0, 1]]
        image_sequence: List[schema.ImageRef]

    @abstractmethod
    def run_session(
            self,
            learner: BinaryLearningModel,
            seed: int
    ) -> RunSessionResult:
        """
        Runs a session of the task with the given learner. Note that the learner is not reset by this function; it runs the
        session using the learner "as given".
        :param learner:
        :param seed:
        :return:
        """
        raise NotImplementedError


class SimpleBinaryTask(BinaryTask):
    """
    A task which samples uniformly (with or without replacement) from a pool of images.
    Feedback consists of +1 or -1 rewards, depending on the class of the image.
    """

    def __init__(
            self,
            classA: List[Union[schema.ImageRef, PIL.Image]],
            classB: List[Union[schema.ImageRef, PIL.Image]],
            ntrials: int,
            replace: bool,
    ):

        classA = self._process_images(images=classA)
        classB = self._process_images(images=classB)
        nclassA = len(classA)
        nclassB = len(classB)

        if replace and nclassA + nclassB < ntrials:
            raise ValueError(f"Specified replace=True, but only {nclassA + nclassB} images available. (len(classA)={nclassA}; len(classB)={nclassB})")

        self.image_refs = classA + classB
        self.image_labels = [0] * nclassA + [1] * nclassB
        self.ntrials = ntrials
        self.replace = replace

    def run_session(
            self,
            learner: BinaryLearningModel,
            seed: int
    ) -> BinaryTask.RunSessionResult:

        reward_seq = []
        action_seq = []
        image_seq = []

        gen = np.random.default_rng(seed=seed)

        if self.replace:
            i_image_seq = gen.integers(len(self.image_refs), size=self.ntrials)
        else:
            i_image_seq = gen.permutation(len(self.image_refs))[:self.ntrials]

        # Iterate over trials
        for i_image in i_image_seq:
            # Get response from learner
            a = learner.get_response(image=self.image_refs[i_image])

            # Calculate feedback based on response
            feedback = 1. if a == self.image_labels[i_image] else -1.

            # Deliver feedback to learner
            learner.give_feedback(feedback)

            # Record results of trial
            reward_seq.append(feedback)
            action_seq.append(a)
            image_seq.append(self.image_refs[i_image])

        return BinaryTask.RunSessionResult(
            reward_sequence=reward_seq,
            action_sequence=action_seq,
            image_sequence=image_seq
        )

    def _process_images(self, images: List[Union[schema.ImageRef, PIL.Image]]) -> List[schema.ImageRef]:
        """
        Converts a list of images to a (sorted) list of unique ImageRefs.
        :param images:
        :return:
        """

        image_refs = [img if isinstance(img, schema.ImageRef) else schema.ImageRef.from_image(img) for img in images]
        image_refs = sorted(set(image_refs))
        return image_refs
