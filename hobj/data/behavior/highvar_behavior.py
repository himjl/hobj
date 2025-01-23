from typing import Dict, List, Tuple

import numpy as np

from hobj.data import images as imagesets, schema as schema
from hobj.data.behavior.template import LearningDataset

from dataclasses import dataclass


# %%
class MutatorHighVarHumanLearningCurves:
    """
    Class containing data from Experiment 1 from the paper, consisting of learning curves (100 trials) for 64 distinct binary subtasks.
    """

    dataset_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-highvar-human-learning-data.json'

    def __init__(self):
        super().__init__()

        self.imageset = imagesets.MutatorHighVarImageset()

        learning_dataset = LearningDataset.from_url(dataset_url = self.dataset_url)

        # Package data
        self._subtask_to_worker_to_perf_seq: Dict[str, Dict[str, np.ndarray]] = {}  # subtask: worker_id: perf_seq
        for session in learning_dataset.sessions:
            worker_id = session.worker_id
            reward_seq = session.reward_seq
            stimulus_seq = session.stimulus_sha256_seq
            assert len(reward_seq) == 104

            # Infer the subtask
            perf_seq = []
            all_categories = set()
            for i_trial in range(len(reward_seq)):

                # Filter out probe trials
                if i_trial in {25, 51, 77, 103}:
                    continue

                annotation = self.imageset.get_annotation(sha256=stimulus_seq[i_trial]).category
                all_categories.add(annotation)
                perf_seq.append(reward_seq[i_trial] > 0)

            assert len(all_categories) == 2
            assert len(perf_seq) == 100, len(perf_seq)
            subtask = ','.join(sorted(all_categories))

            if subtask not in self._subtask_to_worker_to_perf_seq:
                self._subtask_to_worker_to_perf_seq[subtask] = {}

            if worker_id not in self._subtask_to_worker_to_perf_seq[subtask]:
                self._subtask_to_worker_to_perf_seq[subtask][worker_id] = np.array(perf_seq)
            else:
                raise ValueError(f'Worker {worker_id} has already been added to subtask {subtask}')

        self._subtasks = sorted(list(self._subtask_to_worker_to_perf_seq.keys()))
        assert len(self._subtasks) == 64, len(self._subtasks)

        # Get the images associated with each category
        category_to_image_refs = {}
        for ref in self.imageset.image_refs:
            category = self.imageset.get_annotation(sha256=ref.sha256).category
            if category not in category_to_image_refs:
                category_to_image_refs[category] = []
            category_to_image_refs[category].append(ref)

        # Get the images associated with each subtask
        self._subtask_to_images: Dict[str, Tuple[List[schema.ImageRef], List[schema.ImageRef]]] = {}

        for subtask in self.subtasks:
            cat0, cat1 = subtask.split(',')
            images0 = category_to_image_refs[cat0]
            images1 = category_to_image_refs[cat1]

            assert len(images0) == len(images1) == 100
            self._subtask_to_images[subtask] = (category_to_image_refs[cat0], category_to_image_refs[cat1])

    @property
    def subtasks(self) -> List[str]:
        """
        Lists the 64 subtasks.
        :return:
        """
        return self._subtasks

    def get_subtask_images(self, subtask: str) -> Tuple[List[schema.ImageRef], List[schema.ImageRef]]:
        """
        Gives the images which comprise each of the two categories associated with this subtask.
        :param subtask:
        :return:
        """
        return self._subtask_to_images[subtask]

    @property
    def data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns a nested dictionary of subtask : worker_id : [boolean performance across trials]
        :param subtask:
        :return:
        """
        return self._subtask_to_worker_to_perf_seq


# %%

if __name__ == '__main__':

    dataset = MutatorHighVarHumanLearningCurves()
    print(dataset)

    x = dataset.data

