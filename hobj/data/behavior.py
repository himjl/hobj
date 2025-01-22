from abc import ABC
from abc import ABC
from typing import List

import pydantic

import hobj.data.schema as schema
from hobj.data.store import DataStore, default_data_store


class LearningSessionArray(pydantic.BaseModel):
    sessions: List[schema.HumanLearningSession]


class LearningDataset(ABC):
    dataset_url: str

    def __init__(
            self,
            data_store: DataStore = None,
            redownload=False,
    ):
        if data_store is None:
            self.data_store = default_data_store

        # Download the data
        json_data = self.data_store.get_json_data_from_url(
            url = self.dataset_url,
            redownload=redownload
        )

        self._learning_sessions = LearningSessionArray(**json_data)

    @property
    def learning_sessions(self) -> List[schema.HumanLearningSession]:
        return self._learning_sessions.sessions


class MutatorHighVarDataset(LearningDataset):
    dataset_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-highvar-human-learning-data.json'


class MutatorOneShotDataset(LearningDataset):
    dataset_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-oneshot-human-learning-data.json'


if __name__ == '__main__':
    dataset = MutatorHighVarDataset()
    oneshot_dataset = MutatorOneShotDataset(redownload=True)
    #print(dataset.learning_sessions)