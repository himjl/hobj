import datetime
from abc import ABC
from typing import List, Literal

import pydantic

from hobj.data.store import DataStore, default_data_store


# %%
class HumanLearningSession(pydantic.BaseModel):

    worker_id: str = pydantic.Field(
        description='The anonymized worker ID of the participant.'
    )

    stimulus_sha256_seq: List[str] = pydantic.Field(
        description='The SHA256 hash of the image (in np.uint8 format and RGBA mode) shown to the participant.',
    )

    stimulus_duration_msec_seq: List[int] = pydantic.Field(
        description='The duration of the stimulus presentation in milliseconds.',
    )

    action_seq: List[Literal[0, 1, None]] = pydantic.Field(description='The action taken by the participant. 0 maps to "F" (left) and 1 maps to "J" (right). None maps to no action taken.')
    reward_seq: List[Literal[-1, 1]] = pydantic.Field(description='The reward received by the participant following their action.')
    timestamp_start_seq: List[datetime.datetime] = pydantic.Field(description='The timestamp of the start of the trial.')

    @pydantic.field_serializer('timestamp_start_seq')
    def serialize_timestamp_start(self, v: List[datetime.datetime], _info) -> List[float]:
        # Serializes datetime objects into Unix timestamps (seconds)
        return [entry.timestamp() for entry in v]

    @pydantic.model_validator(mode='after')
    def validate_lengths(self) -> 'HumanLearningSession':

        stimulus_len = len(self.stimulus_sha256_seq)
        duration_len = len(self.stimulus_duration_msec_seq)
        action_len = len(self.action_seq)
        reward_len = len(self.reward_seq)
        timestamp_len = len(self.timestamp_start_seq)

        if not stimulus_len == duration_len == action_len == reward_len == timestamp_len:
            raise ValueError(f'All lists must be of the same length. Got: {stimulus_len, duration_len, action_len, reward_len, timestamp_len}')

        return self


class LearningDataset(ABC):
    """
    Abstract class for a set of "raw" human learning sessions, downloaded from dataset_url.
    """
    dataset_url: str

    def __init__(
            self,
            data_store: DataStore = None,
            redownload=False,
    ):
        if data_store is None:
            self.data_store = default_data_store

        # Download the data:
        json_data = self.data_store.get_json_data_from_url(
            url=self.dataset_url,
            redownload=redownload
        )

        # Validate the data:
        class LearningSessionArray(pydantic.BaseModel):
            sessions: List[HumanLearningSession]
        self._learning_sessions = LearningSessionArray(**json_data)

        all_worker_ids = set()
        for session in self._learning_sessions.sessions:
            all_worker_ids.add(session.worker_id)

        self._worker_ids = list(all_worker_ids)

    @property
    def learning_sessions(self) -> List[HumanLearningSession]:
        """
        Returns a list of "raw" learning sessions.
        :return:
        """
        return self._learning_sessions.sessions

    @property
    def worker_ids(self) -> List[str]:
        return self._worker_ids

    def __str__(self):
        return f'{self.__class__.__name__}(sessions={len(self.learning_sessions)}, workers={len(self.worker_ids)})'

    def __repr__(self):
        return str(self)




# %%
class MutatorOneShotDataset(LearningDataset):
    dataset_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-oneshot-human-learning-data.json'

# %%

if __name__ == '__main__':

    oneshot_dataset = MutatorOneShotDataset()
    print(oneshot_dataset)
