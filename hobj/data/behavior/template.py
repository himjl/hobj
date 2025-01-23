import datetime
from abc import ABC
from typing import List, Literal

import pydantic

from hobj.data.store import DataStore, default_data_store


# %%
class HumanLearningSession(pydantic.BaseModel):

    class Config:
        frozen = True

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

# %%
class LearningDataset(pydantic.BaseModel):
    """
    Model which wraps a set of "raw" human learning sessions.

    A .from_url method is provided to instantiate it from a URL.
    """

    class Config:
        frozen = True

    sessions: List[HumanLearningSession]

    @classmethod
    def from_url(
            cls,
            dataset_url: str,
            redownload:bool = False,
            data_store: DataStore = None,
    ) -> 'LearningDataset':
        """
        Instantiates a LearningDataset object from a given URL.
        :param url:
        :param redownload:
        :return:
        """
        if data_store is None:
            data_store = default_data_store

        # Download the data:
        json_data = data_store.get_json_data_from_url(
            url=dataset_url,
            redownload=redownload
        )

        return cls(
            **json_data
        )

    def __str__(self):
        unique_workers = set()
        ntrials =0
        for session in self.sessions:
            unique_workers.add(session.worker_id)
            ntrials += len(session.action_seq)
        nworkers = len(unique_workers)
        return f'{self.__class__.__name__}(sessions={len(self.sessions)}, workers={nworkers}, trials={ntrials})'

    def __repr__(self):
        return self.__str__()
