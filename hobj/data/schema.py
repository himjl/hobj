import datetime
from typing import List, Literal

import PIL.Image
import pydantic

from hobj.data.store import default_data_store
from hobj.utils.hash import hash_image


class _Common(pydantic.BaseModel):
    model_config = dict(
        frozen=True
    )


# %% Image data
class ImageRef(_Common):
    sha256: str = pydantic.Field(pattern=r'^[a-f0-9]{64}$')

    @classmethod
    def from_image(cls, image: PIL.Image, register: bool = True):
        if register:
            default_data_store.register_image(image_data=image)

        return cls(
            sha256=hash_image(image=image)
        )

    def get_image_data(self) -> PIL.Image:
        return default_data_store.load_image(sha256=self.sha256)

    def __lt__(self, other) -> bool:
        return self.sha256 < other.sha256

    def __hash__(self):
        return hash(self.sha256)


# %% Learning tasks


# %% Behavior
class HumanLearningSession(_Common):
    """
    A representation of the "raw" data collected from a human learning session.
    """

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
