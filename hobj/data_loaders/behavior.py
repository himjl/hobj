import datetime
import json
from pathlib import Path
from typing import List, Literal

import pandas as pd
import pydantic

from hobj.utils.file_io import download_json

__all__ = ['load_highvar_behavior', 'load_oneshot_behavior']


class HumanLearningSession(pydantic.BaseModel):
    """
    A representation of the "raw" data collected from a human learning session.
    """
    model_config = dict(
        frozen=True
    )

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
def _load_learning_sessions(
        dataset_url: str,
        cache_filename: str,
        cachedir: Path | None = None,
        redownload: bool = False,
) -> List[HumanLearningSession]:
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (cachedir if cachedir is not None else repo_root / 'data').resolve()
    behavior_dir = cache_root / 'behavior'
    behavior_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = behavior_dir / cache_filename

    if redownload or not dataset_path.exists():
        json_data = download_json(dataset_url)
        dataset_path.write_text(json.dumps(json_data, indent=2))
    else:
        json_data = json.loads(dataset_path.read_text())

    class LearningSessionDataset(pydantic.BaseModel):
        sessions: List[HumanLearningSession]

    learning_dataset = LearningSessionDataset(**json_data)
    return learning_dataset.sessions


# %% Data loaders
def load_highvar_behavior(
        remove_probe_trials: bool = True,
        cachedir: Path | None = None,
        redownload: bool = False,
) -> pd.DataFrame:
    """
    Load the trial-level human learning data from Experiment 1 of Lee and DiCarlo 2023.
    :return:
    """
    if redownload:
        raise ValueError("load_highvar_behavior no longer supports redownload; expected cached CSV artifact instead.")

    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (cachedir if cachedir is not None else repo_root / 'data').resolve()
    dataset_path = cache_root / 'behavior' / 'human-behavior-highvar-tasks.csv'
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected cached highvar behavior CSV to already exist at:\n"
            f"  - {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_columns = {
        'trial',
        'assignment_id',
        'worker_id',
        'subtask',
        'stimulus_id',
        'is_probe_trial',
        'stimulus_duration_msec',
        'reaction_time_msec',
        'timed_out',
        'perf',
        'timestamp_start',
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Highvar behavior CSV missing required columns: {sorted(missing_columns)}")

    if remove_probe_trials:
        df = df.loc[~df['is_probe_trial']].copy()

    return df


def load_oneshot_behavior(
        cachedir: Path | None = None,
        redownload: bool = False,
) -> List[HumanLearningSession]:
    """
    Load the "raw" human learning data from Experiment 2 of Lee and DiCarlo 2023.
    :return:
    """

    sessions = _load_learning_sessions(
        dataset_url='https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-oneshot-human-learning-data.json',
        cache_filename='mutator-oneshot-human-learning-data.json',
        cachedir=cachedir,
        redownload=redownload
    )

    return sessions


if __name__ == '__main__':
    sessions = load_highvar_behavior()
