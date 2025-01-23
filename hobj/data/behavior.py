from typing import List

import pydantic

from hobj.data.schema import HumanLearningSession
from hobj.data.store import DataStore, default_data_store

__all__ = ['load_highvar_behavior', 'load_oneshot_behavior']


def _load_learning_sessions(
        dataset_url: str,
        redownload: bool
) -> List[HumanLearningSession]:
    data_store = default_data_store
    # Download the data:
    json_data = data_store.get_json_data_from_url(
        url=dataset_url,
        redownload=redownload
    )

    #
    class LearningSessionDataset(pydantic.BaseModel):
        sessions: List[HumanLearningSession]

    learning_dataset = LearningSessionDataset(**json_data)
    return learning_dataset.sessions


# %% Data loaders
def load_highvar_behavior(
        redownload: bool = False,
        remove_probe_trials: bool = True
) -> List[HumanLearningSession]:
    """
    Load the "raw" human learning data from Experiment 1 of Lee and DiCarlo 2023.
    :return:
    """

    sessions = _load_learning_sessions(
        dataset_url='https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-highvar-human-learning-data.json',
        redownload=redownload
    )

    if not remove_probe_trials:
        return sessions


    def filter(vals: list):
        probe_trials = {25, 51, 77, 103}
        return [vals[i] for i in range(len(vals)) if i not in probe_trials]

    filtered_sessions = []
    for session in sessions:
        # Filter out indices in probe_trials
        filtered_session = HumanLearningSession(
            worker_id = session.worker_id,
            stimulus_sha256_seq = filter(session.stimulus_sha256_seq),
            stimulus_duration_msec_seq = filter(session.stimulus_duration_msec_seq),
            action_seq = filter(session.action_seq),
            reward_seq = filter(session.reward_seq),
            timestamp_start_seq = filter(session.timestamp_start_seq)
        )

        filtered_sessions.append(filtered_session)

    return filtered_sessions


def load_oneshot_behavior(redownload: bool = False) -> List[HumanLearningSession]:
    """
    Load the "raw" human learning data from Experiment 2 of Lee and DiCarlo 2023.
    :return:
    """

    sessions = _load_learning_sessions(
        dataset_url='https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-oneshot-human-learning-data.json',
        redownload=redownload
    )

    return sessions

if __name__ == '__main__':
    sessions=load_highvar_behavior()