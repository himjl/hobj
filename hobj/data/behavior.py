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
def load_highvar_behavior(redownload: bool = False) -> List[HumanLearningSession]:
    """
    Load the "raw" human learning data from Experiment 1 of Lee and DiCarlo 2023.
    :return:
    """

    sessions = _load_learning_sessions(
        dataset_url='https://hlbdatasets.s3.us-east-1.amazonaws.com/behavior/mutator-highvar-human-learning-data.json',
        redownload=redownload
    )

    return sessions


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
