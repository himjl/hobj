from pathlib import Path
from typing import Literal

import pandas as pd

from hobj.data.download import resolve_data_root


# %%
def load_human_behavior(
    experiment: Literal["highvar", "oneshot"] = "highvar",
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load trial-level human behavior for one of the packaged experiments.

    Args:
        experiment: Name of the experiment to load.
        cachedir: Optional directory containing the packaged ``data`` tree.

    Returns:
        The requested behavior dataset.
    """
    if experiment not in {"highvar", "oneshot"}:
        raise ValueError(
            f"Invalid experiment name: {experiment}. Provide either 'highvar' or 'oneshot'."
        )

    if experiment == "highvar":
        return load_highvar_behavior(cachedir=cachedir)
    return load_oneshot_behavior(cachedir=cachedir)


# %% Data loaders
def load_highvar_behavior(
    remove_probe_trials: bool = True,
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load trial-level human learning data from Experiment 1.

    Args:
        remove_probe_trials: Whether to exclude probe trials and renumber the
            remaining trial index within each assignment.
        cachedir: Optional root directory containing the packaged ``data`` tree.

    Returns:
        A DataFrame containing the high-variance behavior dataset with
        ``stimulus_id`` normalized to ``image_id``.
    """
    cache_root = resolve_data_root(cachedir=cachedir)
    dataset_path = cache_root / "behavior" / "human-behavior-highvar-subtasks.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected highvar behavior CSV to exist after resolving packaged data "
            f"at:\n  - {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_columns = {
        "trial",
        "assignment_id",
        "worker_id",
        "subtask",
        "image_id",
        "trial_type",
        "stimulus_duration_msec",
        "reaction_time_msec",
        "timed_out",
        "perf",
        "timestamp_start",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Highvar behavior CSV missing required columns: {sorted(missing_columns)}"
        )

    if remove_probe_trials:
        df = df.loc[df["trial_type"] != "probe"].copy()
        df = df.sort_values(["assignment_id", "trial"]).copy()
        df["trial"] = df.groupby("assignment_id").cumcount()

    return df


def load_oneshot_behavior(
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load trial-level human learning data from Experiment 2.

    Args:
        cachedir: Optional root directory containing the packaged ``data`` tree.

    Returns:
        A DataFrame containing the one-shot behavior dataset with
        ``stimulus_id`` normalized to ``image_id``.
    """
    cache_root = resolve_data_root(cachedir=cachedir)
    dataset_path = cache_root / "behavior" / "human-behavior-oneshot-subtasks.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected oneshot behavior CSV to exist after resolving packaged data "
            f"at:\n  - {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_columns = {
        "trial",
        "assignment_id",
        "slot",
        "worker_id",
        "subtask",
        "image_id",
        "trial_type",
        "stimulus_duration_msec",
        "reaction_time_msec",
        "timed_out",
        "perf",
        "timestamp_start",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Oneshot behavior CSV missing required columns: {sorted(missing_columns)}"
        )

    return df
