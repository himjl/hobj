from pathlib import Path

import pandas as pd

__all__ = ['load_highvar_behavior', 'load_oneshot_behavior']


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
        'trial_type',
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
        df = df.loc[df['trial_type'] != 'probe'].copy()
        df = df.sort_values(['assignment_id', 'trial']).copy()
        df['trial'] = df.groupby('assignment_id').cumcount()

    return df


def load_oneshot_behavior(
        cachedir: Path | None = None,
        redownload: bool = False,
) -> pd.DataFrame:
    """
    Load the trial-level human learning data from Experiment 2 of Lee and DiCarlo 2023.
    :return:
    """
    if redownload:
        raise ValueError("load_oneshot_behavior no longer supports redownload; expected cached CSV artifact instead.")

    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (cachedir if cachedir is not None else repo_root / 'data').resolve()
    dataset_path = cache_root / 'behavior' / 'human-behavior-oneshot-tasks.csv'
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Expected cached oneshot behavior CSV to already exist at:\n"
            f"  - {dataset_path}"
        )

    df = pd.read_csv(dataset_path)

    required_columns = {
        'trial',
        'session_id',
        'assignment_id',
        'slot',
        'worker_id',
        'subtask',
        'stimulus_id',
        'trial_type',
        'stimulus_duration_msec',
        'reaction_time_msec',
        'timed_out',
        'perf',
        'timestamp_start',
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Oneshot behavior CSV missing required columns: {sorted(missing_columns)}")

    return df


if __name__ == '__main__':
    df = load_highvar_behavior()
    import matplotlib.pyplot as plt
    lc = df.groupby('trial')['perf'].mean()
    plt.plot(lc.index.values, lc)
    plt.show()
