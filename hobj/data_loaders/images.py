"""Load packaged image manifests as pandas DataFrames."""

from pathlib import Path

import pandas as pd


def _load_image_manifest(
        *,
        dataset_name: str,
        required_columns: set[str],
        cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load a packaged image manifest and attach absolute image paths.

    Args:
        dataset_name: Name of the dataset directory under ``data/images``.
        required_columns: Columns that must exist in the manifest.
        cachedir: Optional root directory containing the packaged ``data`` tree.
    Returns:
        A copy of the manifest as a DataFrame with an added ``image_path`` column.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the packaged manifest or images are missing.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (cachedir if cachedir is not None else repo_root / 'data').resolve()
    manifest_path = cache_root / f'meta-{dataset_name}.csv'

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected cached image manifest to already exist at: {manifest_path}"
        )

    manifest_df = pd.read_csv(manifest_path)
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} manifest.csv missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest_df = manifest_df.copy()
    missing_paths = manifest_df.loc[~manifest_df['relpath'].map(lambda p: Path.exists(cache_root / p)), 'relpath']
    if not missing_paths.empty:
        raise FileNotFoundError(
            "Expected packaged images to already exist under:\n"
            f"First missing path: {missing_paths.iloc[0]}"
        )

    return manifest_df


def load_mutator_highvar_images(
        cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the high-variance image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name='MutatorHighVarImageset',
        required_columns={'image_id', 'category', 'sha256', 'relpath'},
        cachedir=cachedir,
    )
    return manifest_df.sort_values('image_id').reset_index(drop=True)


def load_mutator_oneshot_images(
        cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the one-shot image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name='MutatorOneShotImageset',
        required_columns={
            'image_id',
            'category',
            'transformation',
            'transformation_level',
            'base_image_id',
            'sha256',
            'relpath',
        },
        cachedir=cachedir,
    )
    return manifest_df.sort_values('image_id').reset_index(drop=True)


def load_mutator_warmup_images(
        cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the warmup image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name='MutatorWarmupImageset',
        required_columns={'image_id', 'category', 'sha256', 'relpath'},
        cachedir=cachedir,
    )
    return manifest_df.sort_values('image_id').reset_index(drop=True)


def load_probe_images(
        cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the probe image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name='CatchImageset',
        required_columns={'image_id', 'sha256', 'relpath'},
        cachedir=cachedir,
    )
    return manifest_df.sort_values('image_id').reset_index(drop=True)

if __name__ == '__main__':
    df = load_mutator_oneshot_images()
