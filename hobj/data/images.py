"""Load packaged image manifests and images."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from PIL import Image

from hobj.data.download import resolve_data_root
from hobj.types import ImageId


def _load_image_manifest(
    *,
    dataset_name: str,
    required_columns: set[str],
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load a packaged image manifest after validating packaged image files.

    Args:
        dataset_name: Name of the packaged manifest.
        required_columns: Columns that must exist in the manifest.
        cachedir: Optional root directory containing the packaged ``data`` tree.

    Returns:
        A copy of the manifest as a DataFrame.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the packaged manifest or images are missing.
    """
    cache_root = resolve_data_root(cachedir=cachedir)
    manifest_path = cache_root / f"meta-{dataset_name}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Expected image manifest to exist after resolving packaged data at: "
            f"{manifest_path}"
        )

    manifest_df = pd.read_csv(manifest_path)
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} manifest.csv missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest_df = manifest_df.copy()
    missing_paths = manifest_df.loc[
        ~manifest_df["relpath"].map(lambda p: (cache_root / p).exists()), "relpath"
    ]
    if not missing_paths.empty:
        raise FileNotFoundError(
            "Expected packaged images to exist after resolving packaged data "
            "under:\n"
            f"First missing path: {missing_paths.iloc[0]}"
        )

    return manifest_df


def load_imageset_meta_highvar(
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the high-variance image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name="MutatorHighVarImageset",
        required_columns={"image_id", "category", "sha256", "relpath"},
        cachedir=cachedir,
    )
    return manifest_df.sort_values("image_id").reset_index(drop=True)


def load_imageset_meta_oneshot(
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the one-shot image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name="MutatorOneShotImageset",
        required_columns={
            "image_id",
            "category",
            "transformation",
            "transformation_level",
            "base_image_id",
            "sha256",
            "relpath",
        },
        cachedir=cachedir,
    )
    return manifest_df.sort_values("image_id").reset_index(drop=True)


def load_imageset_meta_warmup(
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the warmup image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name="MutatorWarmupImageset",
        required_columns={"image_id", "category", "sha256", "relpath"},
        cachedir=cachedir,
    )
    return manifest_df.sort_values("image_id").reset_index(drop=True)


def load_imageset_meta_catch(
    cachedir: Path | None = None,
) -> pd.DataFrame:
    """Load the probe image manifest."""
    manifest_df = _load_image_manifest(
        dataset_name="CatchImageset",
        required_columns={"image_id", "sha256", "relpath"},
        cachedir=cachedir,
    )
    return manifest_df.sort_values("image_id").reset_index(drop=True)

# %% Image loaders:

@lru_cache(maxsize=None)
def _image_id_to_local_path_table(cache_root: Path) -> dict[ImageId, Path]:
    """Build a mapping from image ID to absolute local path.

    Args:
        cache_root: Directory containing the packaged image manifests.

    Returns:
        A mapping from image ID to absolute image path.
    """
    manifest_paths = sorted(cache_root.glob("meta-*.csv"))
    table = {}
    for manifest_path in manifest_paths:
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            image_id = row["image_id"]
            relpath = row["relpath"]
            abs_path = cache_root / relpath
            table[image_id] = abs_path
    return table


def list_image_ids(cachedir: Path | None = None) -> list[ImageId]:
    """Iterate over all packaged image IDs.

    Args:
        cachedir: Optional directory containing the packaged ``data`` tree.

    Yields:
        Image IDs from the packaged dataset.
    """
    cache_root = resolve_data_root(cachedir=cachedir)
    table = _image_id_to_local_path_table(cache_root)
    return sorted(table.keys())

def load_image(
        image_id: ImageId,
        cachedir: Path | None = None,
) -> Image.Image:
    """Load an image by ``image_id`` from the packaged dataset.

    Args:
        image_id: ID of the image to load.
        cachedir: Optional directory containing the packaged ``data`` tree.

    Returns:
        The requested image.
    """
    path = get_image_path(image_id=image_id, cachedir=cachedir)
    return Image.open(path)



def get_image_path(
        image_id: ImageId,
        cachedir: Path | None = None,
) -> Path:
    """Return the local path for an image in the packaged dataset.

    Args:
        image_id: ID of the image to resolve.
        cachedir: Optional directory containing the packaged ``data`` tree.

    Returns:
        The absolute path to the requested image.

    Raises:
        ValueError: If the image ID is not present in any packaged manifest.
        FileNotFoundError: If the resolved image file is missing on disk.
    """
    cache_root = resolve_data_root(cachedir=cachedir)
    path = _image_id_to_local_path_table(cache_root).get(image_id)
    if path is None:
        raise ValueError(f"Image ID not found in any manifest: {image_id}")
    if not path.exists():
        raise FileNotFoundError(
            f"Expected image file to exist after resolving packaged data at: {path}"
        )
    return path



# %%
if __name__ == "__main__":
    df = list_image_ids()
