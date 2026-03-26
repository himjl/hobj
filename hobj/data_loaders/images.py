"""Load packaged image manifests and images."""

from pathlib import Path

import pandas as pd
from hobj.types import ImageId
from PIL import Image
from functools import lru_cache


@lru_cache(maxsize=1)
def _image_id_to_local_path_table() -> dict[ImageId, Path]:
    """Build a mapping from image_id to absolute local path for all packaged images."""
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = repo_root / "data"
    manifest_paths = list(cache_root.glob("meta-*.csv"))
    table = {}
    for manifest_path in manifest_paths:
        manifest_df = pd.read_csv(manifest_path)
        for _, row in manifest_df.iterrows():
            image_id = row["image_id"]
            relpath = row["relpath"]
            abs_path = cache_root / relpath
            table[image_id] = abs_path
    return table


def load_image(image_id: ImageId) -> Image.Image:
    """Load an image by ``image_id`` from the packaged dataset."""
    path = _image_id_to_local_path_table().get(image_id)
    if path is None:
        raise ValueError(f"Image ID not found in any manifest: {image_id}")
    if not path.exists():
        raise FileNotFoundError(f"Expected image file to already exist at: {path}")
    return Image.open(path)


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
    repo_root = Path(__file__).resolve().parents[2]
    cache_root = (cachedir if cachedir is not None else repo_root / "data").resolve()
    manifest_path = cache_root / f"meta-{dataset_name}.csv"

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
    missing_paths = manifest_df.loc[
        ~manifest_df["relpath"].map(lambda p: Path.exists(cache_root / p)), "relpath"
    ]
    if not missing_paths.empty:
        raise FileNotFoundError(
            "Expected packaged images to already exist under:\n"
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


if __name__ == "__main__":
    df = load_imageset_meta_oneshot()
