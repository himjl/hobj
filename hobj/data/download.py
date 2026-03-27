"""Download packaged HOBJ datasets into a local ``data`` directory."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import requests
from platformdirs import user_data_dir
from tqdm import tqdm


# Public OSF node that hosts the packaged HOBJ dataset files.
OSF_NODE_ID = "pj6wm"
DATA_ARCHIVE_URL = (
    f"https://files.osf.io/v1/resources/{OSF_NODE_ID}/providers/osfstorage/?zip="
)
EXPECTED_DATA_RELATIVE_PATHS = (
    Path("meta-MutatorHighVarImageset.csv"),
    Path("meta-MutatorOneShotImageset.csv"),
    Path("meta-MutatorWarmupImageset.csv"),
    Path("meta-CatchImageset.csv"),
    Path("behavior/human-behavior-highvar-subtasks.csv"),
    Path("behavior/human-behavior-oneshot-subtasks.csv"),
)
_DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


def _get_default_data_root(repo_root: Path | None = None) -> Path:
    """Return the default packaged data directory.

    Args:
        repo_root: Unused legacy parameter kept for compatibility with the
            public function signature.

    Returns:
        A user-scoped persistent data directory outside the installed package.
    """
    del repo_root
    return Path(user_data_dir("hobj", appauthor=False)).resolve() / "data"


def _get_missing_expected_paths(data_root: Path) -> list[Path]:
    """Return expected packaged data paths that are currently missing."""
    return [
        relpath
        for relpath in EXPECTED_DATA_RELATIVE_PATHS
        if not (data_root / relpath).exists()
    ]


def _derive_archive_path(repo_root: Path, url: str) -> Path:
    """Return the on-disk path for the downloaded archive."""
    parsed_url = urlparse(url)
    path_segments = [segment for segment in parsed_url.path.split("/") if segment]
    if (
        parsed_url.query == "zip="
        and len(path_segments) >= 4
        and path_segments[:3] == ["v1", "resources", path_segments[2]]
    ):
        archive_name = f"{path_segments[2]}.zip"
    else:
        archive_name = Path(parsed_url.path).name
    if not archive_name:
        raise ValueError(f"Could not derive archive filename from URL: {url}")
    return repo_root / archive_name


def _download_archive(url: str, archive_path: Path) -> None:
    """Download an archive to disk atomically."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = archive_path.with_suffix(f"{archive_path.suffix}.part")

    with requests.get(url, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))
        with temp_path.open("wb") as handle:
            with tqdm(
                total=total_bytes or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {archive_path.name}",
            ) as progress_bar:
                for chunk in response.iter_content(
                    chunk_size=_DOWNLOAD_CHUNK_SIZE_BYTES
                ):
                    if chunk:
                        handle.write(chunk)
                        progress_bar.update(len(chunk))

    temp_path.replace(archive_path)


def _safe_extract_tarball(archive_path: Path, destination: Path) -> None:
    """Extract a tarball while preventing path traversal."""
    destination = destination.resolve()

    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            member_path.relative_to(destination)
        tar.extractall(destination, filter="data")


def _safe_extract_zipfile(archive_path: Path, destination: Path) -> None:
    """Extract a zipfile while preventing path traversal.

    Args:
        archive_path: Zipfile to extract.
        destination: Destination directory for extracted contents.

    Raises:
        ValueError: If a zip entry would escape the extraction directory.
    """
    destination = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            member_path.relative_to(destination)
        archive.extractall(destination)


def _find_extracted_data_root(extraction_root: Path) -> Path:
    """Return the packaged data directory from an extracted archive tree.

    Args:
        extraction_root: Temporary directory containing the extracted archive.

    Returns:
        The directory that contains the packaged data files.

    Raises:
        FileNotFoundError: If the extracted archive does not contain the
            expected packaged dataset layout.
    """
    nested_data_root = extraction_root / "data"
    if not _get_missing_expected_paths(nested_data_root):
        return nested_data_root

    if not _get_missing_expected_paths(extraction_root):
        return extraction_root

    raise FileNotFoundError(
        "Packaged dataset archive did not contain the expected data directory."
    )


def _extract_nested_images_archive(data_root: Path) -> None:
    """Extract ``images.tar.gz`` into ``data/images`` when present.

    Args:
        data_root: Extracted packaged data directory.
    """
    images_archive_path = data_root / "images.tar.gz"
    images_root = data_root / "images"
    if images_root.exists():
        if images_archive_path.exists():
            images_archive_path.unlink()
        return

    if not images_archive_path.exists():
        return

    _safe_extract_tarball(archive_path=images_archive_path, destination=data_root)
    images_archive_path.unlink()


def resolve_data_root(
    *,
    cachedir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Return a populated packaged data directory, downloading it on demand.

    Args:
        cachedir: Optional directory to use instead of the repository
            ``data`` directory.
        repo_root: Optional repository root used only when ``cachedir`` is not
            provided.

    Returns:
        The absolute path to the packaged data directory.
    """
    data_root = (
        cachedir if cachedir is not None else _get_default_data_root(repo_root)
    ).resolve()
    if _get_missing_expected_paths(data_root):
        return download_data(cachedir=data_root, url=DATA_ARCHIVE_URL)
    _extract_nested_images_archive(data_root)
    return data_root


def download_data(
    *,
    url: str = DATA_ARCHIVE_URL,
    repo_root: Path | None = None,
    cachedir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download and extract the packaged learning dataset into a cache root.

    The function is idempotent: if the expected packaged files already exist
    under the resolved data directory, it returns immediately without
    downloading or extracting. Otherwise it downloads the archive if needed and
    extracts it into the target cache.

    Args:
        url: Archive URL containing the packaged ``data`` tree.
        repo_root: Repository root used when ``cachedir`` is not provided.
        cachedir: Optional directory to use instead of the repository
            ``data`` directory.
        force_download: Whether to redownload the archive even if it is already
            present on disk.

    Returns:
        The absolute path to the extracted ``data`` directory.

    Raises:
        requests.HTTPError: If the archive download fails.
        tarfile.TarError: If a tar archive cannot be unpacked.
        zipfile.BadZipFile: If the zip archive cannot be unpacked.
        ValueError: If the archive path is invalid or extraction would escape
            the extraction directory.
        FileNotFoundError: If extraction completes but the expected dataset
            files are still missing.
    """
    data_root = (
        cachedir if cachedir is not None else _get_default_data_root(repo_root)
    ).resolve()
    missing_paths = _get_missing_expected_paths(data_root)
    if not missing_paths:
        return data_root

    archive_root = data_root.parent
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = _derive_archive_path(archive_root, url)
    if force_download or not archive_path.exists():
        _download_archive(url=url, archive_path=archive_path)

    with TemporaryDirectory(dir=archive_root) as extraction_root_str:
        extraction_root = Path(extraction_root_str)
        _safe_extract_zipfile(archive_path=archive_path, destination=extraction_root)
        extracted_data_root = _find_extracted_data_root(extraction_root)
        _extract_nested_images_archive(extracted_data_root)
        shutil.copytree(extracted_data_root, data_root, dirs_exist_ok=True)

    missing_paths = _get_missing_expected_paths(data_root)
    if missing_paths:
        missing_str = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Packaged dataset archive was extracted, but expected files are still "
            f"missing:\n{missing_str}"
        )

    return data_root


def main() -> None:
    """Download packaged dataset files into a local data directory."""
    parser = argparse.ArgumentParser(
        description="Download and extract the packaged HOBJ dataset archive."
    )
    parser.add_argument(
        "--cachedir",
        type=Path,
        default=None,
        help="Optional directory to use instead of the repository data directory.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the archive even if it already exists locally.",
    )
    args = parser.parse_args()

    data_root = download_data(
        cachedir=args.cachedir,
        force_download=args.force_download,
    )
    print(data_root)


if __name__ == "__main__":
    main()
