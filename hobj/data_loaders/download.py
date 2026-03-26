"""Download packaged HOBJ datasets into the repository ``data`` directory."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


DATA_ARCHIVE_URL = (
    "https://hlbdatasets.s3.us-east-1.amazonaws.com/"
    "lee-dicarlo-2023-learning-data.tar.gz"
)
EXPECTED_DATA_PATHS = (
    Path("data/meta-MutatorHighVarImageset.csv"),
    Path("data/meta-MutatorOneShotImageset.csv"),
    Path("data/meta-MutatorWarmupImageset.csv"),
    Path("data/meta-CatchImageset.csv"),
    Path("data/behavior/human-behavior-highvar-subtasks.csv"),
    Path("data/behavior/human-behavior-oneshot-subtasks.csv"),
)
_DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


def _get_repo_root() -> Path:
    """Return the repository root for this package."""
    return Path(__file__).resolve().parents[2]


def _get_missing_expected_paths(repo_root: Path) -> list[Path]:
    """Return expected packaged data paths that are currently missing."""
    return [
        relpath for relpath in EXPECTED_DATA_PATHS if not (repo_root / relpath).exists()
    ]


def _derive_archive_path(repo_root: Path, url: str) -> Path:
    """Return the on-disk path for the downloaded archive."""
    archive_name = Path(urlparse(url).path).name
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
        tar.extractall(destination)


def download_data(
    *,
    url: str = DATA_ARCHIVE_URL,
    repo_root: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download and extract the packaged learning dataset into the repo root.

    The function is idempotent: if the expected packaged files already exist
    under ``data/``, it returns immediately without downloading or extracting.
    Otherwise it downloads the archive if needed and extracts it into the
    repository root.

    Args:
        url: Archive URL containing the packaged ``data`` tree.
        repo_root: Repository root where the archive should be stored and
            extracted.
        force_download: Whether to redownload the archive even if it is already
            present on disk.

    Returns:
        The absolute path to the extracted ``data`` directory.

    Raises:
        requests.HTTPError: If the archive download fails.
        tarfile.TarError: If the archive cannot be unpacked.
        ValueError: If the archive path is invalid or extraction would escape
            the repository root.
        FileNotFoundError: If extraction completes but the expected dataset
            files are still missing.
    """
    resolved_repo_root = (
        repo_root if repo_root is not None else _get_repo_root()
    ).resolve()
    data_root = resolved_repo_root / "data"
    missing_paths = _get_missing_expected_paths(resolved_repo_root)
    if not missing_paths:
        return data_root

    archive_path = _derive_archive_path(resolved_repo_root, url)
    if force_download or not archive_path.exists():
        _download_archive(url=url, archive_path=archive_path)

    _safe_extract_tarball(archive_path=archive_path, destination=resolved_repo_root)

    missing_paths = _get_missing_expected_paths(resolved_repo_root)
    if missing_paths:
        missing_str = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Packaged dataset archive was extracted, but expected files are still "
            f"missing:\n{missing_str}"
        )

    return data_root


def main() -> None:
    """Download packaged dataset files into the repository data directory."""
    parser = argparse.ArgumentParser(
        description="Download and extract the packaged HOBJ dataset archive."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the archive even if it already exists locally.",
    )
    args = parser.parse_args()

    data_root = download_data(force_download=args.force_download)
    print(data_root)


if __name__ == "__main__":
    main()
