import tarfile
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from hobj.data import behavior as behavior_loaders
from hobj.data import download as download_module
from hobj.data.images import get_image_path, load_image


def _write_minimal_packaged_dataset(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "image_id": "img-highvar",
                "category": "cat-a",
                "sha256": "unused",
                "relpath": "images/highvar.png",
            }
        ]
    ).to_csv(data_root / "meta-MutatorHighVarImageset.csv", index=False)

    pd.DataFrame(
        [
            {
                "image_id": "img-oneshot",
                "category": "cat-b",
                "transformation": "original",
                "transformation_level": 0.0,
                "base_image_id": "img-oneshot",
                "sha256": "unused",
                "relpath": "images/oneshot.png",
            }
        ]
    ).to_csv(data_root / "meta-MutatorOneShotImageset.csv", index=False)

    pd.DataFrame(
        [
            {
                "image_id": "img-warmup",
                "category": "cat-c",
                "sha256": "unused",
                "relpath": "images/warmup.png",
            }
        ]
    ).to_csv(data_root / "meta-MutatorWarmupImageset.csv", index=False)

    pd.DataFrame(
        [
            {
                "image_id": "img-catch",
                "sha256": "unused",
                "relpath": "images/catch.png",
            }
        ]
    ).to_csv(data_root / "meta-CatchImageset.csv", index=False)

    behavior_root = data_root / "behavior"
    behavior_root.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "trial": 0,
                "assignment_id": "assignment-1",
                "worker_id": "worker-1",
                "subtask": "cat-a,cat-d",
                "image_id": "img-highvar",
                "trial_type": "train",
                "stimulus_duration_msec": 100,
                "reaction_time_msec": 200,
                "timed_out": False,
                "perf": True,
                "timestamp_start": "2025-01-01T00:00:00",
            },
            {
                "trial": 1,
                "assignment_id": "assignment-1",
                "worker_id": "worker-1",
                "subtask": "cat-a,cat-d",
                "image_id": "img-highvar",
                "trial_type": "probe",
                "stimulus_duration_msec": 100,
                "reaction_time_msec": 200,
                "timed_out": False,
                "perf": False,
                "timestamp_start": "2025-01-01T00:00:01",
            },
            {
                "trial": 2,
                "assignment_id": "assignment-1",
                "worker_id": "worker-1",
                "subtask": "cat-a,cat-d",
                "image_id": "img-highvar",
                "trial_type": "train",
                "stimulus_duration_msec": 100,
                "reaction_time_msec": 200,
                "timed_out": False,
                "perf": True,
                "timestamp_start": "2025-01-01T00:00:02",
            },
        ]
    ).to_csv(behavior_root / "human-behavior-highvar-subtasks.csv", index=False)

    pd.DataFrame(
        [
            {
                "trial": 0,
                "assignment_id": "assignment-2",
                "slot": 0,
                "worker_id": "worker-2",
                "subtask": "cat-b,cat-e",
                "image_id": "img-oneshot",
                "trial_type": "support",
                "stimulus_duration_msec": 100,
                "reaction_time_msec": 200,
                "timed_out": False,
                "perf": True,
                "timestamp_start": "2025-01-01T00:00:03",
            }
        ]
    ).to_csv(behavior_root / "human-behavior-oneshot-subtasks.csv", index=False)


def _write_packaged_images_archive(data_root: Path) -> None:
    """Write the nested ``images.tar.gz`` archive expected by OSF downloads.

    Args:
        data_root: Packaged data directory containing metadata and behavior CSVs.
    """
    staging_root = data_root.parent / "images-staging"
    images_root = staging_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (2, 3), color=(255, 0, 0)).save(images_root / "highvar.png")
    Image.new("RGB", (2, 3), color=(0, 255, 0)).save(images_root / "oneshot.png")
    Image.new("RGB", (2, 3), color=(0, 0, 255)).save(images_root / "warmup.png")
    Image.new("RGB", (2, 3), color=(255, 255, 0)).save(images_root / "catch.png")

    with tarfile.open(data_root / "images.tar.gz", mode="w:gz") as archive:
        archive.add(images_root, arcname="images")


def test_download_data_extracts_into_custom_cachedir(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging" / "data"
    _write_minimal_packaged_dataset(staging_root)
    _write_packaged_images_archive(staging_root)

    archive_path = tmp_path / "download"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for path in sorted(staging_root.rglob("*")):
            archive.write(path, arcname=Path("data") / path.relative_to(staging_root))

    custom_cache = tmp_path / "custom-cache"
    resolved_cache = download_module.download_data(
        url="https://example.com/download?zip=",
        cachedir=custom_cache,
    )

    assert resolved_cache == custom_cache.resolve()
    assert (custom_cache / "meta-MutatorHighVarImageset.csv").exists()
    assert (custom_cache / "behavior" / "human-behavior-highvar-subtasks.csv").exists()
    assert (custom_cache / "images" / "highvar.png").exists()
    assert not (custom_cache / "images.tar.gz").exists()
    assert not (tmp_path / "data").exists()


def test_resolve_data_root_downloads_missing_data_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom_cache = tmp_path / "resolved-cache"
    calls: list[Path] = []

    def fake_download_data(
        *,
        url: str = download_module.DATA_ARCHIVE_URL,
        repo_root: Path | None = None,
        cachedir: Path | None = None,
        force_download: bool = False,
    ) -> Path:
        assert url == download_module.DATA_ARCHIVE_URL
        assert repo_root is None
        assert force_download is False
        assert cachedir is not None
        calls.append(cachedir)
        _write_minimal_packaged_dataset(cachedir)
        return cachedir.resolve()

    monkeypatch.setattr(download_module, "download_data", fake_download_data)

    resolved_cache = download_module.resolve_data_root(cachedir=custom_cache)
    assert resolved_cache == custom_cache.resolve()
    assert calls == [custom_cache.resolve()]

    resolved_cache = download_module.resolve_data_root(cachedir=custom_cache)
    assert resolved_cache == custom_cache.resolve()
    assert calls == [custom_cache.resolve()]


def test_load_highvar_behavior_uses_custom_cachedir(tmp_path: Path) -> None:
    custom_cache = tmp_path / "behavior-cache"
    _write_minimal_packaged_dataset(custom_cache)

    df = behavior_loaders.load_highvar_behavior(cachedir=custom_cache)

    assert df["trial"].tolist() == [0, 1]
    assert set(df["trial_type"]) == {"train"}
    assert df["image_id"].tolist() == ["img-highvar", "img-highvar"]


def test_load_image_uses_custom_cachedir(tmp_path: Path) -> None:
    custom_cache = tmp_path / "image-cache"
    _write_minimal_packaged_dataset(custom_cache)
    _write_packaged_images_archive(custom_cache)

    image = load_image("img-highvar", cachedir=custom_cache)

    assert image.size == (2, 3)
    image.close()


def test_get_image_path_uses_custom_cachedir(tmp_path: Path) -> None:
    custom_cache = tmp_path / "image-path-cache"
    _write_minimal_packaged_dataset(custom_cache)
    _write_packaged_images_archive(custom_cache)

    resolved_path = get_image_path("img-highvar", cachedir=custom_cache)

    assert resolved_path == custom_cache / "images" / "highvar.png"


def test_resolve_data_root_extracts_nested_images_archive_without_redownload(
    tmp_path: Path,
) -> None:
    custom_cache = tmp_path / "resolved-cache"
    _write_minimal_packaged_dataset(custom_cache)
    _write_packaged_images_archive(custom_cache)

    resolved_cache = download_module.resolve_data_root(cachedir=custom_cache)

    assert resolved_cache == custom_cache.resolve()
    assert (custom_cache / "images" / "highvar.png").exists()
    assert not (custom_cache / "images.tar.gz").exists()
