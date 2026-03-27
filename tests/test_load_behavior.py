import os

import pytest

from hobj.data.behavior import load_highvar_behavior, load_oneshot_behavior
from hobj.data.images import (
    load_image,
    load_imageset_meta_catch,
    load_imageset_meta_highvar,
    load_imageset_meta_oneshot,
    load_imageset_meta_warmup,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Packaged dataset is not available on GitHub Actions runners.",
)


def test_load_highvar():
    df = load_highvar_behavior()
    assert len(df) == 3199 * 100
    assert "image_id" in df.columns
    assert "stimulus_id" not in df.columns


def test_load_oneshot():
    df = load_oneshot_behavior()
    assert len(df) == 2547 * 20
    assert "image_id" in df.columns
    assert "stimulus_id" not in df.columns


def test_load_highvar_images():
    df = load_imageset_meta_highvar()
    assert len(df) == 12800
    assert {"image_id", "category", "sha256", "relpath"} <= set(df.columns)


def test_load_oneshot_images():
    df = load_imageset_meta_oneshot()
    assert len(df) == 3904
    assert {
        "image_id",
        "category",
        "transformation",
        "transformation_level",
        "base_image_id",
        "sha256",
        "relpath",
    } <= set(df.columns)


def test_load_warmup_images():
    df = load_imageset_meta_warmup()
    assert len(df) == 400
    assert {"image_id", "category", "sha256", "relpath"} <= set(df.columns)


def test_load_probe_images():
    df = load_imageset_meta_catch()
    assert len(df) == 2
    assert {"image_id", "sha256", "relpath"} <= set(df.columns)


def test_load_image():
    df = load_imageset_meta_highvar()
    image = load_image(df["image_id"].iloc[0])
    assert image.size[0] > 0
    assert image.size[1] > 0
