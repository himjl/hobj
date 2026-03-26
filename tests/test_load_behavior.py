from hobj.data_loaders.behavior import load_highvar_behavior, load_oneshot_behavior
from hobj.data_loaders.images import (
    load_mutator_highvar_images,
    load_mutator_oneshot_images,
    load_mutator_warmup_images,
    load_probe_images,
)


def test_load_highvar():
    df = load_highvar_behavior()
    assert len(df) == 3199 * 100
    assert 'image_id' in df.columns
    assert 'stimulus_id' not in df.columns


def test_load_oneshot():
    df = load_oneshot_behavior()
    assert len(df) == 2547 * 20
    assert 'image_id' in df.columns
    assert 'stimulus_id' not in df.columns


def test_load_highvar_images():
    df = load_mutator_highvar_images()
    assert len(df) == 12800
    assert {'image_id', 'category', 'sha256', 'relpath', 'image_path'} <= set(df.columns)
    assert df['image_path'].iloc[0].exists()


def test_load_oneshot_images():
    df = load_mutator_oneshot_images()
    assert len(df) == 3904
    assert {
        'image_id',
        'category',
        'transformation',
        'transformation_level',
        'base_image_id',
        'sha256',
        'relpath',
        'image_path',
    } <= set(df.columns)
    assert df['image_path'].iloc[0].exists()


def test_load_warmup_images():
    df = load_mutator_warmup_images()
    assert len(df) == 400
    assert {'image_id', 'category', 'sha256', 'relpath', 'image_path'} <= set(df.columns)
    assert df['image_path'].iloc[0].exists()


def test_load_probe_images():
    df = load_probe_images()
    assert len(df) == 2
    assert {'image_id', 'color', 'text', 'sha256', 'relpath', 'image_path'} <= set(df.columns)
    assert df['image_path'].iloc[0].exists()
