from hobj.data_loaders.behavior import load_highvar_behavior, load_oneshot_behavior


def test_load_highvar():
    df = load_highvar_behavior()
    assert len(df) == 3199 * 100


def test_load_oneshot():
    sessions = load_oneshot_behavior()
    assert len(sessions) == 2547
