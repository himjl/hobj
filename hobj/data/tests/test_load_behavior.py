from hobj.data.behavior import load_highvar_behavior, load_oneshot_behavior


def test_load_highvar():
    sessions = load_highvar_behavior()
    assert len(sessions) == 3199


def test_load_oneshot():
    sessions = load_oneshot_behavior()
    assert len(sessions) == 2547
