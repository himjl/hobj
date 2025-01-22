from pathlib import Path

hlb_cachedir = Path.home() / 'hlb_cache'

if not hlb_cachedir.exists():
    hlb_cachedir.mkdir()

