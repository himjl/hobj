from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    cachedir = Path.home() / 'hobj_cache'

if not hlb_cachedir.exists():
    hlb_cachedir.mkdir()

