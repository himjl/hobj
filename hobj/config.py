from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    cachedir: str = Path.home() / 'hobj_cache'

default_config = Config()
