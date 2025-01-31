from pathlib import Path

import PIL.Image
import hobj.config

from hobj.utils.hash import hash_image, hash_string
from hobj.utils.file_io import download_json, download_file
import json
import tempfile

import mref
__all__ = [
    'default_data_store'
]


# %%
# Default data store
default_data_store = mref.FileSystemStorage(cachedir=hobj.config.cachedir)
