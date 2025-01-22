import json
import tempfile
from pathlib import Path
from typing import List, Iterator

import PIL.Image

import hobj.config as config
import hobj.data.schema as schema
from hobj.utils.hash import hash_image

__all__ = ['DataStore', 'default_data_store']


class _PathManager:

    def __init__(self, cachedir: Path):
        self.cachedir = cachedir

    def get_image_cache_path(self, sha256: str) -> Path:
        path = self.cachedir / 'image_cache' / (sha256 + '.png')
        return path


class DataStore:

    def __init__(self, cachedir: Path):
        self.path_manager = _PathManager(cachedir=cachedir)
        self._image_cache_manifest = set()

    def register_image(self, image_data: PIL.Image) -> schema.ImageRef:
        """
        Register an image in the data store.
        :param image_data:
        :return:
        """
        sha256 = hash_image(image=image_data)
        path = self.path_manager.get_image_cache_path(sha256=sha256)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        image_data.save(path)
        return schema.ImageRef(sha256=sha256)

    def check_image_exists(self, sha256: str) -> bool:
        """
        Checks whether an image is already cached in the data store.
        :param sha256:
        :return:
        """
        if sha256 not in self._image_cache_manifest:
            path = self.path_manager.get_image_cache_path(sha256=sha256)
            if path.exists():
                self._image_cache_manifest.add(sha256)
        return sha256 in self._image_cache_manifest

    def load_image(self, image_ref: schema.ImageRef, as_rgb: bool = True) -> PIL.Image:
        """
        Load an image from the data store.
        If the image is not there, registers it.
        :param image_ref:
        :return:
        """
        path = self.path_manager.get_image_cache_path(sha256=image_ref.sha256)
        with PIL.Image.open(path) as img:
            if as_rgb:
                img = img.convert('RGB')
            loaded_image = img.copy()

        return loaded_image


# Default data store
default_data_store = DataStore(cachedir=config.cachedir)
