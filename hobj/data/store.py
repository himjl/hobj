from pathlib import Path

import PIL.Image
import hobj.config

from hobj.utils.hash import hash_image, hash_string
from hobj.utils.file_io import download_json, download_file
import json
import tempfile

__all__ = ['DataStore']


# %%
class _PathManager:

    def __init__(self, cachedir: Path):
        self.cachedir = cachedir

    def get_image_cache_path(self, sha256: str) -> Path:
        path = self.cachedir / 'image_cache' / (sha256 + '.png')
        return path

    # JSON caching from web
    def get_json_cache_path(self, url: str) -> Path:
        """
        Construct a cache path for a URL.
        :param url:
        :return:
        """
        if not url.endswith('.json'):
            raise ValueError(f"URL {url} does not point to a JSON file.")

        path = self.cachedir / 'json_cache' / (hash_string(url) + '.json')
        return path

    def get_zipped_file_cache_path(self, zipfile_url: str) -> Path:
        if not zipfile_url.endswith('.zip'):
            raise ValueError(f"URL {zipfile_url} does not point to a ZIP file")

        path = self.cachedir / 'zipfile_cache' / (hash_string(zipfile_url) + '.zip')
        return path


class DataStore:

    def __init__(self, cachedir: Path):
        self.path_manager = _PathManager(cachedir=cachedir)
        self._image_cache_manifest = set()

    def register_image(self, image_data: PIL.Image) -> str:
        """
        Register an image in the data store.
        :param image_data:
        :return: The SHA256 hash of the image.
        """
        sha256 = hash_image(image=image_data)
        path = self.path_manager.get_image_cache_path(sha256=sha256)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        image_data.save(path)
        return sha256

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

    def load_image(self, sha256: str, as_rgb: bool = True) -> PIL.Image:
        """
        Load an image from the data store.
        If the image is not there, registers it.
        :param image_ref:
        :return:
        """
        path = self.path_manager.get_image_cache_path(sha256=sha256)
        with PIL.Image.open(path) as img:
            if as_rgb:
                img = img.convert('RGB')
            loaded_image = img.copy()

        return loaded_image

    def get_json_data_from_url(self, url: str, redownload: bool = False):
        """
        Loads JSON data file from a url, caching it if it is not already present.
        :param url:
        :param redownload:
        :return:
        """
        path = self.path_manager.get_json_cache_path(url=url)

        if not path.exists() or redownload:
            json_data = download_json(url=url)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_data))
            print(f"Downloaded JSON data from {url} to {path}")

        return json.loads(path.read_text())

    def get_zipfile(self, zipfile_url: str, redownload: bool = False) -> Path:
        """
        Get the path to the compressed imageset file.
        :param zipfile_url:
        :return:
        """
        path = self.path_manager.get_zipped_file_cache_path(zipfile_url)
        if not path.exists() or redownload:

            # Save it in a tempdir
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                savepath = tempdir / 'tmp.zip'
                download_file(url=zipfile_url, output_path=savepath)

                if not path.parent.exists():
                    path.parent.mkdir(parents=True)

                # Move the file to the correct location
                savepath.rename(path)

        return path


# Default data store
default_data_store = DataStore(cachedir=hobj.config.cachedir)
