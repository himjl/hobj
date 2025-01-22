import json
import tempfile
from pathlib import Path
from typing import List, Iterator

import PIL.Image

from hobj.config import Config, default_config

__all__ = ['DataStore', 'default_data_store']


class _PathManager:
    def __init__(self, config: Config):
        self.config = config

    # JSON caching from web
    def get_json_cache_path(self, url: str) -> Path:
        """
        Construct a cache path for a URL.
        :param url:
        :return:
        """
        if not url.endswith('.json'):
            raise ValueError(f"URL {url} does not point to a JSON file.")

        path = self.config.cachedir / 'json_cache' / (utils.hash_string(url) + '.json')
        return path

    def get_image_cache_path(self, sha256: str) -> Path:
        path = self.config.cachedir / 'image_cache' / (sha256 + '.png')
        return path

    def get_triplet_cache_path(self, url:str) -> Path:

        if not url.endswith('.json.gz'):
            raise ValueError(f"URL {url} does not point to a gzipped JSON file; expecting .json.gz file")

        path = self.config.cachedir / 'compressed_tripletsets' / (utils.hash_string(url) + '.json.gz')
        return path

    def get_unpacked_tripletset_path(self, url: str) -> Path:
        if not url.endswith('.json.gz'):
            raise ValueError(f"URL {url} does not point to a gzipped JSON file; expecting .json.gz file")

        path = self.config.cachedir / 'compressed_tripletsets' / (utils.hash_string(url) + '.json')
        return path

    def get_zipped_imageset_cache_path(self, imageset_url: str) -> Path:
        if not imageset_url.endswith('.zip'):
            raise ValueError(f"URL {imageset_url} does not point to a ZIP file")

        path = self.config.cachedir / 'compressed_imagesets' / (utils.hash_string(imageset_url) + '.zip')
        return path


class DataStore:

    def __init__(self, config: Config):
        self.config = config
        self.path_manager = _PathManager(config=config)
        self._image_cache_manifest = set()
        self._db = None

    # Images
    def register_image(self, image_data: PIL.Image) -> schema.ImageRef:
        """
        Register an image in the data store.
        :param image_data:
        :return:
        """
        sha256 = utils.hash_image(image=image_data)
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

    def list_images(self) -> List[schema.ImageRef]:
        """
        List all images in the data store.
        :return:
        """
        raise NotImplementedError()

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

    def get_compressed_triplet_dataset(self, url:str, redownload: bool = False) -> Path:
        """
        Get the path to the compressed triplet dataset file.
        :param url:
        :param redownload:
        :return:
        """

        path = self.path_manager.get_triplet_cache_path(url=url)
        if not path.exists() or redownload:
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                savepath = tempdir / 'tmp.json.gz'
                utils.download_file(url=url, output_path=savepath)

                if not path.parent.exists():
                    path.parent.mkdir(parents=True)

                # Move the file to the correct location
                savepath.rename(path)

        return path

    def get_compressed_imageset(self, imageset_url: str, redownload: bool = False) -> Path:
        """
        Get the path to the compressed imageset file.
        :param imageset_url:
        :return:
        """
        path = self.path_manager.get_zipped_imageset_cache_path(imageset_url)
        if not path.exists() or redownload:

            # Save it in a tempdir
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir = Path(tempdir)
                savepath = tempdir / 'tmp.zip'
                utils.download_file(url=imageset_url, output_path=savepath)

                if not path.parent.exists():
                    path.parent.mkdir(parents=True)

                # Move the file to the correct location
                savepath.rename(path)

        return path

    def get_json_data_from_url(self, url: str, redownload: bool = False):
        """
        Loads JSON data file from a url, caching it if it is not already present.
        :param url:
        :param redownload:
        :return:
        """
        path = self.path_manager.get_json_cache_path(url=url)

        if not path.exists() or redownload:
            json_data = utils.download_json(url=url)
            if not path.parent.exists():
                path.parent.mkdir(parents=True)
            path.write_text(json.dumps(json_data))
            print(f"Downloaded JSON data from {url} to {path}")

        return json.loads(path.read_text())


# Default data store
default_data_store = DataStore(config=default_config)
