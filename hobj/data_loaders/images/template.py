import json
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Generic, List, TypeVar

import PIL.Image
import pydantic

from hobj.types import ImageId
from hobj.utils.file_io import download_file, download_json, unzip_file
from hobj.utils.hash import hash_image


# %%
class ImageManifestEntry(pydantic.BaseModel, ABC):
    sha256: str = pydantic.Field(pattern=r'^[a-f0-9]{64}$')
    relpath: Path = pydantic.Field(
        description='The relative path to the image file inside of the unzipped imageset directory.'
    )
    annotation: Any = pydantic.Field(
        description='Arbitrary annotation data for the image, in a JSON-valid format.'
    )


class ImageManifest(pydantic.BaseModel):
    entries: Dict[ImageId, ImageManifestEntry] = pydantic.Field(
        description='A mapping from a unique image ID to image manifest entries.'
    )


# This is a type variable for ImageManifestEntry.annotation:
IA = TypeVar('IA')


class Imageset(Generic[IA], ABC):
    """
    Imagesets are a combination of:
    - Images
    - Annotations on those images
    """

    manifest_url: str
    zipped_images_url: str
    annotation_schema: IA

    def __init__(
            self,
            cachedir: Path | None = None,
            redownload=False,
    ):
        """
        Download and materialize the imageset into a local cache directory.
        """

        repo_root = Path(__file__).resolve().parents[3]
        self.cachedir = (cachedir if cachedir is not None else repo_root / 'data').resolve()
        self.cachedir.mkdir(parents=True, exist_ok=True)

        self._dataset_dir = self.cachedir / self.__class__.__name__
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir = self._dataset_dir / 'images'

        manifest_data = self._load_manifest_json(redownload=redownload)
        image_manifest = ImageManifest(**manifest_data)
        self._manifest = image_manifest
        self._ensure_images_present(manifest=image_manifest, redownload=redownload)

        self._image_id_to_annotation: Dict[ImageId, IA] = {}
        self._image_id_to_sha256: Dict[ImageId, str] = {}
        self._image_id_to_relpath: Dict[ImageId, Path] = {}
        self._image_ids: List[ImageId] = []

        for image_id, entry in image_manifest.entries.items():
            self._image_ids.append(image_id)
            self._image_id_to_sha256[image_id] = entry.sha256
            self._image_id_to_relpath[image_id] = entry.relpath
            self._image_id_to_annotation[image_id] = self.annotation_schema(**entry.annotation)

    def _load_manifest_json(self, redownload: bool) -> dict[str, Any]:
        if redownload or not self.manifest_path.exists():
            manifest_data = download_json(self.manifest_url)
            self.manifest_path.write_text(json.dumps(manifest_data, indent=2))
        return json.loads(self.manifest_path.read_text())

    def _ensure_images_present(self, manifest: ImageManifest, redownload: bool = False) -> None:
        """
        Ensure the images for this imageset exist locally.
        """
        if redownload or not self._all_images_present(manifest):
            self._download_and_extract_images(manifest)

    def _all_images_present(self, manifest: ImageManifest) -> bool:
        for entry in manifest.entries.values():
            if not (self._images_dir / entry.relpath).exists():
                return False
        return True

    def _download_and_extract_images(self, manifest: ImageManifest) -> None:
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        if self._images_dir.exists():
            for path in sorted(self._images_dir.rglob('*'), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            self._images_dir.rmdir()

        download_file(self.zipped_images_url, self.archive_path)
        unzip_file(zip_path=self.archive_path, output_dir=self._images_dir)
        self._verify_images(manifest)

    def _verify_images(self, manifest: ImageManifest) -> None:
        for image_id, entry in manifest.entries.items():
            image_path = self._images_dir / entry.relpath
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file for {image_id}: {image_path}")

            with PIL.Image.open(image_path) as image_data:
                observed_sha256 = hash_image(image_data)

            if observed_sha256 != entry.sha256:
                raise ValueError(
                    f"SHA256 mismatch for image {image_id}: {observed_sha256} != {entry.sha256}"
                )

    @property
    def manifest_path(self) -> Path:
        return self._dataset_dir / 'manifest.json'

    @property
    def archive_path(self) -> Path:
        archive_name = Path(self.zipped_images_url).name
        return self._dataset_dir / archive_name

    @property
    def images_dir(self) -> Path:
        return self._images_dir

    @property
    def image_ids(self) -> list[ImageId]:
        """
        List of image refs in this imageset.
        :return:
        """
        return self._image_ids

    def get_annotation(self, *, image_id: ImageId) -> IA:
        """
        Get the annotation for a given image. If an image has multiple annotations, this will throw an error.
        """

        entry = self._image_id_to_annotation[image_id]
        return entry

    def get_image_path(self, *, image_id: ImageId) -> Path:
        return self.images_dir / self._image_id_to_relpath[image_id]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"

    def __str__(self):
        return f"Imageset(n={len(self)})"
