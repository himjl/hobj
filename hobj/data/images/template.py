import tempfile
from abc import ABC
from pathlib import Path
from typing import Any, Dict, TypeVar, Generic, List

import PIL.Image
import pydantic
from tqdm import tqdm

from hobj.utils.file_io import unzip_file
from hobj.utils.hash import hash_image
from hobj.data.schema import ImageRef
from hobj.data.store import DataStore, default_data_store


class ImageManifestEntry(pydantic.BaseModel, ABC):
    sha256: str = pydantic.Field(pattern=r'^[a-f0-9]{64}$')
    relpath: Path = pydantic.Field(
        description='The relative path to the image file inside of the unzipped imageset directory.'
    )
    annotation: Any = pydantic.Field(
        description='Arbitrary annotation data for the image, in a JSON-valid format.'
    )


class ImageManifest(pydantic.BaseModel):
    entries: List[ImageManifestEntry] = pydantic.Field(
        description='A list of image manifest entries.'
    )

    zipped_images_url: str = pydantic.Field(
        description='The URL to the zipped imageset file.'
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
    annotation_schema: IA

    def __init__(
            self,
            data_store: DataStore = None,
            redownload_manifest=False,
    ):
        """
        Unwrap the image manifest and save the images to the cache.
        """

        if not data_store:
            self.data_store: DataStore = default_data_store

        # Load the manifest if it is already cached
        manifest_data = self.data_store.get_json_data_from_url(url=self.manifest_url, redownload=redownload_manifest)
        image_manifest = ImageManifest(**manifest_data)

        self._register_image_urls(manifest=image_manifest)
        self._manifest = image_manifest

        self._image_ref_to_annotation: Dict[ImageRef, IA] = {}
        self._image_refs: List[ImageRef] = []

        for entry in image_manifest.entries:
            image_ref = ImageRef(sha256=entry.sha256)
            self._image_refs.append(image_ref)
            if image_ref in self._image_ref_to_annotation:
                raise ValueError(f"Multiple annotations for image {image_ref}: \n{entry.annotation} \n{self._image_ref_to_annotation[image_ref]}")
            self._image_ref_to_annotation[image_ref] = self.annotation_schema(**entry.annotation)

    def _register_image_urls(self, manifest: ImageManifest):
        """
        Ensures the entries of the manifest are registered in the data store.
        """

        num_undownloaded_images = 0
        for manifest_entry in manifest.entries:


            if not self.data_store.check_image_exists(sha256=manifest_entry.sha256):
                num_undownloaded_images += 1

        if num_undownloaded_images == 0:
            return

        print(f'Missing {num_undownloaded_images}/{len(manifest.entries)} images for this imageset.')

        # Download the images
        zipped_images_path = self.data_store.get_zipfile(zipfile_url=manifest.zipped_images_url)

        # Make a tempdir to unzip the images
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            unzip_file(zip_path=zipped_images_path, output_dir=tempdir)

            # Register the images
            pbar = tqdm(total=len(manifest.entries))
            for manifest_entry in manifest.entries:
                reported_sha256 = manifest_entry.sha256
                relpath = manifest_entry.relpath
                image_path = tempdir / relpath
                image_data = PIL.Image.open(image_path)
                actual_sha256 = hash_image(image=image_data)
                if not actual_sha256 == reported_sha256:
                    raise ValueError(f"SHA256 mismatch for image {manifest_entry}: {actual_sha256} != {reported_sha256}")

                self.data_store.register_image(image_data=image_data)
                pbar.update(1)

    @property
    def image_refs(self) -> List[ImageRef]:
        """
        List of image refs in this imageset.
        :return:
        """
        return self._image_refs

    def get_annotation(self, image_ref: ImageRef) -> IA:
        """
        Get the annotation for a given image. If an image has multiple annotations, this will throw an error.
        """

        entry = self._image_ref_to_annotation[image_ref]
        return self.annotation_schema(**entry.annotation)

    def __len__(self) -> int:
        return len(self.image_refs)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"

    def __str__(self):
        return f"Imageset(n={len(self)})"
