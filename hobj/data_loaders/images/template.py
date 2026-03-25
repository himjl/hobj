from abc import ABC
from pathlib import Path
from typing import Any, Dict, Generic, List, TypeVar

import pandas as pd
import pydantic

from hobj.types import ImageId


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

    annotation_schema: IA

    def __init__(
            self,
            cachedir: Path | None = None,
            redownload: bool = False,
    ) -> None:
        """
        Load a packaged imageset from the local cache directory.
        """
        if redownload:
            raise ValueError(
                f"{self.__class__.__name__} no longer supports redownload; expected cached manifest.csv and images to already exist."
            )

        repo_root = Path(__file__).resolve().parents[3]
        self.cachedir = (cachedir if cachedir is not None else repo_root / 'data').resolve()
        self._dataset_dir = self.cachedir / 'images' / self.__class__.__name__
        self._images_dir = self._dataset_dir / 'images'

        manifest_df = self._load_manifest_df()
        image_manifest = self._parse_manifest_df(manifest_df=manifest_df)
        self._manifest = image_manifest
        self._ensure_images_present(manifest=image_manifest)

        self._image_id_to_annotation: Dict[ImageId, IA] = {}
        self._image_id_to_sha256: Dict[ImageId, str] = {}
        self._image_id_to_relpath: Dict[ImageId, Path] = {}
        self._image_ids: List[ImageId] = []

        for image_id, entry in image_manifest.entries.items():
            self._image_ids.append(image_id)
            self._image_id_to_sha256[image_id] = entry.sha256
            self._image_id_to_relpath[image_id] = entry.relpath
            self._image_id_to_annotation[image_id] = self.annotation_schema(**entry.annotation)

    @property
    def annotation_column_names(self) -> list[str]:
        return list(self.annotation_schema.model_fields.keys())

    def _load_manifest_df(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Expected cached manifest.csv to already exist at: {self.manifest_path}"
            )
        return pd.read_csv(self.manifest_path)

    def _parse_manifest_df(self, *, manifest_df: pd.DataFrame) -> ImageManifest:
        required_columns = {'image_id', 'sha256', 'relpath', *self.annotation_column_names}
        missing_columns = required_columns - set(manifest_df.columns)
        if missing_columns:
            raise ValueError(
                f"{self.__class__.__name__} manifest.csv missing required columns: {sorted(missing_columns)}"
            )

        entries: Dict[ImageId, ImageManifestEntry] = {}
        for _, row in manifest_df.iterrows():
            image_id = str(row['image_id'])
            annotation = {
                column_name: row[column_name]
                for column_name in self.annotation_column_names
            }
            entries[image_id] = ImageManifestEntry(
                sha256=str(row['sha256']),
                relpath=Path(str(row['relpath'])),
                annotation=annotation,
            )

        return ImageManifest(entries=entries)

    def _ensure_images_present(self, manifest: ImageManifest) -> None:
        if not self._all_images_present(manifest):
            raise FileNotFoundError(
                f"Expected packaged images to already exist under: {self.images_dir}"
            )

    def _all_images_present(self, manifest: ImageManifest) -> bool:
        for entry in manifest.entries.values():
            if not (self._images_dir / entry.relpath).exists():
                return False
        return True

    @property
    def manifest_path(self) -> Path:
        return self._dataset_dir / 'manifest.csv'

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
