
from hobj.data.images.template import Imageset, ImageManifestEntry, ImageManifest
import pydantic

from typing import Dict, List

import hobj.data.schema as schema


class MutatorHighVarAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'MutatorB2000_4872',
            'MutatorB2000_419',
        ],
        pattern=r'^MutatorB2000_\d+$'
    )


class MutatorHighVarImageset(Imageset[MutatorHighVarAnnotation]):
    manifest_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-highvar/mutator-highvar-manifest.json'
    zipped_images_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-highvar/MutatorB2000_Subset128_FullVar_Train.zip'
    annotation_schema = MutatorHighVarAnnotation

    def __init__(self):
        super().__init__()

        self._category_to_image_refs: Dict[str, List[schema.ImageRef]] = {}

        for ref in self.image_refs:
            annotation = self.get_annotation(image_ref=ref)
            category = annotation.category
            if category not in self._category_to_image_refs:
                self._category_to_image_refs[category] = []
            self._category_to_image_refs[category].append(ref)

        for category in self._category_to_image_refs:
            self._category_to_image_refs[category] = sorted(self._category_to_image_refs[category])

    @property
    def category_to_image_refs(self) -> Dict[str, List[schema.ImageRef]]:
        return self._category_to_image_refs


if __name__ == '__main__':

    imageset = MutatorHighVarImageset()
