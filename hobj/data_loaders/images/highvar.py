from pathlib import Path
import pydantic

from hobj.data_loaders.images.template import Imageset

from hobj.types import ImageId

# %%

class MutatorHighVarAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'MutatorB2000_4872',
            'MutatorB2000_419',
        ],
        pattern=r'^MutatorB2000_\d+$'
    )


class MutatorHighVarImageset(Imageset[MutatorHighVarAnnotation]):
    annotation_schema = MutatorHighVarAnnotation

    def __init__(self, cachedir: Path | None = None, redownload: bool = False):
        super().__init__(cachedir=cachedir, redownload=redownload)

        self._category_to_image_ids: dict[str, list[ImageId]] = {}

        for ref in self.image_ids:
            annotation = self.get_annotation(image_id=ref)
            category = annotation.category
            if category not in self._category_to_image_ids:
                self._category_to_image_ids[category] = []
            self._category_to_image_ids[category].append(ref)

        for category in self._category_to_image_ids:
            self._category_to_image_ids[category] = sorted(self._category_to_image_ids[category])

    @property
    def category_to_image_ids(self) -> dict[str, list[ImageId]]:
        return self._category_to_image_ids


if __name__ == '__main__':

    imageset = MutatorHighVarImageset()
