from pathlib import Path
from typing import Literal

import pydantic

from hobj.data_loaders.images.template import Imageset


class MutatorOneShotAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'MutatorB2000_2444',
            'MutatorB2000_288',
        ],
        pattern=r'^MutatorB2000_\d+$',
        description='The name for the object model associated with this image.'
    )

    transformation: Literal[
        'combinednat',
        'blur',
        'inplanetranslation',
        'scale',
        'outplanerotation',
        'noise',
        'backgrounds',
        'inplanerotation',
        'delpixels',
        'contrast',
        'original'
    ] = pydantic.Field(description = 'The type of transformation applied to the image. "original" indicates no transformation.')

    transformation_level: float = pydantic.Field(
        description='The extent of transformation applied to the image. The interpretation of this number varies by transformation.'
    )

    base_image_id: str = pydantic.Field(
        description = 'An image ID for the "original" image associated with this one.',
        examples = [
            'MutatorB2000_2444_rx0.00000_ry0.00000_rz0.00000.png',
            'MutatorB2000_4792_rx0.00000_ry0.00000_rz0.00000.png'
        ]
    )


class MutatorOneShotImageset(Imageset[MutatorOneShotAnnotation]):
    manifest_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-oneshot/mutator-oneshot-manifest.json'
    zipped_images_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-oneshot/MutatorB2000_Oneshot64.zip'
    annotation_schema = MutatorOneShotAnnotation

    def __init__(self, cachedir: Path | None = None, redownload: bool = False):
        super().__init__(cachedir=cachedir, redownload=redownload)


if __name__ == '__main__':
    imageset = MutatorOneShotImageset()
