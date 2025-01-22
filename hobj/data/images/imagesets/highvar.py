
from hobj.data.images.template import Imageset, ImageManifestEntry, ImageManifest
import pydantic


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


if __name__ == '__main__':

    imageset = MutatorHighVarImageset()
