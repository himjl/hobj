import pydantic

from hobj.data_loaders.images.template import Imageset


class MutatorWarmupAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'Mutator19',
            'Mutator30',
        ],
        pattern=r'^Mutator\d+$'
    )


class MutatorWarmupImageset(Imageset[MutatorWarmupAnnotation]):
    manifest_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-warmup/mutator-warmup-manifest.json'
    zipped_images_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-warmup/MutatorWarmup.zip'
    annotation_schema = MutatorWarmupAnnotation


if __name__ == '__main__':

    imageset = MutatorWarmupImageset(redownload=True)
