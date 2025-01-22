from hobj.data.images.template import Imageset, ImageManifestEntry, ImageManifest
import pydantic


class WarmupAnnotation(pydantic.BaseModel):
    category: str = pydantic.Field(
        examples=[
            'Mutator19',
            'Mutator30',
        ],
        pattern=r'^Mutator\d+$'
    )


class WarmupImageset(Imageset[WarmupAnnotation]):
    manifest_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-warmup/mutator-warmup-manifest.json'
    zipped_images_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/mutator-warmup/MutatorWarmup.zip'
    annotation_schema = WarmupAnnotation


if __name__ == '__main__':

    imageset = WarmupImageset(redownload=True)
