
from hobj.data_loaders.images.template import Imageset
import pydantic
from typing import Literal


class ProbeAnnotation(pydantic.BaseModel):
    color: Literal['blue', 'orange']
    text: Literal['press right', 'press left']


class ProbeImageset(Imageset[ProbeAnnotation]):
    """
    An imageset consisting of two images: a
    """
    manifest_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/probe-images/probe-images-manifest.json'
    zipped_images_url = 'https://hlbdatasets.s3.us-east-1.amazonaws.com/imagesets/probe-images/probe-images.zip'
    annotation_schema = ProbeAnnotation

# %%
if __name__ == '__main__':
    imageset = ProbeImageset()
