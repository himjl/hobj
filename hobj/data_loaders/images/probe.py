
from typing import Literal

import pydantic

from hobj.data_loaders.images.template import Imageset


class ProbeAnnotation(pydantic.BaseModel):
    color: Literal['blue', 'orange']
    text: Literal['press right', 'press left']


class ProbeImageset(Imageset[ProbeAnnotation]):
    """
    An imageset consisting of two images: a
    """
    annotation_schema = ProbeAnnotation

# %%
if __name__ == '__main__':
    imageset = ProbeImageset()
