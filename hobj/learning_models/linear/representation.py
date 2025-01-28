import numpy as np

import hobj.data.schema as schema
from typing import Union, Dict, Callable, List
import PIL.Image
from tqdm import tqdm

class RepresentationalModel:
    """
    Class which maps images to feature vectors of shape (d,).
    """

    def __init__(
            self,
            d: int,
            image_to_features_func: Callable[[schema.ImageRef], np.ndarray],
    ):

        if not isinstance(d, int):
            raise ValueError(f"Expected d to be an int, but got {d} of type {type(d)}")
        if d <= 0:
            raise ValueError(f"Expected d to be positive, but got {d}")

        self._d = d
        self._image_to_features_func = image_to_features_func
        return

    @property
    def d(self) -> int:
        return self._d

    def get_features(
            self,
            image: Union[schema.ImageRef, PIL.Image]
    ) -> np.ndarray:
        """
        Returns a feature vector for the image_url.
        To work successfully with UpdateRule, the feature vector must be of shape (d,) and have norm 1.

        :param image: schema.ImageRef or PIL.Image
        :return: np.ndarray, shape=(d,)
        """

        if isinstance(image, PIL.Image.Image):
            image = schema.ImageRef.from_image(image=image)

        f = self._image_to_features_func(image)

        if f.shape != (self.d,):
            raise ValueError(f"Expected feature vector of shape {(self.d,)}, but got {f.shape}")

        return f

    @classmethod
    def from_precomputed_features(
            cls,
            image_ref_to_features: Dict[schema.ImageRef, np.ndarray]
    ) -> 'RepresentationalModel':

        """
        Convenience method for creating a RepresentationalModel, when one has a precomputed dict which maps ImageRefs to np.ndarray feature vectors.

        If get_features is called with an ImageRef (or PIL.Image with an ImageRef) not in image_ref_to_features, a KeyError will be raised.
        """
        raise NotImplementedError