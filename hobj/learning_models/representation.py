from typing import Callable, Dict

import numpy as np

from hobj.types import ImageId


# %%
class RepresentationalModel:
    """
    Class which maps images to feature vectors of shape (d,).
    """

    def __init__(
            self,
            d: int,
            image_to_features_func: Callable[[ImageId], np.ndarray],
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
            image: ImageId
    ) -> np.ndarray:
        """
        Returns a feature vector for the image_url.
        To work successfully with UpdateRule, the feature vector must be of shape (d,) and have norm 1.

        :param image: schema.ImageRef or PIL.Image
        :return: np.ndarray, shape=(d,)
        """

        f = self._image_to_features_func(image)

        if f.shape != (self.d,):
            raise ValueError(f"Expected feature vector of shape {(self.d,)}, but got {f.shape}")

        return f

    @classmethod
    def from_precomputed_features(
            cls,
            image_ref_to_features: Dict[ImageId, np.ndarray]
    ) -> 'RepresentationalModel':

        """
        Convenience method for creating a RepresentationalModel, when one has a precomputed dict which maps ImageRefs to np.ndarray feature vectors.

        If get_features is called with an ImageRef (or PIL.Image with an ImageRef) not in image_ref_to_features, a KeyError will be raised.
        """

        def image_to_features_func(image: ImageId) -> np.ndarray:
            return image_ref_to_features[image]

        # Ensure all feature vectors are the same shape
        d = None
        for ref in image_ref_to_features:
            f = image_ref_to_features[ref]
            if not isinstance(f, np.ndarray):
                raise ValueError(f"Expected feature vector to be a np.ndarray, but got {f} of type {type(f)}")
            if not len(f.shape) == 1:
                raise ValueError(f"Expected feature vector to be 1D, but got {f.shape}")

            if d is None:
                d = f.shape[0]

            if not f.shape[0] == d:
                raise ValueError(f"Expected feature vector to be of shape ({d},), but got {f.shape}")

        return cls(d=d, image_to_features_func=image_to_features_func)
