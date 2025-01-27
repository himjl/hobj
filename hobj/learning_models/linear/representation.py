import numpy as np

import hobj.data.schema as schema
from typing import Union, Dict
import PIL.Image

from abc import ABC, abstractmethod


class RepresentationalModel(ABC):
    """
    Model which represents image_urls as feature vectors of shape (d,).
    Meant to be subclassed (see PrecachedRepresentationalModel below for an example).
    """

    def __init__(self, d: int):
        if not isinstance(d, int):
            raise ValueError(f"Expected d to be an int, but got {d} of type {type(d)}")
        if d <= 0:
            raise ValueError(f"Expected d to be positive, but got {d}")

        self.d = d
        return

    @abstractmethod
    def get_features(
            self,
            image: Union[schema.ImageRef, PIL.Image]
    ) -> np.ndarray:
        """
        Returns a feature vector for the image_url.
        To work successfully with UpdateRule, the feature vector must be of shape (d,) and have norm 1.
        :param image_url: str, url of the image
        :return: np.ndarray, shape=(d,)
        """
        return np.zeros(self.d)

