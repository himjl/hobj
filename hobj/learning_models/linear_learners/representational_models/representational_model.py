import numpy as np


class RepresentationalModel(object):
    """
    Model which represents image_urls as feature vectors of shape (d,).
    Meant to be subclassed (see PrecachedRepresentationalModel below for an example).
    """
    def __init__(self, representational_model_id:str):
        self.representational_model_id = representational_model_id
        self._d = 1
        return

    def reset(self):
        """
        Resets the model to its initial state. Not used by all representational models (e.g., fixed intermediate layers of DCNNs), but
        included here for completeness.
        :return: None
        """
        return


    @property
    def d(self):
        """
        Returns the dimension of the feature vectors.
        :return:
        """
        return self._d

    @d.setter
    def d(self, value):
        assert isinstance(value, int)
        assert value > 0
        self._d = value

    def get_features(self, image_url: str):
        """
        Returns a feature vector for the image_url.
        To work successfully with UpdateRule, the feature vector must be of shape (d,) and have norm 1.
        :param image_url: str, url of the image
        :return: np.ndarray, shape=(d,)
        """
        return np.zeros(self.d)


class PrecachedRepresentationalModel(RepresentationalModel):

    def __init__(self, url_to_features: dict, representational_model_id: str):

        self.url_to_features = url_to_features
        super().__init__(representational_model_id=representational_model_id)
        self.d = len(url_to_features[list(url_to_features.keys())[0]])

    def get_features(self, image_url: str):
        return self.url_to_features[image_url]