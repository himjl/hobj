import numpy as np
from typing import Union, List
import xarray as xr

class Environment(object):

    def __init__(self):
        self._meta = {}
        return

    def initialize(self, RS: np.random.RandomState):
        raise NotImplementedError

    def sample_image(self):
        image_url = ''
        return image_url

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, value):
        assert isinstance(value, dict) or isinstance(value, xr.Dataset)
        self._meta = value


    @property
    def current_state_meta(self):
        # Returns meta information for the current environmental state, that will be logged in a simulation. Not allowed to be used by the learner.
        return {}

    def provide_feedback(self, action: int):
        reward = 0.
        return reward

