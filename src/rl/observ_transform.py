"""Sincos transform for the observation space"""

import numpy as np

from gym.spaces import Box


class ObservTransformer(object):
    """Transform observation space (sincos-transform)"""

    def __init__(self, observ_space: Box, transform_type="none"):
        self.transform_type = transform_type
        if self.transform_type == "sincos":
            # add fields cos, sin of theta
            high = np.concatenate([observ_space.high, [1., 1.]])
            low = np.concatenate([observ_space.low, [-1., -1.]])
            self.observ_space = Box(low, high, dtype=np.float32)
        elif self.transform_type == "none":
            # leave input space as is
            self.observ_space = observ_space
        else:
            raise ValueError("Input Transform {} not recognized.".format(
                self.transform_type))

        # save original input space
        self.observ_space_orig = observ_space
        # print("Setup input transformer, from {} to {}".format(
        #     self.input_space_orig.shape, self.input_space.shape))

    def transform_observ(self, observ):
        """Transform given observation"""
        assert observ.shape == self.observ_space_orig.shape
        if self.transform_type == "sincos":
            theta = observ[2]
            observ = np.concatenate(
                [observ, [np.cos(theta), np.sin(theta)]])
        return observ
