"""Quantize (map) between continuous and discrete action space"""

import numpy as np

from gym.spaces import Box, Space, Discrete

from typing import List


class Quantizer1D(object):
    """Works on numpy objects and quantizes/dequantizes actions.
    For simplicity only works with action spaces of length 1.
    """

    def __init__(self, space: Box, num_quant: int):
        self.space = space
        self.num_quant = num_quant
        if type(space) is Box:
            if self.space.shape != (1,):
                print(self.space)
                raise NotImplementedError("action spaces > 1 not supported.")
            self.high = self.space.high[0]
            self.low = self.space.low[0]
        elif type(space) is Discrete:
            raise ValueError("No quantizer for discrete action space needed")

        self.space_int_to_float = np.linspace(
            self.low, self.high, num=num_quant, endpoint=True)

    def int_to_float(self, action_int):
        """Convert integer action to float action

        Args:
            action_int (int): quantized action

        Returns:
            float action
        """
        return self.space_int_to_float[action_int]

    def float_to_int(self, action_float):
        """Convert float action to int action

        Args:
            action_float (float): float action

        Returns:
            quantized action
        """
        assert self.low <= action_float <= self.high, (
            "action {} out of range [{}, {}]".format(
                action_float, self.low, self.high))
        # scale to [0, 1]
        flt = (action_float - self.low) / (self.high - self.low)
        # scale to [0, num_quant-1]
        flt *= (self.num_quant - 1)
        # round to integer
        action_int = np.round(flt).astype(np.int)
        return action_int


class QuantizerND(object):
    """Extension to work with arbitary boxes
    """

    def __init__(self, space: Box, num_quant: int):
        self.space = space
        self.num_quant = num_quant
        if type(space) is Box:
            assert len(self.space.shape) == 1, "multi-dim actions not support"
            self.highs = self.space.high
            self.lows = self.space.low
        elif type(space) is Discrete:
            raise ValueError("No quantizer for discrete action space needed")
        self.dim_action = self.space.shape[0]
        print("NUM QUANT", num_quant)
        self.lows = np.array([-4, -6, -9, -1])

        self.spaces_int_to_float = np.linspace(
            self.lows, self.highs, num=num_quant, endpoint=True)

    def int_to_float(self, action_int):
        """Convert integer action to float action

        Args:
            action_int (int): quantized action

        Returns:
            float action
        """
        action_float = np.diag(self.spaces_int_to_float[action_int, :])
        return action_float

    def float_to_int(self, action_float):
        """Convert float action to int action

        Args:
            action_float (float): float action

        Returns:
            quantized action
        """
        # assert self.low <= action_float <= self.high, (
        #     "action {} out of range [{}, {}]".format(
        #         action_float, self.low, self.high))

        # scale to [0, 1]

        flt = (action_float - self.lows) / (self.highs - self.lows)
        # scale to [0, num_quant-1]
        flt *= (self.num_quant - 1)
        # round to integer
        action_int = np.round(flt).astype(np.int)
        return action_int


def test_quantizer():
    # create box
    box = Box(np.array([-1]), np.array([1]))
    # create quantizer
    num_quant = 16
    q = Quantizer1D(box, num_quant)

    # quantize int to float
    for n in range(num_quant):
        print("{} -> {:.3f}".format(n, q.int_to_float(n)))

    # quantize float to int
    for flt in np.linspace(-1, 1, 50):
        print("{:.3f} -> {}".format(flt, q.float_to_int(flt)))
