"""Stats class to calculate mean/stddev on the fly"""
import numpy as np


class Stats(object):

    # noinspection PyArgumentList
    def __init__(self, array: np.ndarray, axis: int = -1):
        self.array = np.array(array)
        self.axis = axis
        self.dims = len(self.array.shape)

        self.mean = self.array.mean(axis=self.axis)

        # self.var = self.array.var(axis=self.axis)
        # statistically correct: divide by N-1
        self.var = ((self.array - np.expand_dims(self.mean, axis=self.axis)
                     ) ** 2).sum(self.axis) / (self.array.shape[axis] - 1)
        self.stddev = np.sqrt(self.var)

        self.min_ = self.array.min(axis=self.axis)
        self.max_ = self.array.max(axis=self.axis)

    def __str__(self):
        if self.dims == 1:
            return (
                "Array mean {:.3f} stddev {:.3f} valuerange {:.3f}/{:.3f} "
                "var {:.3f} num_values {} ".format(
                    self.mean, self.stddev, self.min_, self.max_, self.var,
                    self.array.shape))
        else:
            return (
                "Array {} mean {} ...\n"
                "fields: mean, var, stddev, min_, max".format(
                    self.array.shape, self.mean.shape)
            )


def average_last(x, n=100):
    """Calculate average of the axis of the last N elements"""
    assert len(x.shape) == 1, "currently only implemented for length 1 axis"
    axis_length = x.shape[0]
    if axis_length < n:
        # trying to average over a length longer than the actual array dims
        # just use the simple mean of all elements up to this point
        out = np.zeros_like(x)
        for i in range(axis_length):
            out[i] = x[: i + 1].mean()
    else:
        # there are points where we can calculate average for last N points
        # faster version as soon as 100 examples are available
        cumsum = np.cumsum(np.insert(x, np.ones((100,)) * x[0], 0))
        out = (cumsum[n:] - cumsum[:-n]) / float(n)
        # correct the first i as running mean of only the first i
        for i in range(n):
            out[i] = x[:i + 1].mean()
    return out
