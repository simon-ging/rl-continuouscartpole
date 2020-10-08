"""Test whether standard deviation calculation is correct"""
import numpy as np
from src.utils.stats import Stats


def main():
    data = [9, 2, 5, 4, 12, 7, 8, 11, 9, 3, 7, 4, 12, 5, 4, 10, 9, 6, 9, 4]
    true_var = 9.368
    true_std = 3.061
    print("***** TRUE")
    print("std", true_std)
    print("var", true_var)

    # 1D arrays
    s = Stats(data)
    print("***** STATS")
    print("std", s.stddev)
    print("var", s.var)

    # n-D
    print("***** N-D")
    data = np.array(data)
    data = np.expand_dims(data, 0)
    data = np.repeat(np.expand_dims(data, -1), 3, axis=-1)
    print(data.shape)
    s = Stats(data, axis=1)

    print("std", s.stddev, s.stddev.shape)
    print("var", s.var, s.var.shape)


if __name__ == '__main__':
    main()
