import numpy as np


def angle_normalize(x):
    """Makes sure all values of the input array are in the range
    [-pi, pi), by adding or subtracting 2*pi until they are.
    Copied from continuous cartpole env.
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward_default(cart_pole):
    if (cart_pole.state[0] < - cart_pole.x_threshold or cart_pole.state[0] >
            cart_pole.x_threshold):
        return -1
    return 1 if -0.1 <= angle_normalize(
        cart_pole.state[2]) <= 0.1 else 0


def reward_shaped_v1(cart_pole):
    # -1000 for falling off
    if (cart_pole.state[0] < - cart_pole.x_threshold or cart_pole.state[0] >
            cart_pole.x_threshold):
        return -1000
    # +10000 - very high reward for being in the good range
    ang_norm = angle_normalize(cart_pole.state[2])
    if -0.1 <= ang_norm <= 0.1:
        return 1000
    else:
        # shaped reward for going up on either side
        # up factor should be 1 when up and 0 when down
        up_factor = 1 - np.abs(ang_norm) / np.pi
        return up_factor * 2 - 1


def reward_shaped_v2(cart_pole):
    # -1 for falling off
    if (cart_pole.state[0] < - cart_pole.x_threshold or cart_pole.state[0] >
            cart_pole.x_threshold):
        return -1
    # +1000 - very high reward for being in the good range
    ang_norm = angle_normalize(cart_pole.state[2])
    if -0.1 <= ang_norm <= 0.1:
        return 100
    else:
        # shaped reward for going up on either side
        # up factor should be 1 when up and 0 when down
        up_factor = 1 - np.abs(ang_norm) / np.pi
        return (np.exp(up_factor) - 1) / (np.e - 1)


def new_reward_v1(cart_pole):
    return +1
