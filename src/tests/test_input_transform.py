import matplotlib.pyplot as plt
import numpy as np

from src.rl.continuous_cartpole import ContinuousCartPoleEnv
from src.rl.observ_transform import ObservTransformer


def main():
    env = ContinuousCartPoleEnv()
    space = env.observation_space

    n = 10000
    samp_angle = []
    for i in range(n):
        samp = space.sample()
        samp_angle.append(samp[2])
    samp_angle = np.array(samp_angle)
    print(samp_angle.min(), samp_angle.max())

    # run env
    tf = ObservTransformer(env.observation_space, "sincos")
    angle_hist, c, s = [], [], []
    old_observ = env.reset()
    old_observ = tf.transform_observ(old_observ)
    while True:
        angle_hist.append(old_observ[2])
        c.append(old_observ[4])
        s.append(old_observ[5])
        action = np.random.random(1) * 2 - 1
        new_observ, reward, done, _ = env.step(action)
        new_observ = tf.transform_observ(new_observ)
        if done:
            break
        old_observ = new_observ
    plt.plot(np.arange(len(angle_hist)), angle_hist)
    plt.plot(np.arange(len(angle_hist)), c)
    plt.plot(np.arange(len(angle_hist)), s)
    plt.show()

    # test epsilon schedulers


if __name__ == '__main__':
    main()
