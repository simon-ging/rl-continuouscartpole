import numpy as np
from gym import Env
from tqdm import tqdm

from src.rl.continuous_cartpole import ContinuousCartPoleEnv
from src.utils.stats import Stats


def main():
    env = ContinuousCartPoleEnv()
    policy_fn = RandomPolicy(env).get_action

    run_env(env, policy_fn, do_print=True, do_render=True)
    env.close()
    print("Done running environment with rendering\n")

    n_runs = 500
    max_steps = 500
    print("Calculate total rewards for {} runs...".format(n_runs))

    discount_factor = 1.
    total_rewards = []
    for n in tqdm(range(n_runs)):
        experience = run_env(env, policy_fn, max_steps=max_steps)
        len_experience = len(experience)
        observs, actions, rewards, is_dones = zip(*experience)
        rewards = np.array(rewards)
        discounted_rewards = rewards * get_discount_array(
            len_experience, discount_factor)
        total_reward = discounted_rewards.sum()
        total_rewards.append(total_reward)

    # get statistics of total rewards per run
    print("Rewards:")
    print(Stats(total_rewards))


def get_discount_array(len_array, discount_factor):
    """ Returns 1D array of shape (Len_array) with discount factors per step,
        element 0: 1.0
        element 1: discount_factor
        element 2: discount_factor ^ 2
        ...
        element k: discount_factor ^ k
    """
    return (np.ones(len_array) * discount_factor) ** np.arange(len_array)


class RandomPolicy(object):
    """Random Policy: Sample action from environment action space.
    """

    def __init__(self, env: Env):
        self.env = env

    def get_action(self, _):
        return self.env.action_space.sample()


def run_env(env: Env, policy, max_steps=500, do_print=False,
            do_render=False):
    """Run given environment and policy function"""
    old_observ = env.reset()
    total_reward = 0
    experience = []
    for i in range(max_steps):
        if do_render:
            env.render()
        action = policy(old_observ)
        new_observ, reward, done, _ = env.step(action)
        total_reward += reward
        experience.append((old_observ, action, reward, done))
        old_observ = new_observ
        if do_print:
            print("{:4}/{:4} S: {} A: {} R: {}".format(
                i, max_steps, old_observ, action, reward))
        if done:
            break
    if do_print:
        print("Last observation: {}".format(old_observ))
        print("Total reward: {}".format(total_reward))

    return experience


if __name__ == '__main__':
    main()
