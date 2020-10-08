"""Map config strings to objects for environments and reward functions"""

import json

import gym

from src.rl.continuous_cartpole import ContinuousCartPoleEnv
from src.rl.reward_functions import reward_default, reward_shaped_v1, \
    reward_shaped_v2, new_reward_v1


ENVS = {
    "default_cartpole": ContinuousCartPoleEnv
}

REWARD_FUNCTIONS = {
    "none": None,
    "default": reward_default,
    "shaped_v1": reward_shaped_v1,  # unused
    "shaped_v2": reward_shaped_v2,
    "new_v1": new_reward_v1  # unused
}


def load_experiment_config(config_json):
    with open(str(config_json), "rt", encoding="utf8") as fh:
        config = json.load(fh)
    return config


def get_env(name):
    env = ENVS.get(name)
    if env is None:
        env = gym.make(name)
    return env


def get_reward_fn(name):
    return REWARD_FUNCTIONS[name]
