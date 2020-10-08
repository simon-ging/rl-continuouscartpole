"""Epsilon decay schedulers: multiplicate, cosine, constant"""
import numpy as np


def get_eps_scheduler(config, max_steps):
    name = config["name"]
    if name == "mult_decay":
        return MultiplicativeEpsilonDecay(
            max_steps, config["eps_start"], config["eps_end"],
            config["eps_decay"], config["warmup"])
    elif name == "cos_decay":
        return CosineEpsilonDecay(
            max_steps, config["eps_start"], config["eps_end"],
            config["warmup"])
    elif name == "none":
        return ConstantEpsilon(config["eps_start"])
    else:
        raise ValueError("Epsilon Scheduler {} unknown".format(
            name))


class MultiplicativeEpsilonDecay(object):
    """Multiplicatively decay epsilon.

    Args:
        max_steps (int): Total number of steps
        eps_start (float): maximum (starting) epsilon value
        eps_end (float): minimum (end) epsilon value, epsilon can never go
            below that
        eps_decay (float): multiplicative decay per episode
        warmup (float): percentage of the episodes to run with start eps
            in [0, 1]
    """

    def __init__(
            self, max_steps, eps_start=1.0, eps_end=0.01, eps_decay=0.9995,
            warmup=0.0):
        self.max_steps = max_steps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.current_epsilon = 0
        self.current_step = 0
        self.warmup_steps = int(max_steps * warmup)

    def reset(self, start_step=0):
        self.current_step = start_step
        step_after_warmup = max(0, self.current_step - self.warmup_steps)
        self.current_epsilon = max(self.eps_end, (
                self.eps_decay ** step_after_warmup) * self.eps_start)
        return self.current_epsilon

    def step(self):
        self.current_step += 1
        if self.current_step >= self.warmup_steps:
            self.current_epsilon = max(
                self.eps_end, self.eps_decay * self.current_epsilon)
        return self.current_epsilon

    def get_epsilon(self):
        return self.current_epsilon


class CosineEpsilonDecay(object):
    """Cosine epsilon decay.

    Args:
        max_steps (int): Total number of steps
        eps_start (float): maximum (starting) epsilon value
        eps_end (float): minimum (end) epsilon value, epsilon can never go
            below that
        warmup (float): percentage of the episodes to run with start eps
            in [0, 1]
    """

    def __init__(
            self, max_steps, eps_start=1.0, eps_end=0.01, warmup=0.0):
        assert 0 <= warmup <= 1, "Epsilon decay warmup must be in " \
                                 "[-1, 1]"
        self.max_steps = max_steps
        self.eps_start = eps_start
        self.eps_end = eps_end

        self.current_epsilon = 0
        self.current_step = 0
        self.warmup_steps = int(max_steps * warmup)

    def reset(self, start_step=0):
        self.current_step = start_step
        self.current_epsilon = self.get_epsilon()
        return self.current_epsilon

    def step(self):
        self.current_step += 1
        self.current_epsilon = self.get_epsilon()
        return self.current_epsilon

    def get_epsilon(self):
        step_after_warmup = max(0, self.current_step - self.warmup_steps)
        total_steps_after_warmup = max(0, self.max_steps - self.warmup_steps)
        cos_factor = (np.cos(np.pi * (
                step_after_warmup / total_steps_after_warmup)) + 1) / 2
        self.current_epsilon = self.eps_end + (
                self.eps_start - self.eps_end) * cos_factor
        return self.current_epsilon


class ConstantEpsilon(object):
    """Constant epsilon"""

    def __init__(self, eps_start=1.0):
        self.eps_start = eps_start

    def reset(self):
        return self.eps_start

    def step(self):
        return self.eps_start

    def get_epsilon(self):
        return self.eps_start
