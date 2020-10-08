"""Abstract agent class template."""

import numpy as np


class AbstractAgent(object):
    """Agent can act and learn in a gym environment."""

    def step(self, observ, action, reward, next_observ, done, total_step):
        """Do one step, e.g. save experiences and update networks

        Args:
            observ (np.ndarray): current observation
            action (np.ndarray): current action
            reward (float): reward
            next_observ (np.ndarray): next observation
            done (bool): next_observ is terminal or not
            total_step (int): total training step
        """
        raise NotImplementedError

    def act(self, observ, eps=0.):
        """Take observation and act epsilon-greedily.

        Args:
            observ (np.ndarray): observation
            eps (float): epsilon, for epsilon-greedy action selection
        """
        raise NotImplementedError

    def set_eval(self):
        """Set all neural networks to evaluation mode"""
        raise NotImplementedError

    def set_train(self):
        """Set all neural networks to train mode"""
        raise NotImplementedError

    def get_state_dict(self):
        """Return a dictionary with all relevant state dictionaries:
        State dicts of all networks and state dict of the optimizer.
        This dict will be saved to file."""
        raise NotImplementedError

    def set_state_dict(self, agent_dict):
        """Update all relevant state dicts (networks, optimizer) from the input
        dictionary."""
        raise NotImplementedError
