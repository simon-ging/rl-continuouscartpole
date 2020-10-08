import random
from collections import deque

import numpy as np
import torch

from src.utils.action_quantization import Quantizer1D
from src.utils.torchutils import get_device


class ReplayMemory(object):
    """Memory for experience replay"""

    def __init__(
            self, buffer_size: int, action_size: int, batch_size: int,
            action_quantizer: Quantizer1D, use_cuda: bool = True):
        """Create the memory"""
        self.action_size = action_size
        self.batch_size = batch_size
        self.action_quantizer = action_quantizer

        self.buffer = deque(maxlen=buffer_size)
        self.device = get_device(use_cuda)

    def add(self, observ, action, reward, next_observ, done):
        """Append experience tuple to the buffer"""
        if self.action_quantizer is not None:
            # quantize the action
            action = self.action_quantizer.float_to_int(action)
        self.buffer.append((observ, action, reward, next_observ, done))

    def sample(self):
        """Sample experience tuples from buffer"""
        # sample random experience tuples
        experiences = random.sample(self.buffer, k=self.batch_size)

        # collect data from tuples
        observs, actions, rewards, next_observs, dones = [], [], [], [], []
        for e in experiences:
            observs.append(e[0])
            actions.append(e[1])
            rewards.append(e[2])
            next_observs.append(e[3])
            dones.append(e[4])

        # transform data to tensors with correct data types, shapes, devices
        observs = torch.from_numpy(np.array(observs, dtype=np.float32)).to(
            self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.int64)).to(
            self.device)
        rewards = torch.from_numpy(np.expand_dims(np.array(
            rewards, dtype=np.float32), axis=-1)).to(self.device)
        next_observs = torch.from_numpy(
            np.array(next_observs, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.expand_dims(np.array(
            dones, dtype=np.float32), axis=-1)).to(self.device)

        return observs, actions, rewards, next_observs, dones
