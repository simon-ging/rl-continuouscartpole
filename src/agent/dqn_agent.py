"""DQN Agent class and creator method"""

import random

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from src.agent.base_agent import AbstractAgent
from src.agent.replay_buffer import ReplayMemory
from src.agent.dqn_model import create_network
from src.opt.optimizers import get_optimizer
from src.utils.action_quantization import Quantizer1D
from src.utils.torchutils import get_device


def create_dqn_agent(
        config_agent: dict, config_optimizer: dict,
        observ_size: int, action_size: int, action_dim: int, batch_size: int,
        discount_factor: float, action_quantizer: Quantizer1D,
        use_cuda: bool = True, verbose: bool = True):
    """Given configuration dictionaries and parameters, create networks,
    optimizer and DQN agent."""
    # create networks
    device = get_device(use_cuda)
    config_net = config_agent["net"]
    net_online = create_network(
        observ_size, action_size, action_dim, config_net, verbose=verbose).to(
        device)
    net_target = create_network(
        observ_size, action_size, action_dim, config_net, verbose=verbose).to(
        device)

    # create optimizer
    optimizer = get_optimizer(config_optimizer, net_online.parameters())

    # create agent
    agent = RainbowDQNAgent(
        observ_size, action_size, action_dim, action_quantizer, net_online,
        net_target,
        optimizer,
        buffer_size=config_agent["buffer_size"],
        batch_size=batch_size,
        discount_factor=discount_factor,
        tau=config_agent["tau"],
        update_every=config_agent["update_every"],
        use_cuda=use_cuda,
        use_double_q_loss=config_agent["use_double_q"]
    )
    return agent


class RainbowDQNAgent(AbstractAgent):
    """
    DQN Agent to act and learn in a gym environment with some rainbow
    implementations completed (paper arxiv 1710.02298):
        - double q learning: yes
        - prioritized replay: no
        - dueling networks: yes
        - multi-step learning: no
        - distributional rl: no
        - noisy nets: no
    """

    def __init__(
            self, observ_size: int, action_size: int, action_dim: int,
            action_quantizer: Quantizer1D, net_online: nn.Module,
            net_target: nn.Module, optimizer: Optimizer,
            buffer_size: int = 100000, batch_size: int = 64,
            discount_factor: float = .99, tau: float = 1e-3,
            update_every: int = 4, use_cuda: bool = True,
            use_double_q_loss: bool = True):
        """
        Args:
            observ_size: size of observation space
            action_size: number of actions
            action_quantizer: quantizer object to quantize continuous actions
            net_online: online network (used to act)
            net_target: target network
            optimizer: optimizer to update online network
            buffer_size: size of the replay buffer
            batch_size: batch size
            discount_factor: discount (gamma)
            tau: parameter for soft update of target network
            update_every: how often to update target network
            use_cuda: use gpu if available
            use_double_q_loss: use double q learning loss instead of normal
                q learning loss
        """
        self.use_double_q_loss = use_double_q_loss  # type:bool
        self.observ_size = observ_size  # type: int
        self.action_size = action_size  # type: int
        self.action_dim = action_dim  # type: int
        self.action_quantizer = action_quantizer  # type: Quantizer1D
        self.net_online = net_online  # type: nn.Module
        self.net_target = net_target  # type: nn.Module
        self.optimizer = optimizer  # type: Optimizer
        self.buffer_size = buffer_size  # type: int
        self.batch_size = batch_size  # type: int
        self.discount_factor = discount_factor  # type: float
        self.tau = tau  # type: float
        self.update_every = update_every  # type: int
        self.use_cuda = use_cuda # type: bool
        self.device = get_device(use_cuda)  # type: str

        self.is_training = True
        self.loss_fn = nn.MSELoss()

        # create replay buffer
        self.memory = ReplayMemory(
            self.buffer_size, self.action_size, self.batch_size,
            self.action_quantizer, use_cuda=use_cuda)

        # start in training mode
        self.set_train()

    def set_eval(self):
        self.net_online.eval()
        self.net_target.eval()
        self.is_training = False

    def set_train(self):
        self.net_online.train()
        self.net_target.train()
        self.is_training = True

    def get_state_dict(self):
        """Return state dict required for saving."""
        return {
            "model": self.net_online.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }

    def set_state_dict(self, agent_dict):
        """Set state dict from given dictionary"""
        self.net_online.load_state_dict(agent_dict["model"])
        self.optimizer.load_state_dict(agent_dict["optimizer"])

    def step(self, observ, action, reward, next_observ, done, total_step):
        """One optimization step of the agent."""
        if not self.is_training:
            raise ValueError(
                "Agent is set to evaluation mode but .step() has been called.")
        # update replay buffer
        self.memory.add(observ, action, reward, next_observ, done)

        if (total_step + 1) % self.update_every != 0:
            # this is not a learn step
            return

        if len(self.memory.buffer) <= self.batch_size:
            # not enough memory in the replay buffer
            return

        # ---------- learning ----------
        experiences = self.memory.sample()

        observs, actions, rewards, next_observs, dones = experiences

        # dirty hack for handling Box shape batch, 1 and Discrete shape batch
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)

        if not self.use_double_q_loss:
            # regular loss
            net_target_out = self.net_target(next_observs)
            q_targets_next = net_target_out.detach()
            q_targets_next = q_targets_next.max(-1)[0]
            q_targets_next = q_targets_next.unsqueeze(-1)
        else:
            # double q loss
            q_local_argmax = self.net_online(
                next_observs).detach().max(1)[1].unsqueeze(1)
            target_out = self.net_target(next_observs)
            q_targets_next = target_out.gather(1, q_local_argmax)

        # compute q targets for current observations
        q_targets = rewards + (self.discount_factor * q_targets_next * (
                1 - dones))

        # compute expect q for current observations
        # print("actions", actions.shape, actions.dtype)
        actions = torch_base2int(
            actions, self.action_size,  self.action_dim,
            use_cuda=self.use_cuda)
        actions = actions.unsqueeze(-1)
        # print("actions", actions.shape)
        local_out = self.net_online(observs)
        # print("local out", local_out.shape)
        q_expected = local_out.gather(1, actions)
        # print("expected", q_expected.shape)

        # td-Loss (MSE)
        loss = self.loss_fn(q_expected, q_targets)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        for target_param, local_param in zip(
                self.net_target.parameters(), self.net_online.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau)
                * target_param.data)

    def act(self, observ, eps=0.):
        """Act given some observation and epsilon"""
        # check epsilon-greedy action selection
        if random.random() > eps:
            # model action (no gradients, eval mode)
            observ = np.expand_dims(np.array(observ, dtype=np.float32), axis=0)
            observ = torch.from_numpy(observ).to(self.device)
            self.net_online.eval()
            with torch.no_grad():
                action_values = self.net_online(observ)
            self.net_online.train()
            action = np.argmax(action_values.cpu().data.numpy())
            # base DIM representation gives back the action
            action = np_int2base(
                action, self.action_size, size=self.action_dim)
        else:
            # random action
            action = np.random.choice(
                self.action_size, size=self.action_dim)

        if self.action_quantizer is not None:
            # environment expects continuous actions
            # dequantize action (int to float)
            action = self.action_quantizer.int_to_float(action)
            # env expects action to be a length 1 numpy array
            return action
            # return np.array([action])
        else:
            # these envs expect no dimension i guess?
            # better would be some flag on what to actually return
            return np.squeeze(action,axis=-1)

def np_int2base(x, base, size=None, decreasing=True):
    x = np.asarray(x)
    if size is None:
        size = int(np.ceil(np.log(np.max(x)) / np.log(base)))
    if decreasing:
        powers = base ** np.arange(size - 1, -1, -1)
    else:
        powers = base ** np.arange(size)
    digits = (x.reshape(x.shape + (1,)) // powers) % base
    return digits


def np_base2int(x_list, base, size=None, decreasing=True):
    x_list = np.asarray(x_list)
    if size is None:
        size = x_list.shape[-1]
    if decreasing:
        powers = base ** np.arange(size - 1, -1, -1)
    else:
        powers = base ** np.arange(size)
    contents = x_list * powers
    contents = contents.sum(axis=-1)
    return contents


def torch_int2base(x, base, size=None, decreasing=True, use_cuda=False):
    device = get_device(use_cuda)
    if size is None:
        size = int(torch.ceil(torch.log(torch.max(x)) / torch.log(base)))
    if decreasing:
        powers = base ** torch.arange(size - 1, -1, -1).float().to(device)
    else:
        powers = base ** torch.arange(size).float().to(device)
    digits = (x.reshape(x.shape + (1,)) // powers) % base
    return digits


def torch_base2int(x_list, base, size=None, decreasing=True, use_cuda=False):
    device = get_device(use_cuda)
    if size is None:
        size = x_list.shape[-1]
    if decreasing:
        powers = base ** torch.arange(size - 1, -1, -1).long().to(device)
    else:
        powers = base ** torch.arange(size).long().to(device)
    contents = x_list * powers
    contents = contents.sum(dim=-1)
    return contents