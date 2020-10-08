"""Networks for DQN"""

from typing import List

import torch.nn as nn
from torch.nn import functional as F
import torch


def create_network(observ_size, action_size, action_dim, config_net,
                   verbose=True) \
        -> nn.Module:
    net_name = config_net["name"]
    if net_name == "qnet1":
        net = DQNModule(
            observ_size, action_size, action_dim, config_net["nonlinearity"],
            config_net["encoder_fc_units"], config_net["batchnorm"],
            config_net["use_dueling"], verbose=verbose)
    else:
        raise NotImplementedError("network {} not implemented".format(
            net_name))
    return net


class DQNModule(nn.Module):
    """Use DQNEncoder to encode observation to hidden state, then encode
    hidden state to action (Q-values)"""

    def __init__(
            self, observ_size=4, action_size=2, action_dim=1,
            nonlinearity="selu", encoder_fc_units=(64, 64), batchnorm=False,
            use_dueling=True, verbose=True):
        """

        Args:
            observ_size: size of observation space
            action_size: size of action space
            nonlinearity: "relu", "selu" or "leakyrelu" for the encoder
            encoder_fc_units: list of neurons per linear layer for the encoder
            batchnorm: use batchnorm in the encoder
            use_dueling: networks
            verbose: print network graph
        """
        super().__init__()

        # setup encoder to map from observation space to hidden space
        self.net_encoder = DQNEncoderModule(
            observ_size, nonlinearity, encoder_fc_units, batchnorm)

        # get size of hidden state vector
        hidden_size = encoder_fc_units[-1]
        if verbose:
            print("---------- network ----------")
            print("Encoder: {}".format(self.net_encoder))

        # map from encoding to output
        self.use_dueling = use_dueling
        self.action_size = action_size
        self.action_dim = action_dim
        if not self.use_dueling:
            # no dueling networks - simply map from hidden space to action
            # with a linear layer
            self.net_output = nn.Linear(hidden_size, action_size ** action_dim)
            if verbose:
                print("Output: {}".format(self.net_output))
        else:
            # ---------- dueling networks ----------
            # state-value network
            self.state_value_stream = nn.Linear(hidden_size, 1)
            # action advantage network
            self.advantage_stream = nn.Linear(
                hidden_size, action_size ** action_dim)
            if verbose:
                print("State value stream: {}".format(
                    self.state_value_stream))
                print("Advantage stream: {}".format(self.advantage_stream))

    def forward(self, x):
        # encode observation space to hidden space
        x = self.net_encoder(x)

        if not self.use_dueling:
            # no dueling, compute q values
            x = self.net_output(x)
            # print("net out", x.shape)
            # x = torch.reshape(x, [-1] + [self.action_size for _ in
            #                              range(self.action_dim)])
            # print("net out reshaped", x.shape)
            return x

        # ---------- dueling networks forward pass ----------
        state_value = self.state_value_stream(x)  # shape (batch, 1)
        advantage_value = self.advantage_stream(x)  # shape (batch, n_actions)
        advantage_sum = 1 / (self.action_size ** self.action_dim) * (
            advantage_value.sum(-1)).unsqueeze(1)  # shape (batch, 1)
        total_sum = state_value + advantage_value - advantage_sum
        return total_sum


class DQNEncoderModule(nn.Module):
    """Encode observation space to hidden space. Network graph:
    observation
        -> [linear, nonlinearity, optional, batchnorm] * N
        -> hidden state

    Args:
        observ_size (int): Dimension of observations
        nonlinearity (str): nonlinearity "relu", "selu" or "leakyrelu"
        encoder_fc_units (List[int]): list of neurons per linear layer
        batchnorm (bool): Use batchnorm

    """

    def __init__(
            self, observ_size: int, nonlinearity: str,
            encoder_fc_units: List[int], batchnorm: bool):
        """Initialize parameters and build model.

        """
        super().__init__()

        # define nonlinearity class
        if nonlinearity == "selu":
            nonlin_class = nn.SELU
        elif nonlinearity == "relu":
            nonlin_class = nn.ReLU
        elif nonlinearity == "leakyrelu":
            nonlin_class = nn.LeakyReLU
        else:
            raise ValueError("Nonlinearity {} not found".format(nonlinearity))

        # define batchnorm class
        def bn_class(_):
            return nn.Sequential()

        if batchnorm:
            bn_class = nn.BatchNorm1d

        # loop over the encoder num units list and create the network
        modules = []
        in_shape = observ_size
        for num_fc_units in encoder_fc_units:
            modules += [
                nn.Linear(in_shape, num_fc_units), nonlin_class(),
                bn_class(num_fc_units)]
            in_shape = num_fc_units
        self.encoder = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.encoder:
            x = module(x)
        return x
