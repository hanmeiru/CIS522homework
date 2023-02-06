import torch
import torch.nn as nn
from typing import Callable


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()  # inheriting

        # defining layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.layers += [
            nn.Linear(input_size, hidden_size),
            self.activation,
        ]  # first layer
        hidden_idx = 0  # idx of hidden layers
        while hidden_idx < hidden_count - 1:
            self.layers += [
                nn.Linear(hidden_size, hidden_size),
                self.activation,
            ]  # hidden layers
            hidden_idx += 1
        self.layers += [nn.Linear(hidden_size, num_classes)]  # last layer

        # initialize using specified initializer
        for m in self.layers:
            if type(m) == nn.Linear:
                initializer(m.weight)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """

        # Flatten inputs to 2D (if more than that)
        x = torch.flatten(x, start_dim=1)

        # Get activations of each layer
        for layer in self.layers:
            x = layer(x)

        return x
