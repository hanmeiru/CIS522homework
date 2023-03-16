"""
This class defines the architecture of the neural net
"""
import torch
from torch import nn


class Model(torch.nn.Module):
    """
    A convolutinal neural net that
    consists of two convolutional layers,
    each followed by relu; one maxpool layer; and two dense layers
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Initializes the layers of the neural net
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(
            196 * 16, 32
        )  # (32-4*2)/2 for width and height, 32 filters
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passing the input image to the model and get output
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)  # (8,2304)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
