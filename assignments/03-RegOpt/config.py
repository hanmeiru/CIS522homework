"""
Hyperparameter configs for the model
"""
from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor


class CONFIG:
    batch_size = 64
    num_epochs = 12
    initial_learning_rate = 0.003
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ToTensor(),
            Normalize(0.5, 0.5),
        ]
    )
