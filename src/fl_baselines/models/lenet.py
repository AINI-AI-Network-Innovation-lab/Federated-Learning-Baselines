"""Configurable LeNet-style model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fl_baselines.core.config import ExperimentConfig


class LeNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNetBuilder:
    name = "lenet"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        return LeNet(
            input_channels=config.input_channels,
            num_classes=config.num_classes,
        )
