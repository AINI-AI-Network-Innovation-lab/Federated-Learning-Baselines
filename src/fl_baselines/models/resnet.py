"""Configurable ResNet model builders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from fl_baselines.core.config import ExperimentConfig


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class ResNet9(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(input_channels, 64),
            ConvBnRelu(64, 128, stride=2),
            ResidualBlock(128),
            ConvBnRelu(128, 256, stride=2),
            ConvBnRelu(256, 512, stride=2),
            ResidualBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_torchvision_resnet(
    factory,
    input_channels: int,
    num_classes: int,
) -> nn.Module:
    model = factory(weights=None, num_classes=num_classes)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )
    return model


class ResNet9Builder:
    name = "resnet9"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        return ResNet9(config.input_channels, config.num_classes)


class ResNet18Builder:
    name = "resnet18"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        return _build_torchvision_resnet(
            models.resnet18,
            config.input_channels,
            config.num_classes,
        )


class ResNet34Builder:
    name = "resnet34"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        return _build_torchvision_resnet(
            models.resnet34,
            config.input_channels,
            config.num_classes,
        )
