"""Configurable Inception v3 model builder."""

from __future__ import annotations

from torch import nn
from torchvision import models

from fl_baselines.core.config import ExperimentConfig


class InceptionBuilder:
    name = "inception"

    def build_model(self, config: ExperimentConfig) -> nn.Module:
        if config.input_height < 75 or config.input_width < 75:
            raise ValueError("Inception requires input height and width >= 75")

        model = models.inception_v3(
            weights=None,
            aux_logits=False,
            num_classes=config.num_classes,
            init_weights=False,
        )
        if config.input_channels != 3:
            first_conv = model.Conv2d_1a_3x3.conv
            model.Conv2d_1a_3x3.conv = nn.Conv2d(
                config.input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
        return model
