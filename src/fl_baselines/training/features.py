"""Feature extraction helpers for algorithms that regularize representation space."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fl_baselines.core.config import ExperimentConfig
from fl_baselines.models.lenet import LeNet
from fl_baselines.models.mnist_cnn import MnistCnn
from fl_baselines.models.resnet import ResNet9


def extract_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Return a 2D embedding tensor without changing model forward contracts."""

    if isinstance(model, MnistCnn):
        x = model.pool(F.relu(model.conv1(inputs)))
        x = model.pool(F.relu(model.conv2(x)))
        x = torch.flatten(x, 1)
        return model.dropout(F.relu(model.fc1(x)))

    if isinstance(model, LeNet):
        x = model.pool(F.relu(model.conv1(inputs)))
        x = model.pool(F.relu(model.conv2(x)))
        x = model.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(model.fc1(x))
        return F.relu(model.fc2(x))

    if isinstance(model, ResNet9):
        x = model.features(inputs)
        return torch.flatten(x, 1)

    if hasattr(model, "fc") and hasattr(model, "layer4"):
        x = model.conv1(inputs)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return torch.flatten(x, 1)

    if hasattr(model, "fc") and hasattr(model, "Mixed_7c"):
        x = model.Conv2d_1a_3x3(inputs)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = model.maxpool1(x)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = model.maxpool2(x)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        x = model.Mixed_7c(x)
        x = model.avgpool(x)
        x = model.dropout(x)
        return torch.flatten(x, 1)

    if isinstance(model, nn.Linear):
        return inputs.reshape(inputs.shape[0], -1)

    raise ValueError(f"Feature extraction is not implemented for model type: {type(model).__name__}")


def logits_from_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Project extracted features back to logits for supported models."""

    if isinstance(model, MnistCnn):
        return model.fc2(features)

    if isinstance(model, LeNet):
        return model.fc3(features)

    if isinstance(model, ResNet9):
        return model.classifier(features)

    if hasattr(model, "fc") and hasattr(model.fc, "weight"):
        return model.fc(features)

    if isinstance(model, nn.Linear):
        return model(features)

    raise ValueError(f"Logit projection is not implemented for model type: {type(model).__name__}")


def classifier_weight_matrix(model: nn.Module) -> torch.Tensor:
    """Return the final classifier weight matrix used by the current model."""

    if isinstance(model, MnistCnn):
        return model.fc2.weight

    if isinstance(model, LeNet):
        return model.fc3.weight

    if isinstance(model, ResNet9):
        return model.classifier.weight

    if hasattr(model, "fc") and hasattr(model.fc, "weight"):
        return model.fc.weight

    if isinstance(model, nn.Linear):
        return model.weight

    raise ValueError(
        f"Classifier weight extraction is not implemented for model type: {type(model).__name__}"
    )


def infer_feature_dim(
    model: nn.Module,
    config: ExperimentConfig,
) -> int:
    """Infer embedding dimension for the current model using a dummy input."""

    dummy = torch.zeros(1, *config.input_shape, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        features = extract_features(model, dummy)
    return int(features.shape[1])
