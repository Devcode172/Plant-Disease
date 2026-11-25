"""AlexNet and EfficientNet-B3 builders (paper-style AADL head)."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


ModelName = Literal["alexnet", "efficientnet_b3"]

# Keras-style regularizer strengths used on the 256-unit dense layer
L1_FACTOR = 0.006
L2_FACTOR = 0.016


class EfficientNetB3AADL(nn.Module):
    """
    Pretrained EfficientNet-B3 + AADL head:
    Conv2d(32, k=2) -> MaxPool(2, stride=2) -> BatchNorm -> Flatten ->
    Linear(256)+ReLU (L1=0.006, L2=0.016) -> Dropout(0.4) -> Linear(num_classes).
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = backbone.features  # (N, 1536, H, W)

        self.conv = nn.Conv2d(1536, 32, kernel_size=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(32)
        # LazyLinear infers flatten width for any input resolution (224/300/smoke 128)
        self.fc256 = nn.LazyLinear(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
        self.fc_out = nn.Linear(256, num_classes)

        self.l1_factor = L1_FACTOR
        self.l2_factor = L2_FACTOR

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc256(x))
        x = self.dropout(x)
        return self.fc_out(x)

    def regularization_loss(self) -> torch.Tensor:
        """L1 (0.006) + L2 (0.016) on the 256-unit dense layer weights."""
        w = self.fc256.weight
        return self.l1_factor * w.abs().sum() + self.l2_factor * w.pow(2).sum()


def build_efficientnet_b3(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    torchvision pretrained EfficientNet-B3 backbone, then
    Conv2d(32, kernel 2) + MaxPool(2, stride 2), BatchNorm, Flatten,
    Linear(256)+ReLU with L1(0.006)/L2(0.016), Dropout 0.4, Linear(num_classes).
    """
    return EfficientNetB3AADL(num_classes=num_classes, pretrained=pretrained)


def build_alexnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """torchvision AlexNet with the final classifier layer replaced."""
    weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.alexnet(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def build_model(
    name: ModelName | str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "alexnet":
        return build_alexnet(num_classes, pretrained=pretrained)
    if key in ("efficientnet_b3", "efficientnetb3", "efficientnet"):
        return build_efficientnet_b3(num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}. Choose 'alexnet' or 'efficientnet_b3'.")


def default_image_size(name: str) -> int:
    key = name.lower().replace("-", "_")
    if key.startswith("efficientnet"):
        return 300
    return 224
