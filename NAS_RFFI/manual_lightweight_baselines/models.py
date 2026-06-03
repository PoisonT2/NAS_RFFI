from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallResBlock(nn.Module):
    """Residual block used by scaled-down CNN baselines."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SmallClassificationNet(nn.Module):
    """Scaled version of the original residual CNN baseline.

    The model keeps the baseline topology but scales channel counts and the
    embedding dimension so it can serve as a fair manual lightweight baseline.
    """

    def __init__(
        self,
        num_classes: int,
        stem_channels: int = 16,
        block_channels: tuple[int, int, int, int] = (16, 16, 32, 32),
        embedding_dim: int = 256,
        kernel_size: int = 3,
    ):
        super().__init__()
        if len(block_channels) != 4:
            raise ValueError("block_channels must contain four values")
        self.conv1 = nn.Conv2d(1, stem_channels, kernel_size=7, stride=2, padding=3)
        self.res1 = SmallResBlock(stem_channels, block_channels[0], kernel_size=kernel_size)
        self.res2 = SmallResBlock(block_channels[0], block_channels[1], kernel_size=kernel_size)
        self.res3 = SmallResBlock(block_channels[1], block_channels[2], kernel_size=kernel_size)
        self.res4 = SmallResBlock(block_channels[2], block_channels[3], kernel_size=kernel_size)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.LazyLinear(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.architecture = {
            "family": "scaled_residual_cnn",
            "stem_channels": int(stem_channels),
            "block_channels": [int(v) for v in block_channels],
            "embedding_dim": int(embedding_dim),
            "kernel_size": int(kernel_size),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    def get_architecture(self) -> dict[str, Any]:
        return dict(self.architecture)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable convolution block for the MobileCNN baseline."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] = 1,
    ):
        super().__init__()
        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MobileClassificationNet(nn.Module):
    """Depthwise-separable lightweight CNN baseline."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, int, int, int] = (16, 32, 48, 64),
        embedding_dim: int = 128,
        pool_size: tuple[int, int] = (4, 4),
        dropout: float = 0.2,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain four values")
        self.features = nn.Sequential(
            ConvBNAct(1, channels[0], kernel_size=5, stride=(2, 1)),
            DepthwiseSeparableBlock(channels[0], channels[1], kernel_size=3),
            DepthwiseSeparableBlock(channels[1], channels[2], kernel_size=3, stride=(2, 1)),
            DepthwiseSeparableBlock(channels[2], channels[3], kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.embedding = nn.Sequential(
            nn.Linear(channels[-1] * pool_size[0] * pool_size[1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.architecture = {
            "family": "depthwise_separable_cnn",
            "channels": [int(v) for v in channels],
            "embedding_dim": int(embedding_dim),
            "pool_size": [int(v) for v in pool_size],
            "dropout": float(dropout),
        }

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    def get_architecture(self) -> dict[str, Any]:
        return dict(self.architecture)


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "small_cnn_half": {
        "class": SmallClassificationNet,
        "kwargs": {
            "stem_channels": 16,
            "block_channels": (16, 16, 32, 32),
            "embedding_dim": 256,
        },
        "description": "Manual half-width residual CNN baseline.",
    },
    "small_cnn_quarter": {
        "class": SmallClassificationNet,
        "kwargs": {
            "stem_channels": 8,
            "block_channels": (8, 8, 16, 16),
            "embedding_dim": 128,
        },
        "description": "Manual quarter-width residual CNN baseline.",
    },
    "mobile_cnn": {
        "class": MobileClassificationNet,
        "kwargs": {
            "channels": (16, 32, 48, 64),
            "embedding_dim": 128,
            "pool_size": (4, 4),
            "dropout": 0.2,
        },
        "description": "Depthwise-separable lightweight CNN baseline.",
    },
}


def available_models() -> list[str]:
    return list(MODEL_CONFIGS)


def create_model(name: str, num_classes: int) -> nn.Module:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")
    config = MODEL_CONFIGS[name]
    model_cls = config["class"]
    kwargs = deepcopy(config["kwargs"])
    return model_cls(num_classes=num_classes, **kwargs)


def model_metadata(name: str) -> dict[str, Any]:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}. Available: {available_models()}")
    config = MODEL_CONFIGS[name]
    kwargs = deepcopy(config["kwargs"])
    serializable_kwargs = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in kwargs.items()
    }
    return {
        "name": name,
        "class_name": config["class"].__name__,
        "description": config["description"],
        "kwargs": serializable_kwargs,
    }

