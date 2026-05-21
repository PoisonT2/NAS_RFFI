from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class ChildModel(nn.Module):
    """CNN child model used by NAS."""

    def __init__(
        self,
        actions: Sequence[int],
        num_classes: int = 10,
        stride_pattern: Optional[Sequence[tuple[int, int]]] = None,
        in_channels: int = 1,
        embedding_dim: int = 256,
        pool_size: tuple[int, int] = (4, 4),
        dropout: float = 0.2,
    ):
        super().__init__()
        if len(actions) != 8:
            raise ValueError("actions must be [k1, f1, k2, f2, k3, f3, k4, f4]")
        if stride_pattern is None:
            stride_pattern = [(2, 1), (1, 1), (2, 1), (1, 1)]
        if len(stride_pattern) != 4:
            raise ValueError("stride_pattern must contain four stride tuples")

        blocks = []
        arch_layers = []
        current_channels = in_channels
        for layer_idx in range(4):
            kernel = int(actions[2 * layer_idx])
            filters = int(actions[2 * layer_idx + 1])
            stride = stride_pattern[layer_idx]
            blocks.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        filters,
                        kernel_size=(kernel, kernel),
                        stride=stride,
                        padding=(kernel // 2, kernel // 2),
                        bias=False,
                    ),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                ]
            )
            if layer_idx >= 2 and dropout > 0:
                blocks.append(nn.Dropout2d(p=min(0.3, dropout / 2.0)))
            arch_layers.append(
                {
                    "kernel": kernel,
                    "filters": filters,
                    "stride": [int(stride[0]), int(stride[1])],
                }
            )
            current_channels = filters

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.embedding = nn.Sequential(
            nn.Linear(current_channels * pool_size[0] * pool_size[1], embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.architecture = {
            "kernel": [layer["kernel"] for layer in arch_layers],
            "filters": [layer["filters"] for layer in arch_layers],
            "stride_pattern": [layer["stride"] for layer in arch_layers],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return self.classifier(x)

    def get_architecture(self):
        return self.architecture


def actions_from_config(config: dict) -> list[int]:
    kernels = config["kernel"]
    filters = config["filters"]
    if len(kernels) != len(filters):
        raise ValueError("kernel and filters lists must have the same length")
    actions = []
    for kernel, channel_count in zip(kernels, filters):
        actions.extend([int(kernel), int(channel_count)])
    return actions


def config_from_actions(actions: Sequence[int]) -> dict:
    if len(actions) != 8:
        raise ValueError("actions must contain eight integers")
    return {
        "kernel": [int(actions[i]) for i in range(0, 8, 2)],
        "filters": [int(actions[i]) for i in range(1, 8, 2)],
    }


def model_fn(actions, num_classes: int = 10, stride_pattern=None, in_channels: int = 1):
    return ChildModel(
        [int(v) for v in actions],
        num_classes=num_classes,
        stride_pattern=stride_pattern,
        in_channels=in_channels,
    )


if __name__ == "__main__":
    test_model = model_fn([3, 32, 5, 64, 3, 32, 7, 96], num_classes=10)
    output = test_model(torch.randn(4, 1, 102, 62))
    print(test_model.get_architecture())
    print("output", output.shape)
