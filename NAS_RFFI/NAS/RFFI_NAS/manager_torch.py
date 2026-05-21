from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from training_utils import TrainConfig, parameter_summary, preserve_rng_state, set_seed, train_model_from_arrays
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training_utils import TrainConfig, parameter_summary, preserve_rng_state, set_seed, train_model_from_arrays


class NetworkManager:
    """Train child networks and return validation rewards for NAS."""

    def __init__(
        self,
        dataset,
        epochs: int = 5,
        child_batchsize: int = 128,
        acc_beta: float = 0.8,
        clip_rewards: float = 0.0,
        device: str = "auto",
        num_classes: int = 10,
        stride_pattern: Optional[Sequence[tuple[int, int]]] = None,
        seed: int = 42,
    ):
        self.x_train, self.y_train, self.x_val, self.y_val = dataset
        self.epochs = int(epochs)
        self.batchsize = int(child_batchsize)
        self.clip_rewards = float(clip_rewards)
        self.device = device
        self.num_classes = int(num_classes)
        self.stride_pattern = stride_pattern
        self.beta = float(acc_beta)
        self.moving_acc = 0.0
        self.seed = int(seed)
        self.trial = 0

    def get_rewards(self, model_fn, actions):
        with preserve_rng_state():
            set_seed(self.seed)
            model = model_fn(
                actions,
                num_classes=self.num_classes,
                stride_pattern=self.stride_pattern,
                in_channels=1,
            )
            config = TrainConfig(
                epochs=self.epochs,
                batch_size=self.batchsize,
                learning_rate=1e-3,
                weight_decay=1e-4,
                patience=max(3, min(10, self.epochs)),
                reduce_lr_patience=max(2, min(5, self.epochs // 2 or 1)),
                optimizer="adam",
                seed=self.seed,
                device=self.device,
                log_interval=max(1, self.epochs),
                monitor="val_accuracy",
                monitor_mode="max",
                min_delta=0.0,
            )
            model, result = train_model_from_arrays(
                model,
                self.x_train,
                self.y_train,
                self.x_val,
                self.y_val,
                config=config,
                quiet=True,
            )
            final_acc = float(result["best_val_accuracy"])
            summary = parameter_summary(model)

        reward = final_acc - self.moving_acc
        if self.clip_rewards > 0:
            reward = float(np.clip(reward, -self.clip_rewards, self.clip_rewards))
        if 0.0 < self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1.0 - self.beta) * final_acc
        self.trial += 1

        print(
            f"Manager: val_acc={final_acc:.4f}, reward={reward:.6f}, "
            f"params={summary['total_params']}"
        )
        return reward, final_acc, result, summary
