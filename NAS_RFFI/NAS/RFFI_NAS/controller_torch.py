from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class StateSpace:
    """Small helper that stores discrete NAS choices."""

    def __init__(self):
        self.states = OrderedDict()

    def add_state(self, name: str, values: Sequence[int]):
        state_id = len(self.states)
        values = [int(v) for v in values]
        self.states[state_id] = {
            "id": state_id,
            "name": name,
            "values": values,
            "size": len(values),
            "index_map": {idx: value for idx, value in enumerate(values)},
            "value_map": {value: idx for idx, value in enumerate(values)},
        }
        return state_id

    def index_to_value(self, state_id: int, index: int):
        return self[state_id]["index_map"][int(index)]

    def value_to_index(self, state_id: int, value: int):
        return self[state_id]["value_map"][int(value)]

    def parse_indices(self, indices: Sequence[int]):
        return [self.index_to_value(i, idx) for i, idx in enumerate(indices)]

    def parse_state_space_list(self, state_list):
        indices = [int(np.argmax(state_one_hot, axis=-1)[0]) for state_one_hot in state_list]
        return self.parse_indices(indices)

    def to_one_hot(self, state_id: int, value: int):
        state = self[state_id]
        index = state["value_map"][int(value)]
        one_hot = np.zeros((1, state["size"]), dtype=np.float32)
        one_hot[0, index] = 1.0
        return one_hot

    def get_random_state_space(self, num_layers: int):
        states = []
        for state_id in range(self.size * num_layers):
            state = self[state_id]
            value = np.random.choice(state["values"])
            states.append(self.to_one_hot(state_id, int(value)))
        return states

    def print_actions(self, actions):
        print("Actions:")
        for idx, value in enumerate(actions):
            if idx % self.size == 0:
                print(f"  Layer {idx // self.size + 1}")
            print(f"    {self[idx]['name']}: {value}")

    def __getitem__(self, state_id: int):
        return self.states[state_id % self.size]

    @property
    def size(self):
        return len(self.states)


class Controller(nn.Module):
    """LSTM controller trained with a REINFORCE-style policy gradient."""

    def __init__(
        self,
        num_layers,
        state_space,
        reg_param=0.0,
        exploration=1.0,
        controller_cells=64,
        embedding_dim=32,
        entropy_weight=0.01,
        clip_norm=5.0,
        restore_controller=False,
        use_baseline=True,
        baseline_decay=0.95,
        device="cpu",
        learning_rate=1e-3,
        checkpoint_path="weights/controller_torch.ckpt",
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.state_space = state_space
        self.state_size = state_space.size
        self.reg_param = float(reg_param)
        self.exploration = float(exploration)
        self.controller_cells = int(controller_cells)
        self.embedding_dim = int(embedding_dim)
        self.entropy_weight = float(entropy_weight)
        self.clip_norm = float(clip_norm)
        self.use_baseline = bool(use_baseline)
        self.baseline_decay = float(baseline_decay)
        self.baseline = 0.0
        self.global_step = 0
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested = torch.device(device)
            if requested.type == "cuda" and not torch.cuda.is_available():
                print("CUDA requested but not available; using CPU for controller.")
                requested = torch.device("cpu")
            self.device = requested
        self.checkpoint_path = Path(checkpoint_path)

        self.embeddings = nn.ModuleList(
            [nn.Embedding(state["size"] + 1, self.embedding_dim) for state in self.state_space.states.values()]
        )
        self.start_token = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.lstm_cell = nn.LSTMCell(self.embedding_dim, self.controller_cells)
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.controller_cells, self.state_space[i]["size"]) for i in range(self.state_size * self.num_layers)]
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)

        if restore_controller:
            self.restore_checkpoint()

    def sample_actions(self):
        h = torch.zeros(1, self.controller_cells, device=self.device)
        c = torch.zeros(1, self.controller_cells, device=self.device)
        cell_input = self.start_token
        actions = []
        action_indices = []
        log_probs = []
        entropies = []
        temperature = max(0.2, self.exploration)

        for step in range(self.state_size * self.num_layers):
            state_id = step % self.state_size
            h, c = self.lstm_cell(cell_input, (h, c))
            logits = self.classifiers[step](h) / temperature
            dist = Categorical(logits=logits)
            action_index = dist.sample()
            action_value = self.state_space.index_to_value(step, int(action_index.item()))
            actions.append(action_value)
            action_indices.append(int(action_index.item()))
            log_probs.append(dist.log_prob(action_index))
            entropies.append(dist.entropy())
            next_index = action_index.view(1, 1) + 1
            cell_input = self.embeddings[state_id](next_index).squeeze(1)

        return {
            "actions": actions,
            "action_indices": action_indices,
            "log_probs": log_probs,
            "entropies": entropies,
        }

    def get_action(self, state=None):
        return self.sample_actions()["actions"]

    def train_step(self, rollout, reward: float):
        reward = float(reward)
        if self.use_baseline:
            self.baseline = self.baseline_decay * self.baseline + (1.0 - self.baseline_decay) * reward
            advantage = reward - self.baseline
        else:
            advantage = reward

        log_prob_sum = torch.stack(rollout["log_probs"]).sum()
        entropy_sum = torch.stack(rollout["entropies"]).sum()
        loss = -log_prob_sum * advantage - self.entropy_weight * entropy_sum
        if self.reg_param > 0:
            reg_loss = sum(torch.sum(param ** 2) for param in self.parameters())
            loss = loss + self.reg_param * reg_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        self.global_step += 1

        if self.exploration > 0.2:
            self.exploration *= 0.995
        if self.global_step % 10 == 0:
            self.save_checkpoint()

        return float(loss.detach().cpu()), float(advantage)

    def save_checkpoint(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_step": self.global_step,
                "exploration": self.exploration,
                "baseline": self.baseline,
                "state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.checkpoint_path,
        )

    def restore_checkpoint(self):
        if not self.checkpoint_path.exists():
            print(f"No controller checkpoint found at {self.checkpoint_path}")
            return
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.exploration = float(checkpoint.get("exploration", self.exploration))
        self.baseline = float(checkpoint.get("baseline", 0.0))
        print(f"Restored controller checkpoint from {self.checkpoint_path}")

    def remove_files(self, output_dir="."):
        output_dir = Path(output_dir)
        for file_name in ["train_history.csv", "buffers.txt"]:
            path = output_dir / file_name
            if path.exists():
                path.unlink()


def save_architecture(path, actions, accuracy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "kernel": [int(actions[i]) for i in range(0, 8, 2)],
        "filters": [int(actions[i]) for i in range(1, 8, 2)],
        "best_accuracy": float(accuracy),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return config


if __name__ == "__main__":
    state_space = StateSpace()
    state_space.add_state("kernel", [3, 5, 7])
    state_space.add_state("filters", [16, 32, 64])
    controller = Controller(4, state_space, device="auto")
    rollout = controller.sample_actions()
    print(rollout["actions"])
    print(controller.train_step(rollout, reward=0.1))
