from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_learning_models import ClassificationNet
from training_utils import (
    TrainConfig,
    compare_model_parameters,
    evaluate_model_from_arrays,
    prepare_dataset,
    prepare_test_dataset,
    save_checkpoint,
    save_confusion_matrix,
    save_json,
    set_seed,
    train_model_from_arrays,
)

from RFFI_NAS.controller_torch import Controller, StateSpace, save_architecture
from RFFI_NAS.manager_torch import NetworkManager
from RFFI_NAS.model_torch import actions_from_config, config_from_actions, model_fn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "LoRa_RFF" / "dataset"
CHANNEL_DIR = DATA_ROOT / "Test" / "channel_problem"
DEFAULT_ARCH_PATH = Path(__file__).resolve().parent / "nas_best_architecture.json"
STRIDE_PATTERN = [(2, 1), (1, 1), (2, 1), (1, 1)]
AUGMENTED_A_FILE_NAMES = [
    "A_aug_0hz.h5",
    "A_aug_10hz.h5",
    "A_aug_30hz.h5",
    "A_aug_50hz.h5",
    "A_aug_100hz.h5",
]


def _default_test_files():
    return [CHANNEL_DIR / name for name in ["B.h5", "C.h5", "D.h5", "E.h5", "F.h5"]]


def _channel_training_files(train_file, use_channel_augmentations=True):
    train_file = Path(train_file)
    if not use_channel_augmentations or train_file.name != "A.h5":
        return train_file
    files = [train_file]
    files.extend(train_file.parent / name for name in AUGMENTED_A_FILE_NAMES)
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Requested A.h5 channel augmentations, but these files are missing: "
            + ", ".join(missing)
        )
    return files


def _print_training_files(stage, file_paths):
    files = [file_paths] if isinstance(file_paths, (str, Path)) else list(file_paths)
    print(f"{stage} training files:")
    for file_path in files:
        print(f"  {Path(file_path)}")


def _range(start, stop):
    return range(int(start), int(stop))


def build_state_space():
    state_space = StateSpace()
    state_space.add_state("kernel", [3, 5, 7])
    state_space.add_state("filters", [16, 32, 64, 96])
    return state_space


def build_config(args, epochs):
    monitor_mode = args.monitor_mode
    if monitor_mode == "auto":
        monitor_mode = "min" if args.monitor == "val_loss" else "max"
    return TrainConfig(
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        min_delta=args.min_delta,
        min_lr=args.min_lr,
        monitor=args.monitor,
        monitor_mode=monitor_mode,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
    )


def load_architecture(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_rollout(controller, seen_actions, duplicate_resample_attempts):
    duplicate_resample_attempts = max(0, int(duplicate_resample_attempts))
    last_rollout = None
    last_key = None
    for attempt in range(duplicate_resample_attempts + 1):
        rollout = controller.sample_actions()
        key = tuple(int(v) for v in rollout["actions"])
        last_rollout = rollout
        last_key = key
        if key not in seen_actions:
            return rollout, key, attempt, False
    return last_rollout, last_key, duplicate_resample_attempts, True


def search_architecture(args):
    set_seed(args.seed)
    dev_range = _range(args.dev_start, args.dev_stop)
    pkt_range = _range(args.pkt_start, args.pkt_stop)
    train_snr_range = None if args.no_train_noise else (args.train_snr_min, args.train_snr_max)
    search_train_files = _channel_training_files(args.search_train_file, args.search_channel_augmentations)
    _print_training_files("NAS search", search_train_files)
    data = prepare_dataset(
        search_train_files,
        dev_range=dev_range,
        pkt_range=pkt_range,
        val_split=args.val_split,
        seed=args.seed,
        train_snr_range=train_snr_range,
        normalize_spectrogram=args.normalize_spectrogram,
    )

    state_space = build_state_space()
    controller = Controller(
        num_layers=4,
        state_space=state_space,
        reg_param=args.controller_l2,
        exploration=args.exploration,
        controller_cells=args.controller_cells,
        embedding_dim=args.embedding_dim,
        entropy_weight=args.entropy_weight,
        restore_controller=args.restore_controller,
        use_baseline=True,
        baseline_decay=args.baseline_decay,
        device=args.device,
        learning_rate=args.controller_lr,
        checkpoint_path=args.output_dir / "controller_torch.ckpt",
    )
    manager = NetworkManager(
        dataset=(data.x_train, data.y_train, data.x_val, data.y_val),
        epochs=args.search_epochs,
        child_batchsize=args.search_batch_size,
        acc_beta=args.reward_beta,
        clip_rewards=args.clip_rewards,
        device=args.device,
        num_classes=data.num_classes,
        stride_pattern=STRIDE_PATTERN,
        seed=args.seed,
    )

    best_acc = -1.0
    best_actions = None
    history = []
    seen_actions = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for trial in range(1, args.max_trials + 1):
        print(f"\nTrial {trial}/{args.max_trials}")
        rollout, action_key, resample_attempts, duplicate_after_resample = sample_rollout(
            controller,
            seen_actions,
            args.duplicate_resample_attempts,
        )
        actions = rollout["actions"]
        if resample_attempts > 0:
            print(f"Resampled duplicate architectures {resample_attempts} time(s).")
        if duplicate_after_resample:
            print("Warning: evaluating a duplicate architecture after exhausting resample attempts.")
        state_space.print_actions(actions)
        reward, val_acc, child_result, param_summary = manager.get_rewards(model_fn, actions)
        loss, advantage = controller.train_step(rollout, reward)
        seen_actions.add(action_key)

        row = {
            "trial": trial,
            "actions": [int(v) for v in actions],
            "architecture": config_from_actions(actions),
            "is_duplicate": bool(duplicate_after_resample),
            "resample_attempts": int(resample_attempts),
            "reward": float(reward),
            "advantage": float(advantage),
            "controller_loss": float(loss),
            "val_accuracy": float(val_acc),
            "child_epochs": int(child_result["epochs_ran"]),
            "parameter_summary": param_summary,
        }
        history.append(row)
        print(
            f"Trial {trial}: val_acc={val_acc:.4f}, reward={reward:.6f}, "
            f"advantage={advantage:.6f}, controller_loss={loss:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_actions = [int(v) for v in actions]
            save_architecture(args.arch_config, best_actions, best_acc)
            print(f"New best architecture saved to {args.arch_config}: {config_from_actions(best_actions)}")

        save_json(history, args.output_dir / "nas_search_history.json")

    best_config = config_from_actions(best_actions)
    best_config["best_accuracy"] = float(best_acc)
    save_json(best_config, args.arch_config)
    return best_config


def train_full_model(model_name, model, data, train_config, output_path, architecture=None, output_dir=None):
    model, train_result = train_model_from_arrays(
        model,
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        config=train_config,
    )
    save_checkpoint(
        output_path,
        model,
        model_name=model_name,
        class_values=data.class_values,
        train_stats=data.train_stats,
        train_config=train_config,
        train_result=train_result,
        architecture=architecture,
    )
    if output_dir is not None:
        save_json(train_result, Path(output_dir) / f"{model_name}_train_history.json")
    print(f"Saved {model_name} checkpoint to {output_path}")
    return model, train_result


def evaluate_named_model(model_name, model, test_files, class_values, pkt_range, train_stats, args):
    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for file_path in test_files:
        x_test, y_test = prepare_test_dataset(file_path, class_values, pkt_range, train_stats)
        metrics = evaluate_model_from_arrays(
            model,
            x_test,
            y_test,
            batch_size=args.eval_batch_size,
            device=args.device,
        )
        results[Path(file_path).name] = {
            key: value for key, value in metrics.items() if key != "confusion_matrix"
        }
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_values,
            model_dir / f"confusion_matrix_{Path(file_path).stem}.pdf",
        )
        print(
            f"{model_name} on {Path(file_path).name}: "
            f"acc={metrics['accuracy']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    save_json(results, model_dir / "test_metrics.json")
    return results


def final_train_and_compare(args, arch_config):
    set_seed(args.seed)
    dev_range = _range(args.dev_start, args.dev_stop)
    pkt_range = _range(args.pkt_start, args.pkt_stop)
    train_snr_range = None if args.no_train_noise else (args.train_snr_min, args.train_snr_max)
    full_train_files = _channel_training_files(args.full_train_file, args.use_channel_augmentations)
    _print_training_files("Full training for cnn_baseline and nas_best", full_train_files)
    data = prepare_dataset(
        full_train_files,
        dev_range=dev_range,
        pkt_range=pkt_range,
        val_split=args.val_split,
        seed=args.seed,
        train_snr_range=train_snr_range,
        normalize_spectrogram=args.normalize_spectrogram,
    )
    train_config = build_config(args, args.full_epochs)

    set_seed(args.seed)
    baseline = ClassificationNet(data.num_classes)
    baseline, baseline_train = train_full_model(
        "cnn_baseline",
        baseline,
        data,
        train_config,
        args.output_dir / "cnn_baseline.pth",
        output_dir=args.output_dir,
    )

    set_seed(args.seed)
    actions = actions_from_config(arch_config)
    nas_model = model_fn(actions, num_classes=data.num_classes, stride_pattern=STRIDE_PATTERN, in_channels=1)
    nas_model, nas_train = train_full_model(
        "nas_best",
        nas_model,
        data,
        train_config,
        args.output_dir / "cnn_nas_best.pth",
        architecture=arch_config,
        output_dir=args.output_dir,
    )

    test_files = [Path(p) for p in args.test_files]
    baseline_results = evaluate_named_model(
        "cnn_baseline", baseline, test_files, data.class_values, pkt_range, data.train_stats, args
    )
    nas_results = evaluate_named_model(
        "nas_best", nas_model, test_files, data.class_values, pkt_range, data.train_stats, args
    )
    comparison = compare_model_parameters("cnn_baseline", baseline, "nas_best", nas_model)
    comparison["best_architecture"] = arch_config
    comparison["validation"] = {
        "cnn_baseline": baseline_train["best_val_accuracy"],
        "nas_best": nas_train["best_val_accuracy"],
    }
    comparison["test_metrics"] = {
        "cnn_baseline": baseline_results,
        "nas_best": nas_results,
    }
    save_json(comparison, args.output_dir / "model_parameter_comparison.json")
    print(f"Saved parameter comparison to {args.output_dir / 'model_parameter_comparison.json'}")
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description="NAS search and final LoRa RFFI model comparison")
    parser.add_argument("--mode", choices=["search", "final", "search-final", "smoke"], default="final")
    parser.add_argument("--search-train-file", type=Path, default=CHANNEL_DIR / "A.h5")
    parser.add_argument("--full-train-file", type=Path, default=CHANNEL_DIR / "A.h5")
    parser.add_argument("--test-files", type=Path, nargs="*", default=_default_test_files())
    parser.add_argument("--arch-config", type=Path, default=DEFAULT_ARCH_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "nas")
    parser.add_argument("--dev-start", type=int, default=30)
    parser.add_argument("--dev-stop", type=int, default=40)
    parser.add_argument("--pkt-start", type=int, default=0)
    parser.add_argument("--pkt-stop", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--search-epochs", type=int, default=30)
    parser.add_argument("--full-epochs", type=int, default=400)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.2)
    parser.add_argument("--min-delta", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--monitor", choices=["val_loss", "val_accuracy"], default="val_accuracy")
    parser.add_argument("--monitor-mode", choices=["auto", "min", "max"], default="auto")
    parser.add_argument("--optimizer", choices=["rmsprop", "adam"], default="rmsprop")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--train-snr-min", type=float, default=20.0)
    parser.add_argument("--train-snr-max", type=float, default=80.0)
    parser.add_argument("--no-train-noise", action="store_true")
    parser.add_argument("--use-channel-augmentations", dest="use_channel_augmentations", action="store_true", default=True)
    parser.add_argument("--no-channel-augmentations", dest="use_channel_augmentations", action="store_false")
    parser.add_argument("--search-channel-augmentations", dest="search_channel_augmentations", action="store_true", default=True)
    parser.add_argument("--no-search-channel-augmentations", dest="search_channel_augmentations", action="store_false")
    parser.add_argument("--normalize-spectrogram", action="store_true")
    parser.add_argument("--exploration", type=float, default=1.0)
    parser.add_argument("--controller-cells", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--controller-lr", type=float, default=1e-3)
    parser.add_argument("--controller-l2", type=float, default=0.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--baseline-decay", type=float, default=0.95)
    parser.add_argument("--reward-beta", type=float, default=0.8)
    parser.add_argument("--clip-rewards", type=float, default=0.0)
    parser.add_argument("--restore-controller", action="store_true")
    parser.add_argument("--duplicate-resample-attempts", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        args.max_trials = 1
        args.search_epochs = 1
        args.full_epochs = 1
        args.batch_size = 8
        args.search_batch_size = 8
        args.eval_batch_size = 16
        args.pkt_stop = min(args.pkt_stop, args.pkt_start + 2)
        args.test_files = args.test_files[:1]
        args.output_dir = Path("results") / "smoke_nas"
        args.arch_config = args.output_dir / "nas_best_architecture.json"
        args.mode = "search-final"
        args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "search":
        search_architecture(args)
    elif args.mode == "final":
        final_train_and_compare(args, load_architecture(args.arch_config))
    elif args.mode == "search-final":
        best_config = search_architecture(args)
        final_train_and_compare(args, best_config)


if __name__ == "__main__":
    main()
