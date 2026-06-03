from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
NAS_RFFI_ROOT = THIS_DIR.parents[0]
WORKSPACE_ROOT = THIS_DIR.parents[2]
DATA_ROOT = WORKSPACE_ROOT / "LoRa_RFF" / "dataset"
CHANNEL_DIR = DATA_ROOT / "Test" / "channel_problem"
DEFAULT_RESULTS_DIR = NAS_RFFI_ROOT / "results" / "manual_lightweight_baselines"
DEFAULT_INPUT_SHAPE = (1, 102, 62)

sys.path.insert(0, str(NAS_RFFI_ROOT))
sys.path.insert(0, str(THIS_DIR))

try:
    from .models import available_models, create_model, model_metadata
except ImportError:
    from models import available_models, create_model, model_metadata

from model_efficiency_compare import benchmark_inference, count_macs_and_flops
from training_utils import (
    TrainConfig,
    evaluate_model_from_arrays,
    parameter_summary,
    prepare_dataset,
    prepare_test_dataset,
    resolve_device,
    save_checkpoint,
    save_confusion_matrix,
    save_json,
    set_seed,
    train_model_from_arrays,
)


def _default_test_files() -> list[Path]:
    return [CHANNEL_DIR / name for name in ["B.h5", "C.h5", "D.h5", "E.h5", "F.h5"]]


def _range(start: int, stop: int) -> range:
    return range(int(start), int(stop))


def _channel_training_files(train_file: Path, use_channel_augmentations: bool = True):
    train_file = Path(train_file)
    if not use_channel_augmentations or train_file.name != "A.h5":
        return train_file
    aug_names = [
        "A_aug_0hz.h5",
        "A_aug_10hz.h5",
        "A_aug_30hz.h5",
        "A_aug_50hz.h5",
        "A_aug_100hz.h5",
    ]
    files = [train_file]
    files.extend(train_file.parent / name for name in aug_names if (train_file.parent / name).exists())
    return files


def _file_list_for_json(file_paths) -> list[str]:
    paths = file_paths if isinstance(file_paths, list) else [file_paths]
    return [str(Path(path)) for path in paths]


def build_train_config(args: argparse.Namespace) -> TrainConfig:
    monitor_mode = args.monitor_mode
    if monitor_mode == "auto":
        monitor_mode = "min" if args.monitor == "val_loss" else "max"
    return TrainConfig(
        epochs=args.epochs,
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
        use_amp=not args.no_amp,
        log_interval=args.log_interval,
    )


def evaluate_files(
    model: torch.nn.Module,
    test_files: list[Path],
    class_values: list[int],
    pkt_range: range,
    train_stats: dict[str, float],
    output_dir: Path,
    batch_size: int,
    device: str,
    save_matrices: bool,
) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}
    for file_path in test_files:
        file_path = Path(file_path)
        print(f"Testing {file_path.name}")
        x_test, y_test = prepare_test_dataset(file_path, class_values, pkt_range, train_stats)
        metrics = evaluate_model_from_arrays(
            model,
            x_test,
            y_test,
            batch_size=batch_size,
            device=device,
        )
        results[file_path.name] = {
            key: value for key, value in metrics.items() if key != "confusion_matrix"
        }
        if save_matrices:
            save_confusion_matrix(
                metrics["confusion_matrix"],
                class_values,
                output_dir / f"confusion_matrix_{file_path.stem}.pdf",
            )
        print(
            f"{file_path.name}: acc={metrics['accuracy']:.4f}, "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
    save_json(results, output_dir / "test_metrics.json")
    return results


def build_efficiency_summary(model: torch.nn.Module, args: argparse.Namespace) -> dict[str, Any]:
    summary = parameter_summary(model)
    if args.skip_efficiency:
        return summary

    device = resolve_device(args.device)
    input_shape = (args.input_channels, args.input_height, args.input_width)
    complexity = count_macs_and_flops(model, input_shape, device)
    inference = benchmark_inference(
        model,
        input_shape,
        device,
        warmup=args.warmup,
        iterations=args.iterations,
        batch_size=args.inference_batch_size,
    )
    summary.update(complexity)
    summary["inference"] = inference
    return summary


def _summarize_train_result(train_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_val_loss": train_result["best_val_loss"],
        "best_val_accuracy": train_result["best_val_accuracy"],
        "best_epoch": train_result["best_epoch"],
        "monitor": train_result["monitor"],
        "best_monitor": train_result["best_monitor"],
        "epochs_ran": train_result["epochs_ran"],
    }


def train_and_evaluate_model(
    model_name: str,
    data,
    train_config: TrainConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n=== Training {model_name} ===")
    set_seed(train_config.seed)
    model = create_model(model_name, data.num_classes)
    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model, train_result = train_model_from_arrays(
        model,
        data.x_train,
        data.y_train,
        data.x_val,
        data.y_val,
        config=train_config,
        quiet=args.quiet_training,
    )
    architecture = model.get_architecture() if hasattr(model, "get_architecture") else model_metadata(model_name)
    checkpoint_path = model_dir / f"{model_name}.pth"
    save_checkpoint(
        checkpoint_path,
        model,
        model_name=model_name,
        class_values=data.class_values,
        train_stats=data.train_stats,
        train_config=train_config,
        train_result=train_result,
        architecture=architecture,
        extra={"manual_baseline_metadata": model_metadata(model_name)},
    )
    save_json(train_result, model_dir / "train_history.json")

    test_results = evaluate_files(
        model,
        [Path(path) for path in args.test_files],
        data.class_values,
        _range(args.pkt_start, args.pkt_stop),
        data.train_stats,
        model_dir,
        batch_size=args.eval_batch_size,
        device=args.device,
        save_matrices=not args.no_confusion_matrix,
    )
    efficiency = build_efficiency_summary(model, args)
    model_summary = {
        "model_name": model_name,
        "model_metadata": model_metadata(model_name),
        "architecture": architecture,
        "checkpoint": str(checkpoint_path),
        "train_result": _summarize_train_result(train_result),
        "test_metrics": test_results,
        "efficiency": efficiency,
    }
    save_json(model_summary, model_dir / "model_summary.json")
    return model_summary


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_files = _channel_training_files(args.train_file, args.use_channel_augmentations)
    train_snr_range = None if args.no_train_noise else (args.train_snr_min, args.train_snr_max)
    dev_range = _range(args.dev_start, args.dev_stop)
    pkt_range = _range(args.pkt_start, args.pkt_stop)
    train_config = build_train_config(args)

    print("Manual lightweight baseline experiment")
    print(f"Models: {args.models}")
    print(f"Training files: {_file_list_for_json(train_files)}")
    print(f"Output dir: {args.output_dir}")

    data = prepare_dataset(
        train_files,
        dev_range=dev_range,
        pkt_range=pkt_range,
        val_split=args.val_split,
        seed=args.seed,
        train_snr_range=train_snr_range,
        normalize_spectrogram=args.normalize_spectrogram,
    )

    model_summaries = {}
    for model_name in args.models:
        model_summaries[model_name] = train_and_evaluate_model(
            model_name,
            data,
            train_config,
            args,
        )

    experiment_summary = {
        "experiment": "manual_lightweight_baselines",
        "models": args.models,
        "train_files": _file_list_for_json(train_files),
        "test_files": [str(Path(path)) for path in args.test_files],
        "class_values": data.class_values,
        "train_stats": data.train_stats,
        "train_config": asdict(train_config),
        "data_config": {
            "dev_start": args.dev_start,
            "dev_stop": args.dev_stop,
            "pkt_start": args.pkt_start,
            "pkt_stop": args.pkt_stop,
            "val_split": args.val_split,
            "train_snr_range": train_snr_range,
            "use_channel_augmentations": args.use_channel_augmentations,
            "normalize_spectrogram": args.normalize_spectrogram,
        },
        "model_summaries": model_summaries,
    }
    save_json(experiment_summary, args.output_dir / "manual_lightweight_summary.json")
    return experiment_summary


def list_models() -> None:
    print("Available manual lightweight baselines:")
    for name in available_models():
        metadata = model_metadata(name)
        print(f"  {name}: {metadata['description']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate manual lightweight CNN baselines for LoRa RFFI."
    )
    parser.add_argument("--mode", choices=["train-eval", "smoke", "list-models"], default="train-eval")
    parser.add_argument("--models", nargs="*", choices=available_models(), default=available_models())
    parser.add_argument("--train-file", type=Path, default=CHANNEL_DIR / "A.h5")
    parser.add_argument("--test-files", type=Path, nargs="*", default=_default_test_files())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dev-start", type=int, default=30)
    parser.add_argument("--dev-stop", type=int, default=40)
    parser.add_argument("--pkt-start", type=int, default=0)
    parser.add_argument("--pkt-stop", type=int, default=200)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=400)
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
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--quiet-training", action="store_true")
    parser.add_argument("--train-snr-min", type=float, default=20.0)
    parser.add_argument("--train-snr-max", type=float, default=80.0)
    parser.add_argument("--no-train-noise", action="store_true")
    parser.add_argument("--use-channel-augmentations", dest="use_channel_augmentations", action="store_true", default=True)
    parser.add_argument("--no-channel-augmentations", dest="use_channel_augmentations", action="store_false")
    parser.add_argument("--normalize-spectrogram", action="store_true")
    parser.add_argument("--no-confusion-matrix", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true")
    parser.add_argument("--input-channels", type=int, default=DEFAULT_INPUT_SHAPE[0])
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_SHAPE[1])
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_SHAPE[2])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--inference-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "list-models":
        list_models()
        return
    if args.mode == "smoke":
        args.epochs = 1
        args.batch_size = 8
        args.eval_batch_size = 16
        args.pkt_stop = min(args.pkt_stop, args.pkt_start + 2)
        args.test_files = args.test_files[:1]
        args.output_dir = DEFAULT_RESULTS_DIR / "smoke"
        args.warmup = 0
        args.iterations = 1
        args.no_confusion_matrix = True
        args.mode = "train-eval"
    run_experiment(args)


if __name__ == "__main__":
    main()

