from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from deep_learning_models import ClassificationNet
from NAS.RFFI_NAS.model_torch import actions_from_config, model_fn
from training_utils import parameter_summary, resolve_device


DEFAULT_RESULTS_DIR = Path("results") / "nas"
DEFAULT_INPUT_SHAPE = (1, 102, 62)
STRIDE_PATTERN = [(2, 1), (1, 1), (2, 1), (1, 1)]


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _repair_lazy_linear_metadata(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Linear) and getattr(module, "in_features", None) == 0:
            weight = getattr(module, "weight", None)
            if weight is not None and weight.ndim == 2:
                module.in_features = int(weight.shape[1])


def _replace_lazy_linear_modules(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LazyLinear):
            weight = child.weight.detach()
            bias = child.bias.detach() if child.bias is not None else None
            replacement = nn.Linear(
                in_features=int(weight.shape[1]),
                out_features=int(weight.shape[0]),
                bias=bias is not None,
            )
            replacement.weight.data.copy_(weight)
            if bias is not None:
                replacement.bias.data.copy_(bias)
            setattr(module, name, replacement)
        else:
            _replace_lazy_linear_modules(child)


def load_baseline_model(checkpoint_path: Path):
    checkpoint = _load_checkpoint(checkpoint_path)
    num_classes = len(checkpoint["class_values"])
    model = ClassificationNet(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    _repair_lazy_linear_metadata(model)
    _replace_lazy_linear_modules(model)
    return model, checkpoint


def load_nas_model(checkpoint_path: Path, fallback_architecture=None):
    checkpoint = _load_checkpoint(checkpoint_path)
    num_classes = len(checkpoint["class_values"])
    architecture = checkpoint.get("architecture") or fallback_architecture
    if architecture is None:
        raise ValueError("NAS checkpoint does not contain architecture and no fallback was provided")
    actions = actions_from_config(architecture)
    model = model_fn(actions, num_classes=num_classes, stride_pattern=STRIDE_PATTERN, in_channels=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    _repair_lazy_linear_metadata(model)
    _replace_lazy_linear_modules(model)
    return model, checkpoint, architecture


def count_macs_and_flops(model: nn.Module, input_shape, device):
    """Count Conv2d/Linear MACs and FLOPs for one input sample.

    FLOPs are reported as 2 * MACs, a common convention for multiply-add pairs.
    BatchNorm/ReLU/pooling are intentionally excluded so both models are compared
    under the same conv/linear-dominant convention.
    """
    model = model.to(device)
    model.eval()
    macs_by_module = {}
    hooks = []

    def conv_hook(module, inputs, output):
        out = output.detach()
        batch = max(1, int(out.shape[0]))
        out_elements_per_sample = out.numel() // batch
        kernel_ops = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * (module.in_channels // module.groups)
        )
        macs_by_module[id(module)] = macs_by_module.get(id(module), 0) + int(out_elements_per_sample * kernel_ops)

    def linear_hook(module, inputs, output):
        out = output.detach()
        batch = max(1, int(out.shape[0]))
        out_features_per_sample = out.numel() // batch
        if inputs and hasattr(inputs[0], "shape"):
            in_features = int(inputs[0].shape[-1])
        else:
            in_features = int(module.weight.shape[1])
        macs_by_module[id(module)] = macs_by_module.get(id(module), 0) + int(in_features * out_features_per_sample)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn((1, *input_shape), device=device)
    with torch.no_grad():
        model(dummy)

    for hook in hooks:
        hook.remove()

    macs = int(sum(macs_by_module.values()))
    return {
        "macs": macs,
        "flops": int(2 * macs),
        "macs_m": macs / 1e6,
        "flops_m": (2 * macs) / 1e6,
        "conv_linear_macs": macs,
        "conv_linear_macs_m": macs / 1e6,
        "conv_linear_flops_2x": int(2 * macs),
        "conv_linear_flops_2x_m": (2 * macs) / 1e6,
        "flops_method": "forward_hooks",
        "flops_convention": "Conv2d/Linear only, FLOPs = 2 * MACs, per single sample",
        "macs_convention": "Conv2d/Linear multiply-accumulate pairs only; bias adds, normalization, activation, pooling, and tensor ops are excluded.",
    }


def count_macs_and_flops_thop(model: nn.Module, input_shape, device):
    """Count MACs with thop.profile.

    THOP returns a raw operation count usually described as MACs. In practice,
    THOP's default handlers may also include non-Conv/Linear operators such as
    BatchNorm/Norm, so this script keeps the raw THOP value as the primary
    THOP FLOPs/ops value and stores a separate 2x estimate for users who want
    multiply and add counted as two operations.
    """
    if importlib.util.find_spec("thop") is None:
        raise RuntimeError("thop is not installed. Install it with: pip install thop")

    import thop
    from thop import profile

    try:
        thop_version = importlib.metadata.version("thop")
    except importlib.metadata.PackageNotFoundError:
        thop_version = getattr(thop, "__version__", "unknown")

    model = model.to(device)
    model.eval()
    dummy = torch.randn((1, *input_shape), device=device)
    macs, thop_params = profile(model, inputs=(dummy,), verbose=False)
    thop_ops = int(macs)
    return {
        "macs": thop_ops,
        "flops": thop_ops,
        "macs_m": thop_ops / 1e6,
        "flops_m": thop_ops / 1e6,
        "thop_raw_ops": thop_ops,
        "thop_raw_ops_m": thop_ops / 1e6,
        "thop_raw_macs": thop_ops,
        "thop_raw_params": int(thop_params),
        "thop_version": thop_version,
        "flops_2x_macs_estimate": int(2 * thop_ops),
        "flops_2x_macs_estimate_m": (2 * thop_ops) / 1e6,
        "flops_method": "thop",
        "flops_convention": "THOP raw module-hook operation count is used as FLOPs/ops; no extra x2 is applied. See flops_2x_macs_estimate for the alternate 2x MACs convention.",
        "macs_convention": "For THOP mode, macs stores the raw THOP op count for backward-compatible comparison, not a pure Conv/Linear-only MAC count. Use conv_linear_macs for pure Conv2d/Linear MACs.",
        "thop_limitations": [
            "THOP default handlers may include module ops such as BatchNorm/LayerNorm while omitting functional tensor ops such as torch.nn.functional.normalize.",
            "This THOP version counts Conv/Linear multiply-accumulate pairs as one raw op in the dominant terms.",
        ],
    }


def benchmark_inference(model: nn.Module, input_shape, device, warmup, iterations, batch_size):
    model = model.to(device)
    model.eval()
    dummy = torch.randn((batch_size, *input_shape), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    per_batch_ms = (elapsed / iterations) * 1000.0
    per_sample_ms = per_batch_ms / batch_size
    return {
        "device": str(device),
        "batch_size": int(batch_size),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(iterations),
        "avg_batch_latency_ms": per_batch_ms,
        "avg_sample_latency_ms": per_sample_ms,
        "throughput_samples_per_sec": 1000.0 / per_sample_ms if per_sample_ms > 0 else None,
    }


def build_efficiency_summary(model, input_shape, device, warmup, iterations, batch_size, flops_method):
    params = parameter_summary(model)
    conv_linear_complexity = count_macs_and_flops(model, input_shape, device)
    if flops_method == "thop":
        complexity = count_macs_and_flops_thop(model, input_shape, device)
        complexity.update(
            {
                "conv_linear_macs": conv_linear_complexity["conv_linear_macs"],
                "conv_linear_macs_m": conv_linear_complexity["conv_linear_macs_m"],
                "conv_linear_flops_2x": conv_linear_complexity["conv_linear_flops_2x"],
                "conv_linear_flops_2x_m": conv_linear_complexity["conv_linear_flops_2x_m"],
            }
        )
    elif flops_method == "hook":
        complexity = conv_linear_complexity
    else:
        raise ValueError(f"Unsupported FLOPs method: {flops_method}")
    inference = benchmark_inference(model, input_shape, device, warmup, iterations, batch_size)
    summary = dict(params)
    summary.update(complexity)
    summary["inference"] = inference
    return summary


def update_comparison_json(args):
    comparison_path = args.comparison_json
    comparison = _load_json(comparison_path)
    device = resolve_device(args.device)
    input_shape = (args.input_channels, args.input_height, args.input_width)

    baseline_model, _ = load_baseline_model(args.baseline_checkpoint)
    nas_model, _, architecture = load_nas_model(
        args.nas_checkpoint,
        fallback_architecture=comparison.get("best_architecture"),
    )

    baseline_summary = build_efficiency_summary(
        baseline_model,
        input_shape,
        device,
        args.warmup,
        args.iterations,
        args.batch_size,
        args.flops_method,
    )
    nas_summary = build_efficiency_summary(
        nas_model,
        input_shape,
        device,
        args.warmup,
        args.iterations,
        args.batch_size,
        args.flops_method,
    )

    comparison["cnn_baseline"] = {
        **comparison.get("cnn_baseline", {}),
        **baseline_summary,
    }
    comparison["nas_best"] = {
        **comparison.get("nas_best", {}),
        **nas_summary,
    }
    comparison["candidate_to_reference_param_ratio"] = (
        nas_summary["total_params"] / max(1, baseline_summary["total_params"])
    )
    comparison["candidate_to_reference_flops_ratio"] = (
        nas_summary["flops"] / max(1, baseline_summary["flops"])
    )
    comparison["candidate_to_reference_latency_ratio"] = (
        nas_summary["inference"]["avg_sample_latency_ms"]
        / max(1e-12, baseline_summary["inference"]["avg_sample_latency_ms"])
    )
    comparison["parameter_delta"] = nas_summary["total_params"] - baseline_summary["total_params"]
    comparison["flops_delta"] = nas_summary["flops"] - baseline_summary["flops"]
    comparison["best_architecture"] = architecture
    comparison["efficiency_settings"] = {
        "input_shape_nchw": [1, *input_shape],
        "device": str(device),
        "batch_size": int(args.batch_size),
        "warmup_iterations": int(args.warmup),
        "timed_iterations": int(args.iterations),
        "flops_method": args.flops_method,
        "flops_counted_modules": (
            ["torch.nn.Conv2d", "torch.nn.Linear"]
            if args.flops_method == "hook"
            else "thop default supported module handlers"
        ),
        "complexity_notes": [
            "All complexity values are per single input sample with shape NCHW = [1, channels, height, width].",
            "Inference latency is runtime and hardware dependent; use ratios only for this local run.",
            "conv_linear_macs and conv_linear_flops_2x are always stored as a cross-check independent of THOP.",
        ],
    }

    _save_json(comparison, comparison_path)
    return comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compute model params, FLOPs, and inference time")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--comparison-json", type=Path, default=None)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--nas-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-channels", type=int, default=DEFAULT_INPUT_SHAPE[0])
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_SHAPE[1])
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_SHAPE[2])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--flops-method", choices=["hook", "thop"], default="hook")
    args = parser.parse_args()

    if args.comparison_json is None:
        args.comparison_json = args.results_dir / "model_parameter_comparison.json"
    if args.baseline_checkpoint is None:
        args.baseline_checkpoint = args.results_dir / "cnn_baseline.pth"
    if args.nas_checkpoint is None:
        args.nas_checkpoint = args.results_dir / "cnn_nas_best.pth"
    return args


def main():
    args = parse_args()
    comparison = update_comparison_json(args)
    print(f"Updated {args.comparison_json}")
    for name in ["cnn_baseline", "nas_best"]:
        item = comparison[name]
        print(
            f"{name}: params={item['total_params']:,}, "
            f"FLOPs={item['flops_m']:.3f}M, "
            f"latency={item['inference']['avg_sample_latency_ms']:.4f} ms/sample"
        )
    print(
        "NAS/Baseline ratios: "
        f"params={comparison['candidate_to_reference_param_ratio']:.6f}, "
        f"FLOPs={comparison['candidate_to_reference_flops_ratio']:.6f}, "
        f"latency={comparison['candidate_to_reference_latency_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
