from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
NAS_RFFI_ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(NAS_RFFI_ROOT))
sys.path.insert(0, str(THIS_DIR))

try:
    from .models import available_models, create_model, model_metadata
except ImportError:
    from models import available_models, create_model, model_metadata

from training_utils import parameter_summary


def main() -> None:
    results = {}
    x = torch.randn(2, 1, 102, 62)
    for name in available_models():
        model = create_model(name, num_classes=10)
        model.eval()
        with torch.no_grad():
            y = model(x)
        if tuple(y.shape) != (2, 10):
            raise AssertionError(f"{name} returned shape {tuple(y.shape)}, expected (2, 10)")
        results[name] = {
            "metadata": model_metadata(name),
            "output_shape": list(y.shape),
            "parameter_summary": parameter_summary(model),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

