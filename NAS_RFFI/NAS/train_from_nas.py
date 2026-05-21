"""Compatibility entrypoint for full training from a saved NAS architecture.

The maintained implementation lives in train_torch.py. This wrapper keeps old
commands working while ensuring NAS final training uses the same training logic
as the DeepLearning_torch CNN baseline.
"""

from __future__ import annotations

import sys

from train_torch import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.insert(1, "final")
        sys.argv.insert(1, "--mode")
    main()
