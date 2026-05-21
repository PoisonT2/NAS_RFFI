"""Compatibility entrypoint for CNN-vs-NAS final comparison.

The maintained implementation lives in train_torch.py.
"""

from __future__ import annotations

import sys

from train_torch import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.insert(1, "final")
        sys.argv.insert(1, "--mode")
    main()
