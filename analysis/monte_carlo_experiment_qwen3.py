#!/usr/bin/env python3
"""
Qwen3-1.7B variant of monte_carlo_experiment.py.

This is a thin wrapper that:
  - Sets model_name = "Qwen/Qwen3-1.7B"
  - Redirects all output files to ./qwen3_outputs/  (so existing Qwen2.5 results stay)

Usage:
    cd analysis/
    python monte_carlo_experiment_qwen3.py
"""

import os
import sys
import random
import numpy as np
import torch

# Make the original module importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import monte_carlo_experiment as mc_orig


# Output directory specific to Qwen3 — keeps Qwen2.5 artifacts untouched
QWEN3_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_outputs")
os.makedirs(QWEN3_OUT_DIR, exist_ok=True)


def main():
    # Build config with Qwen3 model + isolated output dir
    config = mc_orig.Config()
    config.model_name = "Qwen/Qwen3-1.7B"
    config.save_dir = QWEN3_OUT_DIR

    # Reproduce mc_orig.main()'s flow but use our config
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # The simplest robust approach: temporarily monkey-patch Config so any
    # call to Config() inside the original main() gets our values.
    _OrigConfig = mc_orig.Config
    def _patched_Config(*args, **kwargs):
        c = _OrigConfig(*args, **kwargs)
        c.model_name = "Qwen/Qwen3-1.7B"
        c.save_dir = QWEN3_OUT_DIR
        return c
    mc_orig.Config = _patched_Config

    try:
        mc_orig.main()
    finally:
        mc_orig.Config = _OrigConfig

    print(f"\n[done] Qwen3 monte-carlo outputs written under: {QWEN3_OUT_DIR}")


if __name__ == "__main__":
    main()
