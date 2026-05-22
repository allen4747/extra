#!/usr/bin/env python3
"""
Qwen3-1.7B variant of resampling_experiment.py.

The original script hardcodes Qwen2.5-1.5B inside main() via load_models()
(no kwarg passed).  This wrapper monkey-patches load_models so it loads
Qwen3-1.7B instead.

NOTE: the original script also hardcodes CUDA_VISIBLE_DEVICES="4,5,6,7"
at the top of the file.  Override by exporting CUDA_VISIBLE_DEVICES
*before* invoking this script.

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python resampling_experiment_qwen3.py
"""

import os
import sys

# Allow user to override the hardcoded CUDA_VISIBLE_DEVICES from the
# original module by setting it before import.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Install shims for missing legacy deps BEFORE importing the original.
import _qwen3_shims  # noqa: F401

import resampling_experiment as rs_orig

QWEN3_MODEL = "Qwen/Qwen3-1.7B"

# Wrap load_models so the wrapper supplies the new model name even though
# the original main() calls load_models() with no arguments.
_orig_load_models = rs_orig.load_models

def _patched_load_models(model_name=QWEN3_MODEL):
    return _orig_load_models(model_name=model_name)

rs_orig.load_models = _patched_load_models


if __name__ == "__main__":
    import torch, numpy as np, random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    rs_orig.main()
    print("\n[done] Qwen3 resampling experiment finished.")
