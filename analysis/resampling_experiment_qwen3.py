#!/usr/bin/env python3
"""
Qwen3-1.7B variant of resampling_experiment.py.

Improvements over a naive port:
  - Forces use of all available GPUs: HF scoring model on cuda:0,
    vLLM with tensor_parallel_size=(N-1) on cuda:1..(N-1).
  - Disables vLLM's torch.compile path (VLLM_USE_V1=0) to avoid the
    "failed to get the hash of the compiled graph" assertion seen with
    some vLLM/torch combinations.
  - Forces enforce_eager=True on vLLM as a safety belt.

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python resampling_experiment_qwen3.py
"""

import os
import sys

# CUDA_VISIBLE_DEVICES must be set before any cuda init.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Disable vLLM's torch.compile path (works around hash-of-graph errors).
os.environ.setdefault("VLLM_USE_V1", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Install shims for missing legacy deps BEFORE importing the original.
import _qwen3_shims  # noqa: F401

# The legacy script unconditionally overwrites CUDA_VISIBLE_DEVICES on
# import.  Save the user's value first, then restore it after import.
_user_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

import resampling_experiment as rs_orig  # noqa: E402

if _user_visible is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _user_visible

import torch  # noqa: E402
from vllm import LLM as _OrigLLM  # noqa: E402


QWEN3_MODEL = "Qwen/Qwen3-1.7B"

# Qwen3-1.7B has 16 attention heads.  vLLM requires TP to divide that
# evenly, so valid TP sizes are 1, 2, 4, 8, 16.  We use TP=N if all
# N visible GPUs evenly divide head count, else fall back to the
# largest divisor.
def _pick_tp(n_gpus: int) -> int:
    for tp in (n_gpus, 4, 2, 1):
        if tp <= n_gpus and 16 % tp == 0:
            return tp
    return 1


_n_gpus = max(1, torch.cuda.device_count())
TP_SIZE = _pick_tp(_n_gpus)


# ----- Patch the vLLM constructor used inside resampling_experiment -----
class _PatchedLLM(_OrigLLM):
    def __init__(self, *args, **kwargs):
        kwargs["tensor_parallel_size"] = TP_SIZE
        kwargs.setdefault("enforce_eager", True)
        # Lower memory utilization since the HF scoring model also lives
        # on cuda:0 alongside one of the vLLM TP slots.
        kwargs.setdefault("gpu_memory_utilization", 0.6)
        super().__init__(*args, **kwargs)


rs_orig.LLM = _PatchedLLM


# ----- Wrap load_models to inject the Qwen3 name -----
_orig_load_models = rs_orig.load_models


def _patched_load_models(model_name=QWEN3_MODEL):
    return _orig_load_models(model_name=model_name)


rs_orig.load_models = _patched_load_models


if __name__ == "__main__":
    import numpy as np
    import random

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    print(
        f"[qwen3] using {_n_gpus} GPU(s): "
        f"HF scoring on cuda:0, vLLM TP={TP_SIZE} on cuda:1..{_n_gpus - 1}"
    )
    rs_orig.run_experiment()
    print("\n[done] Qwen3 resampling experiment finished.")
