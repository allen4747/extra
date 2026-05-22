#!/usr/bin/env python3
"""
Qwen3-1.7B variant of monte_carlo_experiment.py.

Improvements:
  - HF metric-computation model on cuda:0; vLLM with tensor_parallel_size
    = (N-1) on cuda:1..(N-1).  Uses all available GPUs.
  - Disables vLLM torch.compile (VLLM_USE_V1=0) and forces enforce_eager.
  - All Qwen3 outputs go to ./qwen3_outputs/ to avoid clobbering Qwen2.5
    artifacts.

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python monte_carlo_experiment_qwen3.py
"""

import os
import sys
import random
import numpy as np

# CUDA_VISIBLE_DEVICES must be set before any cuda init.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Disable vLLM's torch.compile path.
os.environ.setdefault("VLLM_USE_V1", "0")

# Make legacy modules importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Install shims for `simplified_evaluator` and `openrlhf`.
import _qwen3_shims  # noqa: F401

import torch  # noqa: E402
import monte_carlo_experiment as mc_orig  # noqa: E402
from vllm import LLM as _OrigLLM  # noqa: E402


QWEN3_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_outputs")
os.makedirs(QWEN3_OUT_DIR, exist_ok=True)

_n_gpus = max(1, torch.cuda.device_count())
TP_SIZE = max(1, _n_gpus - 1)


# ----- Patch vLLM to use all (N-1) GPUs after the HF model takes cuda:0 -----
class _PatchedLLM(_OrigLLM):
    def __init__(self, *args, **kwargs):
        kwargs["tensor_parallel_size"] = TP_SIZE
        kwargs.setdefault("enforce_eager", True)
        kwargs.setdefault("gpu_memory_utilization", 0.85)
        super().__init__(*args, **kwargs)


mc_orig.LLM = _PatchedLLM


# ----- Patch HF model loader to use cuda:0 only -----
_orig_load_model = mc_orig.load_model


def _patched_load_model(model_name):
    print(f"[qwen3] loading HF metric model {model_name} on cuda:0 only")
    tokenizer = mc_orig.AutoTokenizer.from_pretrained(model_name)
    model = mc_orig.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


mc_orig.load_model = _patched_load_model


# ----- Patch Config so model name + save dir + TP size are correct -----
_OrigConfig = mc_orig.Config


def _patched_Config(*args, **kwargs):
    c = _OrigConfig(*args, **kwargs)
    c.model_name = "Qwen/Qwen3-1.7B"
    c.save_dir = QWEN3_OUT_DIR
    c.vllm_tensor_parallel_size = TP_SIZE
    return c


def main():
    config = _patched_Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print(
        f"[qwen3] using {_n_gpus} GPU(s): "
        f"HF metric model on cuda:0, vLLM TP={TP_SIZE} on cuda:1..{_n_gpus - 1}"
    )

    # Replace Config in the legacy module so its main() picks up our values.
    mc_orig.Config = _patched_Config
    try:
        mc_orig.main()
    finally:
        mc_orig.Config = _OrigConfig

    print(f"\n[done] Qwen3 monte-carlo outputs written under: {QWEN3_OUT_DIR}")


if __name__ == "__main__":
    main()
