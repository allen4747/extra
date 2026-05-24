#!/usr/bin/env python3
"""
Qwen3-1.7B variant of monte_carlo_experiment.py.

Improvements:
  - HF metric-computation model on cuda:0; vLLM with tensor_parallel_size
    = (N-1) on cuda:1..(N-1).  Uses all available GPUs.
  - Disables vLLM torch.compile (VLLM_USE_V1=0) and forces enforce_eager.
  - All Qwen3 outputs go to ./qwen3_outputs/ to avoid clobbering Qwen2.5
    artifacts.
  - ROBUSTNESS: dual-logs stdout/stderr to a timestamped log file;
    persists the correlations JSON BEFORE attempting any plot; wraps
    each plot call in try/except so a matplotlib failure does not lose
    numerical results.

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0,1,2,3 python monte_carlo_experiment_qwen3.py
"""

import json
import os
import random
import sys
import traceback

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
import _qwen3_robust as robust  # noqa: F401, E402

import torch  # noqa: E402
import monte_carlo_experiment as mc_orig  # noqa: E402
from vllm import LLM as _OrigLLM  # noqa: E402


QWEN3_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_outputs")
os.makedirs(QWEN3_OUT_DIR, exist_ok=True)


# Qwen3-1.7B has 16 attention heads.  vLLM requires TP to divide that
# evenly, so valid TP sizes are 1, 2, 4, 8, 16.
def _pick_tp(n_gpus: int) -> int:
    for tp in (n_gpus, 4, 2, 1):
        if tp <= n_gpus and 16 % tp == 0:
            return tp
    return 1


_n_gpus = max(1, torch.cuda.device_count())
TP_SIZE = _pick_tp(_n_gpus)


# ----- Patch vLLM to use all visible GPUs (with valid TP size) -----
class _PatchedLLM(_OrigLLM):
    def __init__(self, *args, **kwargs):
        kwargs["tensor_parallel_size"] = TP_SIZE
        kwargs.setdefault("enforce_eager", True)
        kwargs.setdefault("gpu_memory_utilization", 0.6)
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


# ----- Robustness: dump correlation results as soon as they're computed,
#       so a later plotting failure cannot lose them.  Also dump per-row
#       prefix data so the user can re-plot from scratch. -----
_orig_analyze_correlations = mc_orig.analyze_correlations


def _patched_analyze_correlations(prefix_data_points):
    correlation_results = _orig_analyze_correlations(prefix_data_points)

    # Persist the raw correlation dict immediately.
    serializable = {}
    for name, info in correlation_results.items():
        ser = {}
        for k, v in info.items():
            try:
                ser[k] = float(v) if hasattr(v, "__float__") else v
            except Exception:
                ser[k] = v
        serializable[name] = ser

    robust.safe_save_json(
        {
            "model_name": "Qwen/Qwen3-1.7B",
            "n_prefix_data_points": len(prefix_data_points),
            "correlations": serializable,
        },
        os.path.join(QWEN3_OUT_DIR, "prefix_correlations_raw.json"),
    )

    # Also dump a flat numerical table that's easy to copy into LaTeX.
    rows = []
    for name, info in correlation_results.items():
        rows.append({
            "metric": name,
            "within_problem_corr": float(info.get("within_problem_corr", float("nan"))),
            "within_problem_p": float(info.get("within_problem_p", float("nan"))),
            "pooled_corr": float(info.get("corr_pass_rate", float("nan"))),
            "pooled_p": float(info.get("p_pass_rate", float("nan"))),
            "binary_corr": float(info.get("corr_binary", float("nan"))),
            "n_problems_used": int(info.get("n_problems_used", 0)),
        })
    rows.sort(key=lambda r: -abs(r["within_problem_corr"]))
    robust.safe_save_json(
        {"rows": rows},
        os.path.join(QWEN3_OUT_DIR, "prefix_correlations_table.json"),
    )

    # Also save in CSV form for direct LaTeX paste.
    import csv
    csv_path = os.path.join(QWEN3_OUT_DIR, "prefix_correlations_table.csv")
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"[robustness] saved CSV: {csv_path}")

    return correlation_results


mc_orig.analyze_correlations = _patched_analyze_correlations


# ----- Wrap each plotting function so a matplotlib failure is non-fatal -----
def _wrap_plot_fn(fn, label):
    def _safe(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[robustness] plot {label!r} failed: {e}")
            traceback.print_exc(file=sys.stdout)
            return None
    return _safe


mc_orig.plot_pass_rate_distribution = _wrap_plot_fn(
    mc_orig.plot_pass_rate_distribution, "pass_rate_distribution")
mc_orig.plot_metric_vs_pass_rate = _wrap_plot_fn(
    mc_orig.plot_metric_vs_pass_rate, "metric_vs_pass_rate")
mc_orig.plot_value_trajectories = _wrap_plot_fn(
    mc_orig.plot_value_trajectories, "value_trajectories")


def main():
    robust.dual_log(QWEN3_OUT_DIR)

    config = _patched_Config()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print(
        f"[qwen3] using {_n_gpus} GPU(s): "
        f"HF metric model on cuda:0, vLLM TP={TP_SIZE} on cuda:1..{_n_gpus - 1}"
    )

    # Replace Config so the legacy main() picks up our values.
    mc_orig.Config = _patched_Config

    # If matplotlib is available, also patch savefig to write a sidecar
    # data-only JSON next to each PNG.
    try:
        robust.patch_pyplot_savefig(QWEN3_OUT_DIR)
    except Exception as e:
        print(f"[robustness] could not patch pyplot.savefig: {e}")

    try:
        mc_orig.main()
    finally:
        mc_orig.Config = _OrigConfig

    print(f"\n[done] Qwen3 monte-carlo outputs written under: {QWEN3_OUT_DIR}")


if __name__ == "__main__":
    main()
