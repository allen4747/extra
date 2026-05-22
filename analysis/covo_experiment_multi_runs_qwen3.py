#!/usr/bin/env python3
"""
Qwen3-1.7B variant of covo_experiment_multi_runs.py.

The original script:
  - Uses HF generate only (no vLLM); a single GPU is the natural unit.
  - Hardcodes the model on `cuda:3`, which assumes a 4-GPU layout.

This wrapper rewrites the model device to `cuda:0` (the first visible
GPU after CUDA_VISIBLE_DEVICES filtering) and points the cache file at
analysis/qwen3_outputs/.  Note: covo cannot meaningfully use multiple
GPUs because the HF generate call is per-prompt with batch=n_samples;
splitting the 1.7B model across cards would actually be slower due to
communication overhead.  We therefore use only cuda:0 here.

Usage:
    cd analysis/
    CUDA_VISIBLE_DEVICES=0 python covo_experiment_multi_runs_qwen3.py
        # or  CUDA_VISIBLE_DEVICES=0,1,2,3 ...  (only cuda:0 is used)
"""

import os
import sys
from pathlib import Path

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HERE = Path(__file__).resolve().parent
QWEN3_OUT_DIR = HERE / "qwen3_outputs"
QWEN3_OUT_DIR.mkdir(exist_ok=True)
QWEN3_CACHE = QWEN3_OUT_DIR / "covo_collected_data_qwen3.pkl"


def main():
    src_path = HERE / "covo_experiment_multi_runs.py"
    src = src_path.read_text()

    # Inject the Qwen3 model name into the load_model() call in __main__.
    src = src.replace(
        "model, tokenizer = load_model()",
        'model, tokenizer = load_model(model_name="Qwen/Qwen3-1.7B")',
    )
    # Move the HF model from cuda:3 to cuda:0 (works regardless of how
    # many GPUs are visible).
    src = src.replace('device_map="cuda:3"', 'device_map="cuda:0"')
    # Redirect the cache to qwen3_outputs/.
    src = src.replace(
        'data_file = os.path.join(os.path.dirname(__file__), "new_collected_data_2b.pkl")',
        f'data_file = r"{QWEN3_CACHE}"',
    )

    sys.path.insert(0, str(HERE))
    # Install shims so the exec'd source resolves simplified_evaluator
    # and openrlhf imports.
    import _qwen3_shims  # noqa: F401

    print(f"[qwen3] using cuda:0 (covo experiment cannot benefit from multi-GPU)")
    ns = {"__name__": "__main__", "__file__": str(src_path)}
    exec(compile(src, str(src_path), "exec"), ns)

    print(f"\n[done] Qwen3 covo experiment finished. Cache: {QWEN3_CACHE}")


if __name__ == "__main__":
    main()
