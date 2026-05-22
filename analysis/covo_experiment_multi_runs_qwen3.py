#!/usr/bin/env python3
"""
Qwen3-1.7B variant of covo_experiment_multi_runs.py.

The original puts everything in `if __name__ == "__main__":` and hardcodes:
  - load_model()  (default Qwen2.5-1.5B-Instruct)
  - data_file = "new_collected_data_2b.pkl"  (Qwen2.5 cache)

This wrapper loads the source, rewrites those two lines so it uses
Qwen3-1.7B and a Qwen3-specific cache file, and execs.

Usage:
    cd analysis/
    python covo_experiment_multi_runs_qwen3.py
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QWEN3_OUT_DIR = HERE / "qwen3_outputs"
QWEN3_OUT_DIR.mkdir(exist_ok=True)
QWEN3_CACHE = QWEN3_OUT_DIR / "covo_collected_data_qwen3.pkl"


def main():
    src_path = HERE / "covo_experiment_multi_runs.py"
    src = src_path.read_text()

    # Rewrite the model name passed to load_model() in __main__
    src = src.replace(
        "model, tokenizer = load_model()",
        'model, tokenizer = load_model(model_name="Qwen/Qwen3-1.7B")',
    )
    # Rewrite the cache path
    src = src.replace(
        'data_file = os.path.join(os.path.dirname(__file__), "new_collected_data_2b.pkl")',
        f'data_file = r"{QWEN3_CACHE}"',
    )

    sys.path.insert(0, str(HERE))
    # Install shims for missing legacy deps so the exec'd source resolves
    # `simplified_evaluator` and `openrlhf` imports.
    import _qwen3_shims  # noqa: F401
    ns = {"__name__": "__main__", "__file__": str(src_path)}
    exec(compile(src, str(src_path), "exec"), ns)

    print(f"\n[done] Qwen3 covo experiment finished. Cache: {QWEN3_CACHE}")


if __name__ == "__main__":
    main()
