#!/usr/bin/env python3
"""
Qwen3-1.7B variant of entropy_signal_analysis.py.

This script consumes prefix_metrics_data.pkl produced by the monte_carlo
experiment.  Run monte_carlo_experiment_qwen3.py FIRST so the Qwen3
prefix metrics are available, then run this.

Usage:
    cd analysis/
    python monte_carlo_experiment_qwen3.py        # produce qwen3_outputs/prefix_metrics_data.pkl
    python entropy_signal_analysis_qwen3.py       # consume it
"""

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QWEN3_OUT_DIR = HERE / "qwen3_outputs"
QWEN3_OUT_DIR.mkdir(exist_ok=True)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    # The original script reads DATA_PATH = analysis/prefix_metrics_data.pkl
    # We override it (and OUT_DIR) to point at qwen3_outputs/.
    orig_path = HERE / "entropy_signal_analysis.py"
    if not orig_path.exists():
        print(f"ERROR: cannot find {orig_path}")
        sys.exit(1)

    # We can't trivially monkey-patch module-level Path constants before main()
    # runs (they're evaluated at import time).  Easiest workaround: load source,
    # rewrite the two constants in-memory, exec.
    src = orig_path.read_text()

    src = src.replace(
        'DATA_PATH = Path(__file__).parent / "prefix_metrics_data.pkl"',
        f'DATA_PATH = Path(r"{QWEN3_OUT_DIR}") / "2b_prefix_metrics_data.pkl"',
    )
    src = src.replace(
        "OUT_DIR = Path(__file__).parent",
        f'OUT_DIR = Path(r"{QWEN3_OUT_DIR}")',
    )

    # Make the original module's other relative imports still resolve.
    sys.path.insert(0, str(HERE))

    # Robustness helpers: dual_log + savefig sidecar.
    import _qwen3_robust as robust  # noqa: F401
    robust.dual_log(str(QWEN3_OUT_DIR))
    try:
        robust.patch_pyplot_savefig(str(QWEN3_OUT_DIR))
    except Exception as e:
        print(f"[robustness] could not patch pyplot.savefig: {e}")

    ns = {"__name__": "__main__", "__file__": str(orig_path)}
    try:
        exec(compile(src, str(orig_path), "exec"), ns)
    except Exception as e:
        import traceback
        print(f"[robustness] entropy_signal exec failed: {e}")
        traceback.print_exc()

    print(f"\n[done] Qwen3 entropy-signal outputs written under: {QWEN3_OUT_DIR}")


if __name__ == "__main__":
    main()
