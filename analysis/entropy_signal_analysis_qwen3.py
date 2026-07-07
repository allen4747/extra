#!/usr/bin/env python3
"""
Qwen3-1.7B variant of entropy_signal_analysis.py.

This script consumes prefix_metrics_data.pkl produced by the monte
carlo experiment.  Run monte_carlo_experiment_qwen3.py FIRST so the
Qwen3 prefix metrics are available, then run this.

The rebuilt monte_carlo_experiment_qwen3.py writes both `prefix_mte`
(new canonical name, matches training's guided-resampling scoring)
AND `prefix_entropy_mean` (legacy alias) into each data point, so the
legacy entropy-signal analysis code here still runs unmodified against
either name.

We add a schema-version guard: if the on-disk .pkl was produced by an
older monte_carlo_experiment_qwen3.py (pre-training-alignment), refuse
to run and instruct the user to delete + rerun.

Usage:
    cd analysis/
    python monte_carlo_experiment_qwen3.py     # produce qwen3_outputs/2b_prefix_metrics_data.pkl
    python entropy_signal_analysis_qwen3.py    # consume it
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QWEN3_OUT_DIR = HERE / "qwen3_outputs"
QWEN3_OUT_DIR.mkdir(exist_ok=True)

# Must match monte_carlo_experiment_qwen3.SCHEMA_VERSION.
EXPECTED_SCHEMA_VERSION = 2
SCHEMA_MARKER_FILE = QWEN3_OUT_DIR / "2b_prefix_metrics_data.schema"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_schema() -> None:
    pkl = QWEN3_OUT_DIR / "2b_prefix_metrics_data.pkl"
    if not pkl.exists():
        print(f"[qwen3] ERROR: {pkl} not found.  Run "
              f"monte_carlo_experiment_qwen3.py first.")
        sys.exit(1)

    if not SCHEMA_MARKER_FILE.exists():
        print(f"[qwen3] ERROR: schema marker {SCHEMA_MARKER_FILE} missing.\n"
              f"  The .pkl was produced by an older "
              f"monte_carlo_experiment_qwen3.py.\n"
              f"  Delete these files and rerun the monte-carlo step:\n"
              f"    rm {pkl}\n"
              f"    rm {QWEN3_OUT_DIR / '2b_prefix_value_data.pkl'}\n"
              f"    rm {QWEN3_OUT_DIR / '2b_prefix_collected_data.pkl'}")
        sys.exit(1)

    try:
        marker = json.loads(SCHEMA_MARKER_FILE.read_text())
        actual = int(marker.get("schema_version", 0))
    except Exception as e:
        print(f"[qwen3] ERROR: could not parse schema marker: {e}")
        sys.exit(1)

    if actual != EXPECTED_SCHEMA_VERSION:
        print(f"[qwen3] ERROR: schema version mismatch "
              f"(expected {EXPECTED_SCHEMA_VERSION}, got {actual}).\n"
              f"  Delete {pkl} and rerun.")
        sys.exit(1)

    print(f"[qwen3] schema v{actual}, "
          f"strip_thinking={marker.get('strip_thinking', 'unknown')}")


def main():
    _check_schema()

    orig_path = HERE / "entropy_signal_analysis.py"
    if not orig_path.exists():
        print(f"ERROR: cannot find {orig_path}")
        sys.exit(1)

    # We can't trivially monkey-patch module-level Path constants
    # before main() runs (they're evaluated at import time).  Easiest
    # workaround: load source, rewrite the two constants in memory,
    # then exec.
    src = orig_path.read_text()

    src = src.replace(
        'DATA_PATH = Path(__file__).parent / "prefix_metrics_data.pkl"',
        f'DATA_PATH = Path(r"{QWEN3_OUT_DIR}") / "2b_prefix_metrics_data.pkl"',
    )
    src = src.replace(
        "OUT_DIR = Path(__file__).parent",
        f'OUT_DIR = Path(r"{QWEN3_OUT_DIR}")',
    )

    sys.path.insert(0, str(HERE))

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
