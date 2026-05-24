"""
Robustness helpers for the Qwen3 analysis wrappers.

Provides:
  - dual_log(out_dir): redirects stdout/stderr to both terminal and a
    timestamped log file inside out_dir.  Idempotent across multiple
    wrappers running in the same process / shell.
  - safe_save_json(obj, path): writes JSON atomically (tmp + rename)
    so partial writes never corrupt files.
  - safe_pickle(obj, path): same for pickle.
  - patch_pyplot(out_dir, prefix=""): replaces matplotlib.pyplot.savefig
    with a wrapper that catches any error and instead dumps the figure's
    underlying axes data to a sidecar JSON.  This way even if matplotlib
    is broken, you still have the numbers.
  - safe_plot(callable_, out_dir, fallback_name): runs a plotting fn
    inside a try/except; on failure, logs and continues.
"""
import json
import os
import pickle
import sys
import traceback
from datetime import datetime


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def dual_log(out_dir):
    """Tee stdout & stderr to a timestamped log file inside out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"run_{ts}.log")
    fh = open(log_path, "a", buffering=1)  # line buffered
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[robustness] logging to {log_path}")
    return log_path


def safe_save_json(obj, path):
    """Atomic JSON write: write to .tmp then rename."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[robustness] saved JSON: {path}")


def safe_pickle(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
    print(f"[robustness] saved PKL : {path}")


def safe_plot(callable_, label="(plot)"):
    """Run a plotting function; on error, log traceback and continue."""
    try:
        return callable_()
    except Exception as e:
        print(f"[robustness] WARNING: plot {label!r} failed: {e}")
        traceback.print_exc(file=sys.stdout)
        return None


def patch_pyplot_savefig(record_dir):
    """Wrap pyplot.savefig so each call writes a sidecar JSON listing the
    arrays from each Axes (line data, scatter offsets) and the title.
    The PNG/PDF still gets written if matplotlib succeeds; if it fails,
    at least the data is preserved.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[robustness] matplotlib not available; cannot patch savefig")
        return

    os.makedirs(record_dir, exist_ok=True)
    _orig = plt.savefig

    def _wrapped(fname, *args, **kwargs):
        # Try the real savefig.
        try:
            result = _orig(fname, *args, **kwargs)
        except Exception as e:
            print(f"[robustness] savefig({fname}) failed: {e}")
            result = None
        # Always also dump the underlying data so the user can re-plot.
        try:
            sidecar = {"path": str(fname), "axes": []}
            fig = plt.gcf()
            for ax in fig.get_axes():
                ax_data = {
                    "title": ax.get_title(),
                    "xlabel": ax.get_xlabel(),
                    "ylabel": ax.get_ylabel(),
                    "lines": [],
                    "scatter": [],
                }
                for ln in ax.get_lines():
                    ax_data["lines"].append({
                        "label": ln.get_label(),
                        "x": ln.get_xdata().tolist(),
                        "y": ln.get_ydata().tolist(),
                    })
                for coll in ax.collections:
                    try:
                        offs = coll.get_offsets()
                        ax_data["scatter"].append({
                            "x": offs[:, 0].tolist(),
                            "y": offs[:, 1].tolist(),
                        })
                    except Exception:
                        pass
                sidecar["axes"].append(ax_data)
            base = os.path.basename(str(fname))
            sidecar_path = os.path.join(record_dir, f"{base}.data.json")
            with open(sidecar_path, "w") as f:
                json.dump(sidecar, f, default=str)
            print(f"[robustness] saved figure-data sidecar: {sidecar_path}")
        except Exception as e:
            print(f"[robustness] sidecar dump for {fname} failed: {e}")
        return result

    plt.savefig = _wrapped
