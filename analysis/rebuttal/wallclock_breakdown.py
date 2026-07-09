#!/usr/bin/env python3
"""
wallclock_breakdown.py — EMNLP rebuttal analysis A3.

Reports per-step wall-clock time for each run, and % overhead of ExTra vs
the paired GRPO baseline. Answers XB9Q W5/Q5.

Two data sources tried in order:
  1. Weights & Biases (via `wandb.Api()`), if `wandb` is importable and
     `WANDB_ENTITY`/`WANDB_PROJECT` are set. Reads `perf/time_per_step` (which
     verl.trainer.ppo.metric_utils.compute_throughput populates).
  2. Fallback: parse `outputs/*/main_ppo.log` timestamps to get elapsed time
     per step. This is coarser but works offline.

Usage:
  python analysis/rebuttal/wallclock_breakdown.py \
    --wandb_project ExTra_Rebuttal \
    --pairs 01_GRPO_NanoNemotron_8B,02_ExTra_Full_NanoNemotron_8B \
    --pairs 03_GRPO_Qwen3_seed2,04_ExTra_Full_Qwen3_seed2 \
    --out wallclock.csv

  # Offline fallback (no wandb):
  python analysis/rebuttal/wallclock_breakdown.py \
    --log_glob 'outputs/**/main_ppo.log' \
    --out wallclock.csv
"""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median


TIME_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})[.,]?\d*.*?(?:step|iter)[:= ]*(?P<step>\d+)",
    re.IGNORECASE,
)


def parse_log_timestamps(log_path: Path) -> list[tuple[int, float]]:
    """Return [(step, epoch_seconds), …] parsed from a training log."""
    import datetime as _dt

    out: list[tuple[int, float]] = []
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="ignore").splitlines():
        m = TIME_RE.search(line)
        if not m:
            continue
        try:
            ts = _dt.datetime.strptime(m.group("ts")[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
            out.append((int(m.group("step")), ts.timestamp()))
        except Exception:
            continue
    return out


def per_step_dt(pairs: list[tuple[int, float]]) -> list[float]:
    pairs = sorted(pairs)
    dt = []
    prev_step, prev_t = None, None
    for step, t in pairs:
        if prev_t is not None and step > (prev_step or -1):
            dt.append(t - prev_t)
        prev_step, prev_t = step, t
    # Drop the first outlier (val_before_train may inflate step 1).
    return [d for d in dt if d > 0][1:] if len(dt) > 1 else dt


def try_wandb(project: str, run_name: str) -> list[float] | None:
    try:
        import wandb
        api = wandb.Api()
    except Exception:
        return None
    entity = None  # let API default
    try:
        runs = api.runs(f"{entity + '/' if entity else ''}{project}", {"display_name": run_name})
    except Exception:
        return None
    dts = []
    for run in runs:
        try:
            hist = run.history(keys=["perf/time_per_step"], pandas=False)
        except Exception:
            continue
        for row in hist:
            v = row.get("perf/time_per_step")
            if v is not None:
                dts.append(float(v))
    return dts or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument(
        "--pairs",
        action="append",
        default=[],
        help='Comma-separated "grpo_run,extra_run" pairs. Repeat for multiple.',
    )
    ap.add_argument("--log_glob", default=None, help="Glob for main_ppo.log files (offline mode)")
    ap.add_argument("--out", default="wallclock.csv")
    args = ap.parse_args()

    rows = []

    if args.wandb_project and args.pairs:
        for pair in args.pairs:
            grpo, extra = [x.strip() for x in pair.split(",")]
            grpo_dts = try_wandb(args.wandb_project, grpo) or []
            extra_dts = try_wandb(args.wandb_project, extra) or []
            g_med = median(grpo_dts) if grpo_dts else float("nan")
            e_med = median(extra_dts) if extra_dts else float("nan")
            overhead = (e_med - g_med) / g_med if g_med else float("nan")
            rows.append(
                {
                    "source": "wandb",
                    "grpo_run": grpo,
                    "extra_run": extra,
                    "grpo_median_step_s": round(g_med, 2),
                    "extra_median_step_s": round(e_med, 2),
                    "overhead_frac": round(overhead, 3),
                    "n_grpo": len(grpo_dts),
                    "n_extra": len(extra_dts),
                }
            )

    if args.log_glob:
        for log_path in sorted(Path().glob(args.log_glob)):
            pairs = parse_log_timestamps(log_path)
            dts = per_step_dt(pairs)
            if not dts:
                continue
            rows.append(
                {
                    "source": "log",
                    "log_path": str(log_path),
                    "median_step_s": round(median(dts), 2),
                    "mean_step_s": round(mean(dts), 2),
                    "n_steps": len(dts),
                }
            )

    if not rows:
        print("[wallclock] no data (provide --wandb_project+--pairs or --log_glob)")
        return

    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[wallclock] wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
