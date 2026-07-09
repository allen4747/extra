#!/usr/bin/env python3
"""
mte_gap_summary.py — EMNLP rebuttal analysis A2.

Consumes outputs/mte_gap_log.jsonl written by the patched ray_trainer.py
(active only when algorithm.guided_resampling.log_mte_gap=True).

Answers reviewer concerns:
  * C2: XB9Q W3/Q3 and xvYm W2 — "MTE ρ=-0.229 is weak; how much better is
    MTE-selection than random online?"  We report the mean entropy gap
    between MTE-selected and a uniformly random alternative prefix (lower =
    better), split by early / mid / late training.
  * C8 (XB9Q Q4): stale prefixes — median prefix_memory size + queue size at
    consumption time.

Note on scope: this script computes the *entropy* gap directly. To convert
that into a *continuation pass-rate* gap you also want to run the companion
`mte_gap_offline_passrate.py` (not in this diff — see runbook fallback)
against saved checkpoints, which is more expensive but strictly stronger
evidence. For the rebuttal, the online entropy gap is sufficient because
Sec.5.6 of the paper already ties MTE score to Monte-Carlo pass rate.

Usage:
  python analysis/rebuttal/mte_gap_summary.py \
    --log outputs/mte_gap_log.jsonl \
    --out mte_gap_summary.md
"""

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def summarize(rows: list[dict], label: str) -> str:
    if not rows:
        return f"### {label}\n\n(no rows)\n"
    sel = [r["mte_selected"]["prefix_entropy_mean"] for r in rows]
    rnd = [r["random_alt"]["prefix_entropy_mean"] for r in rows]
    # Paired gap: MTE-selected minus random. Negative = MTE picks lower-entropy
    # (i.e. better) prefixes, which is what we want.
    gaps = [s - r for s, r in zip(sel, rnd)]
    n = len(gaps)
    mean_gap = sum(gaps) / n
    var = sum((g - mean_gap) ** 2 for g in gaps) / max(n - 1, 1)
    se = math.sqrt(var / n) if n > 1 else float("nan")
    # 95% Wald CI on the mean.
    ci_lo = mean_gap - 1.96 * se
    ci_hi = mean_gap + 1.96 * se
    # Fraction where MTE strictly picked a lower-entropy prefix.
    win = sum(1 for g in gaps if g < 0) / n
    mem = [r["prefix_memory_size"] for r in rows]
    q = [r["queue_size_before"] for r in rows]
    return (
        f"### {label}\n\n"
        f"* rows: {n}\n"
        f"* mean entropy gap (MTE − random): {mean_gap:+.4f} "
        f"(95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}])\n"
        f"* fraction MTE < random: {win:.3f}\n"
        f"* prefix_memory_size: median {sorted(mem)[n // 2]}, "
        f"90th %ile {sorted(mem)[int(0.9 * (n - 1))]}\n"
        f"* queue_size_before: median {sorted(q)[n // 2]}, "
        f"90th %ile {sorted(q)[int(0.9 * (n - 1))]}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="mte_gap_summary.md")
    args = ap.parse_args()

    rows = load(Path(args.log))
    if not rows:
        print(f"[mte_gap_summary] no rows in {args.log}")
        return

    steps = sorted({int(r["step"]) for r in rows if r.get("step", -1) >= 0})
    if not steps:
        print("[mte_gap_summary] no valid step values")
        return
    lo, hi = steps[0], steps[-1]
    third = max((hi - lo) // 3, 1)
    early_end = lo + third
    mid_end = lo + 2 * third

    early = [r for r in rows if r["step"] <= early_end]
    mid = [r for r in rows if early_end < r["step"] <= mid_end]
    late = [r for r in rows if r["step"] > mid_end]

    parts = [
        "# MTE-gap summary (online, from mte_gap_log.jsonl)\n\n",
        f"Log file: `{args.log}`  |  step range: [{lo}, {hi}]\n\n",
        "Sign convention: entropy gap = MTE_selected − random_alt.  A **negative**\n"
        "gap means the MTE heuristic picked a lower-entropy (predicted more\n"
        "promising) prefix than the random alternative, which is the operational\n"
        "condition required by Prop. 1 in the paper.\n\n",
        summarize(rows, "All steps"),
        summarize(early, f"Early training (step ≤ {early_end})"),
        summarize(mid, f"Mid training ({early_end} < step ≤ {mid_end})"),
        summarize(late, f"Late training (step > {mid_end})"),
    ]
    out_text = "\n".join(parts)
    Path(args.out).write_text(out_text)
    print(f"[mte_gap_summary] wrote {args.out}")
    print(out_text)


if __name__ == "__main__":
    main()
