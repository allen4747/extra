#!/usr/bin/env python3
"""
Aggregate metrics.json files from multiple servers into a single paper table.

Usage:
    python aggregate_eval_results.py --results_dir ~/extra_eval_results --output paper_table.csv

Reads all metrics.json files under results_dir/{RUN_NAME}/step_{N}/metrics.json
and outputs a CSV ready for the paper.
"""

import argparse
import json
from pathlib import Path
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="Root dir containing {run}/step_{N}/metrics.json")
    p.add_argument("--output", default="paper_table.csv", help="Output CSV path")
    args = p.parse_args()

    root = Path(args.results_dir)
    metrics_files = sorted(root.glob("*/step_*/metrics.json"))

    if not metrics_files:
        print(f"No metrics.json found under {root}")
        return

    rows = []
    all_benchmarks = set()
    for mf in metrics_files:
        try:
            data = json.loads(mf.read_text())
        except Exception as e:
            print(f"[skip] {mf}: {e}")
            continue

        row = {
            "run_name": data.get("run_name", mf.parent.parent.name),
            "step": data.get("step", int(mf.parent.name.replace("step_", ""))),
            "n_samples": data.get("n_samples", 32),
        }
        for bench, scores in data.get("benchmarks", {}).items():
            all_benchmarks.add(bench)
            row[f"{bench}_avg"] = scores.get("avg@n")
            row[f"{bench}_best"] = scores.get("best@n")
            row[f"{bench}_distinct4"] = scores.get("distinct_4gram")
            row[f"{bench}_avglen"] = scores.get("avg_output_length")
        rows.append(row)

    # Build column order
    base_cols = ["run_name", "step", "n_samples"]
    bench_cols = []
    for b in sorted(all_benchmarks):
        bench_cols += [f"{b}_avg", f"{b}_best", f"{b}_distinct4", f"{b}_avglen"]
    cols = base_cols + bench_cols

    out_path = Path(args.output)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})

    print(f"Wrote {out_path} with {len(rows)} rows, {len(bench_cols)} benchmark columns")
    print()

    # Pretty-print summary
    print(f"{'Run':<40} {'Step':>5}", end="")
    for b in sorted(all_benchmarks):
        print(f" {b[:8]:>10}", end="")
    print()
    print("-" * (45 + 11 * len(all_benchmarks)))
    for row in sorted(rows, key=lambda r: (r["run_name"], r["step"])):
        print(f"{row['run_name'][:40]:<40} {row['step']:>5}", end="")
        for b in sorted(all_benchmarks):
            v = row.get(f"{b}_avg")
            print(f" {v if v is not None else '---':>10}", end="")
        print()


if __name__ == "__main__":
    main()
