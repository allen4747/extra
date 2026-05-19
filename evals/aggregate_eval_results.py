#!/usr/bin/env python3
"""
Aggregate metrics.json files from multiple servers into paper-ready tables.

Usage:
    python aggregate_eval_results.py --results_dir ~/extra_eval_results --output paper_table

Reads all metrics.json files under results_dir/{RUN_NAME}/step_{N}/metrics.json
and outputs:
  - {output}_pass1.csv          : pass@1 (mean correctness) per benchmark
  - {output}_passk.csv          : pass@k (k=n_samples, best-of-n) per benchmark
  - {output}_combined.csv       : both metrics side-by-side
  - {output}_full.csv           : all metrics (pass@1, pass@k, distinct_4gram, avg_len)
And prints a pretty side-by-side table to stdout showing pass@1 and pass@k.
"""

import argparse
import json
from pathlib import Path
import csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, help="Root dir containing {run}/step_{N}/metrics.json")
    p.add_argument("--output", default="paper_table", help="Output prefix (without .csv)")
    args = p.parse_args()

    root = Path(args.results_dir)
    metrics_files = sorted(root.glob("*/step_*/metrics.json"))

    if not metrics_files:
        print(f"No metrics.json found under {root}")
        return

    rows = []
    all_benchmarks = set()
    n_samples_seen = set()
    for mf in metrics_files:
        try:
            data = json.loads(mf.read_text())
        except Exception as e:
            print(f"[skip] {mf}: {e}")
            continue

        n = data.get("n_samples", 16)
        n_samples_seen.add(n)
        row = {
            "run_name": data.get("run_name", mf.parent.parent.name),
            "step": data.get("step", int(mf.parent.name.replace("step_", ""))),
            "n_samples": n,
        }
        for bench, scores in data.get("benchmarks", {}).items():
            all_benchmarks.add(bench)
            row[f"{bench}_pass@1"] = scores.get("avg@n")
            row[f"{bench}_pass@k"] = scores.get("best@n")
            row[f"{bench}_distinct4"] = scores.get("distinct_4gram")
            row[f"{bench}_avglen"] = scores.get("avg_output_length")
        rows.append(row)

    rows.sort(key=lambda r: (r["run_name"], r["step"]))
    bench_order = sorted(all_benchmarks)

    # Pick a label for pass@k header: if all runs use same N, label is "pass@N"
    if len(n_samples_seen) == 1:
        k_label = f"pass@{next(iter(n_samples_seen))}"
    else:
        k_label = "pass@k"

    # --- 1. Combined CSV (pass@1 + pass@k side-by-side) ---
    base_cols = ["run_name", "step", "n_samples"]
    combined_cols = base_cols.copy()
    for b in bench_order:
        combined_cols += [f"{b}_pass@1", f"{b}_{k_label}"]
    out_path = Path(f"{args.output}_combined.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=combined_cols)
        w.writeheader()
        for row in rows:
            mapped = {k: row.get(k.replace(f"_{k_label}", "_pass@k"), "") for k in combined_cols}
            mapped.update({k: row.get(k, "") for k in base_cols + [c for c in combined_cols if "pass@1" in c]})
            w.writerow(mapped)
    print(f"[+] Wrote {out_path}")

    # --- 2. Pass@1 only ---
    p1_cols = base_cols + [f"{b}_pass@1" for b in bench_order]
    out_path = Path(f"{args.output}_pass1.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=p1_cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in p1_cols})
    print(f"[+] Wrote {out_path}")

    # --- 3. Pass@k only ---
    pk_cols = base_cols + [f"{b}_{k_label}" for b in bench_order]
    out_path = Path(f"{args.output}_passk.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pk_cols)
        w.writeheader()
        for row in rows:
            mapped = {k: row.get(k, "") for k in base_cols}
            for b in bench_order:
                mapped[f"{b}_{k_label}"] = row.get(f"{b}_pass@k", "")
            w.writerow(mapped)
    print(f"[+] Wrote {out_path}")

    # --- 4. Full CSV (everything) ---
    full_cols = base_cols.copy()
    for b in bench_order:
        full_cols += [f"{b}_pass@1", f"{b}_{k_label}", f"{b}_distinct4", f"{b}_avglen"]
    out_path = Path(f"{args.output}_full.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=full_cols)
        w.writeheader()
        for row in rows:
            mapped = {k: row.get(k, "") for k in base_cols}
            for b in bench_order:
                mapped[f"{b}_pass@1"] = row.get(f"{b}_pass@1", "")
                mapped[f"{b}_{k_label}"] = row.get(f"{b}_pass@k", "")
                mapped[f"{b}_distinct4"] = row.get(f"{b}_distinct4", "")
                mapped[f"{b}_avglen"] = row.get(f"{b}_avglen", "")
            w.writerow(mapped)
    print(f"[+] Wrote {out_path}")

    # --- Pretty-print combined table ---
    print()
    print("=" * 100)
    print(f"  pass@1 / {k_label} SIDE-BY-SIDE  ({len(rows)} runs, {len(bench_order)} benchmarks)")
    print("=" * 100)

    name_col = max(len("Run"), max((len(r["run_name"]) for r in rows), default=10))
    header1 = f"{'Run':<{name_col}} {'Step':>5}"
    for b in bench_order:
        header1 += f"  {b[:14]:^14}"
    print(header1)
    header2 = " " * (name_col + 6)
    for b in bench_order:
        header2 += f"   {'p@1':>6} {k_label[-4:] if k_label[-4:].startswith('p@') else 'p@k':>6}"
    print(header2)
    print("-" * len(header1))

    for row in rows:
        line = f"{row['run_name']:<{name_col}} {row['step']:>5}"
        for b in bench_order:
            p1 = row.get(f"{b}_pass@1")
            pk = row.get(f"{b}_pass@k")
            p1_s = f"{p1:.3f}" if isinstance(p1, (int, float)) else "  ---"
            pk_s = f"{pk:.3f}" if isinstance(pk, (int, float)) else "  ---"
            line += f"   {p1_s:>6} {pk_s:>6}"
        print(line)
    print()


if __name__ == "__main__":
    main()
