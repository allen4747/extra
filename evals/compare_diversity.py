#!/usr/bin/env python3
"""
Compare diversity metrics between an ExTra run and a GRPO baseline.

Usage:
    python3 evals/compare_diversity.py \
        --extra_dir eval_outputs_final/03_ExTra_RegenOnly_R1Distill_1.5B_step_150 \
        --grpo_dir  eval_outputs_final/01_GRPO_R1Distill_1.5B_step_150 \
        --out_csv   eval_outputs_final/diversity_03_vs_01_step150.csv

Reuses compute_diversity() and check_correct() from
verl/examples/experiments/ExTra_runs/eval_passk_diversity.py via importlib.
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Import helpers from eval_passk_diversity.py (digit-prefixed sibling files
# break normal imports, so we use importlib).
# ---------------------------------------------------------------------------
_EVAL_PASSK_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "verl", "examples", "experiments", "ExTra_runs", "eval_passk_diversity.py",
)
_EVAL_PASSK_PATH = os.path.abspath(_EVAL_PASSK_PATH)

spec = importlib.util.spec_from_file_location("eval_passk_diversity", _EVAL_PASSK_PATH)
_epd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_epd)

compute_diversity = _epd.compute_diversity
check_correct = _epd.check_correct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_eval_dir(eval_dir: str) -> dict[str, list[dict]]:
    """
    Read all gen_vllm jsonls in *eval_dir* and group rows by task.

    Returns {task_name: [{example_id, responses: [...], ground_truth}, ...]}.
    Each example_id collects all responses across seeds.
    """
    task_data: dict[str, dict[int, dict]] = defaultdict(dict)

    for jsonl_path in sorted(Path(eval_dir).glob("*.jsonl")):
        # Filename: {taskname}_t{T}_p{P}_n{N}-MNT{M}.jsonl
        task_name = jsonl_path.stem.split("_")[0].upper()
        # Normalize: math-500 -> MATH-500
        if task_name == "MATH":
            task_name = "MATH-500"

        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                eid = int(row["example_id"])
                if eid not in task_data[task_name]:
                    task_data[task_name][eid] = {
                        "responses": [],
                        "ground_truth": row["answer"],
                    }
                task_data[task_name][eid]["responses"].append(row["response"])

    # Convert to list format expected by compute_diversity / check_correct
    result = {}
    for task, examples in task_data.items():
        result[task] = list(examples.values())
    return result


def compute_pass1(items: list[dict]) -> float:
    """Compute pass@1 = fraction of problems where any response is correct."""
    if not items:
        return 0.0
    correct = 0
    for item in items:
        for r in item["responses"]:
            if check_correct(r, item["ground_truth"]):
                correct += 1
                break
    return correct / len(items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare diversity between ExTra and GRPO eval outputs"
    )
    parser.add_argument("--extra_dir", required=True, help="Path to ExTra eval output dir")
    parser.add_argument("--grpo_dir", required=True, help="Path to GRPO eval output dir")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    extra_data = load_eval_dir(args.extra_dir)
    grpo_data = load_eval_dir(args.grpo_dir)

    all_tasks = sorted(set(extra_data.keys()) | set(grpo_data.keys()))
    if not all_tasks:
        print("ERROR: No tasks found in either directory.", file=sys.stderr)
        sys.exit(1)

    # Extract method name and step from directory names
    extra_name = Path(args.extra_dir).name
    grpo_name = Path(args.grpo_dir).name

    rows = []
    fieldnames = [
        "method", "step", "task",
        "avg_cosine_distance", "std_cosine_distance",
        "avg_logdet_volume", "avg_correct_cosine_distance",
        "pass@1", "n_problems",
    ]

    for task in all_tasks:
        for label, data, dirname in [
            ("ExTra", extra_data, extra_name),
            ("GRPO", grpo_data, grpo_name),
        ]:
            items = data.get(task, [])
            if not items:
                continue
            div = compute_diversity(items)
            p1 = compute_pass1(items)

            # Parse step from dir name (e.g. "..._step_150")
            step = "?"
            parts = dirname.rsplit("_step_", 1)
            if len(parts) == 2:
                step = parts[1]

            rows.append({
                "method": label,
                "step": step,
                "task": task,
                "avg_cosine_distance": f"{div['avg_cosine_distance']:.4f}" if div["avg_cosine_distance"] is not None else "",
                "std_cosine_distance": f"{div['std_cosine_distance']:.4f}" if div["std_cosine_distance"] is not None else "",
                "avg_logdet_volume": f"{div['avg_logdet_volume']:.4f}" if div["avg_logdet_volume"] is not None else "",
                "avg_correct_cosine_distance": f"{div['avg_correct_cosine_distance']:.4f}" if div["avg_correct_cosine_distance"] is not None else "",
                "pass@1": f"{p1:.4f}",
                "n_problems": len(items),
            })

        # Delta row (ExTra - GRPO)
        extra_row = next((r for r in rows if r["task"] == task and r["method"] == "ExTra"), None)
        grpo_row = next((r for r in rows if r["task"] == task and r["method"] == "GRPO"), None)
        if extra_row and grpo_row:
            delta = {"method": "delta(ExTra-GRPO)", "step": extra_row["step"], "task": task}
            for col in ["avg_cosine_distance", "std_cosine_distance", "avg_logdet_volume",
                         "avg_correct_cosine_distance", "pass@1"]:
                try:
                    delta[col] = f"{float(extra_row[col]) - float(grpo_row[col]):+.4f}"
                except (ValueError, TypeError):
                    delta[col] = ""
            delta["n_problems"] = ""
            rows.append(delta)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")

    # Markdown table to stdout
    print()
    col_widths = {fn: max(len(fn), max((len(str(r.get(fn, ""))) for r in rows), default=0)) for fn in fieldnames}
    header = " | ".join(fn.ljust(col_widths[fn]) for fn in fieldnames)
    sep = "-|-".join("-" * col_widths[fn] for fn in fieldnames)
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(fn, "")).ljust(col_widths[fn]) for fn in fieldnames))


if __name__ == "__main__":
    main()
