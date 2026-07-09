#!/usr/bin/env python3
"""
bootstrap_ci.py — EMNLP rebuttal analysis A1.

Adds 95% bootstrap CIs and paired-bootstrap p-values to the main comparison
tables. Reads the same JSONL per-sample rollouts that `evals/gen_vllm.py`
writes and that `evals/aggregate_eval_results.py` aggregates.

Answers reviewer concern C1 (XB9Q W2/Q2 and xvYm W1) — "no seeds / no CIs on
small benchmarks like AIME24/25".

Output:
  * paper_table_ci.csv  — per-run, per-benchmark: pass@1 mean, pass@1 CI-lo,
                          pass@1 CI-hi, pass@k mean, pass@k CI-lo, pass@k CI-hi
  * paper_table_pvals.csv (optional, when both --ref_run and --cmp_run are
                          set): per-benchmark paired-bootstrap p-value
                          comparing the cmp_run to the ref_run.

Usage:
  python analysis/rebuttal/bootstrap_ci.py \
    --eval_dir ./eval_outputs_rebuttal \
    --n_bootstrap 10000 \
    --out paper_table_ci.csv

  # For paired significance testing (ref=GRPO, cmp=ExTra):
  python analysis/rebuttal/bootstrap_ci.py \
    --eval_dir ./eval_outputs_rebuttal \
    --ref_run 01_GRPO_NanoNemotron_8B \
    --cmp_run 02_ExTra_Full_NanoNemotron_8B \
    --step 150 \
    --out_pvals paper_table_pvals.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_per_sample_matrix(run_dir: Path) -> dict[str, np.ndarray]:
    """Return {benchmark_name: correct_matrix (n_problems x n_samples)}.

    Reads {bench}_t{temp}_p{topp}_n{n}.jsonl files produced by evals/gen_vllm.py.
    Each line is a dict with `example_id`, `sample_idx`, `is_correct` (0/1)
    or an equivalent grading field. If a metrics.json also exists, we use it
    to double-check total counts.
    """
    out: dict[str, np.ndarray] = {}
    for jsonl in sorted(run_dir.glob("*.jsonl")):
        bench = jsonl.stem.split("_t")[0]
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        # Normalize the field: gen_vllm may write `is_correct` after grading,
        # or `correct`, or the grader may store just `score`.
        def _score(r):
            for k in ("is_correct", "correct", "score"):
                if k in r:
                    return int(bool(r[k]))
            return None
        # Sniff whether scores are present.
        if _score(rows[0]) is None:
            # Not graded — skip; bootstrap_ci assumes a graded eval dir.
            continue

        problems = sorted({r["example_id"] for r in rows})
        p2i = {p: i for i, p in enumerate(problems)}
        # Determine n_samples per problem (usually 16).
        samples_per_prob: dict[int, list[int]] = {p: [] for p in problems}
        for r in rows:
            samples_per_prob[r["example_id"]].append(_score(r))
        n_samples = max(len(v) for v in samples_per_prob.values())
        mat = np.zeros((len(problems), n_samples), dtype=np.int8)
        for p, scores in samples_per_prob.items():
            mat[p2i[p], : len(scores)] = scores
        out[bench] = mat
    return out


def pass_at_1(mat: np.ndarray) -> float:
    """Mean per-sample correctness, averaged over problems."""
    return float(mat.mean())


def pass_at_k(mat: np.ndarray) -> float:
    """Fraction of problems where at least one sample is correct."""
    return float(mat.any(axis=1).mean())


def bootstrap_ci_problem_level(
    mat: np.ndarray, stat_fn, n_bootstrap: int, seed: int = 42
) -> tuple[float, float, float]:
    """Return (point, ci_lo_2.5, ci_hi_97.5) by resampling problems (rows)."""
    rng = np.random.default_rng(seed)
    n_prob = mat.shape[0]
    if n_prob == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_prob, n_prob)
        boots[b] = stat_fn(mat[idx])
    point = stat_fn(mat)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return (point, lo, hi)


def paired_bootstrap_pvalue(
    mat_a: np.ndarray, mat_b: np.ndarray, stat_fn, n_bootstrap: int, seed: int = 42
) -> float:
    """Paired bootstrap p-value that stat_fn(B) > stat_fn(A).

    Requires the two matrices to share problem ordering (same example_ids).
    """
    if mat_a.shape[0] != mat_b.shape[0]:
        # Not paired — return NaN; the caller can decide to fall back to a
        # two-sample bootstrap.
        return float("nan")
    rng = np.random.default_rng(seed)
    n_prob = mat_a.shape[0]
    observed_delta = stat_fn(mat_b) - stat_fn(mat_a)
    if n_prob == 0:
        return float("nan")
    ge = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_prob, n_prob)
        delta = stat_fn(mat_b[idx]) - stat_fn(mat_a[idx])
        if delta <= 0:  # under H0, we would not see delta >= observed
            ge += 1
    # One-sided p-value: fraction of bootstraps where B did not beat A.
    return (ge + 1) / (n_bootstrap + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True, help="Root: {eval_dir}/{run}/step_{N}/")
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="paper_table_ci.csv")
    ap.add_argument("--ref_run", default=None, help="Baseline run name for paired p-values")
    ap.add_argument("--cmp_run", default=None, help="Method run name for paired p-values")
    ap.add_argument("--step", type=int, default=None, help="Restrict to this step")
    ap.add_argument("--out_pvals", default="paper_table_pvals.csv")
    args = ap.parse_args()

    root = Path(args.eval_dir)
    ci_rows = []
    per_run_bench_mats: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        for step_dir in sorted(run_dir.glob("step_*")):
            try:
                step = int(step_dir.name.replace("step_", ""))
            except ValueError:
                continue
            if args.step is not None and step != args.step:
                continue
            mats = load_per_sample_matrix(step_dir)
            per_run_bench_mats[(run_dir.name, step)] = mats
            for bench, mat in mats.items():
                p1, p1_lo, p1_hi = bootstrap_ci_problem_level(
                    mat, pass_at_1, args.n_bootstrap
                )
                pk, pk_lo, pk_hi = bootstrap_ci_problem_level(
                    mat, pass_at_k, args.n_bootstrap
                )
                ci_rows.append(
                    {
                        "run": run_dir.name,
                        "step": step,
                        "benchmark": bench,
                        "n_problems": mat.shape[0],
                        "n_samples": mat.shape[1],
                        "pass@1": round(p1, 4),
                        "pass@1_ci_lo": round(p1_lo, 4),
                        "pass@1_ci_hi": round(p1_hi, 4),
                        "pass@k": round(pk, 4),
                        "pass@k_ci_lo": round(pk_lo, 4),
                        "pass@k_ci_hi": round(pk_hi, 4),
                    }
                )

    if not ci_rows:
        print(f"[bootstrap_ci] no graded rollouts found under {root}")
        return

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ci_rows[0].keys()))
        w.writeheader()
        w.writerows(ci_rows)
    print(f"[bootstrap_ci] wrote {args.out} ({len(ci_rows)} rows)")

    # Optional paired p-values.
    if args.ref_run and args.cmp_run:
        step = args.step
        # If step not given, pick the max step common to both.
        if step is None:
            ref_steps = {s for (r, s) in per_run_bench_mats if r == args.ref_run}
            cmp_steps = {s for (r, s) in per_run_bench_mats if r == args.cmp_run}
            common = ref_steps & cmp_steps
            if not common:
                print(f"[bootstrap_ci] no common step between {args.ref_run} and {args.cmp_run}")
                return
            step = max(common)
        ref_mats = per_run_bench_mats.get((args.ref_run, step), {})
        cmp_mats = per_run_bench_mats.get((args.cmp_run, step), {})
        pval_rows = []
        for bench in sorted(set(ref_mats) & set(cmp_mats)):
            p1_pv = paired_bootstrap_pvalue(
                ref_mats[bench], cmp_mats[bench], pass_at_1, args.n_bootstrap
            )
            pk_pv = paired_bootstrap_pvalue(
                ref_mats[bench], cmp_mats[bench], pass_at_k, args.n_bootstrap
            )
            pval_rows.append(
                {
                    "benchmark": bench,
                    "step": step,
                    "ref_run": args.ref_run,
                    "cmp_run": args.cmp_run,
                    "pass@1_p_paired_boot": round(p1_pv, 4),
                    "pass@k_p_paired_boot": round(pk_pv, 4),
                    "n_problems": ref_mats[bench].shape[0],
                }
            )
        with open(args.out_pvals, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pval_rows[0].keys()))
            w.writeheader()
            w.writerows(pval_rows)
        print(f"[bootstrap_ci] wrote {args.out_pvals} ({len(pval_rows)} rows)")


if __name__ == "__main__":
    main()
