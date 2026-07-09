#!/usr/bin/env python3
"""
decontam_ngram.py — EMNLP rebuttal analysis A4.

13-gram overlap check between MATH-DAPO training prompts and each of the six
evaluation benchmarks. Answers XB9Q Q6 ("was decontamination performed?").

Approach: exact 13-gram matching on lowercased whitespace-normalized problem
statements. Reports (i) fraction of eval problems whose text shares any
13-gram with any training prompt, and (ii) mean overlap fraction per eval
problem. 13-grams are a common contamination-detection choice; short enough
to catch paraphrases, long enough to avoid trivial vocabulary overlap.

Usage:
  python analysis/rebuttal/decontam_ngram.py \
    --train_file $HOME/data/math_dapo/train.parquet \
    --eval_dir  $HOME/my_efs/datasets \
    --n 13 \
    --out decontam_table.csv
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


EVAL_SETS = ["AIME24", "AIME25", "AMC23", "MATH-500", "Minerva", "OlympiadBench"]


def normalize(text: str) -> list[str]:
    text = text.lower()
    # Collapse whitespace, drop most punctuation (keep math tokens simple).
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def load_problem_texts(parquet_path: Path) -> list[str]:
    """Extract the problem statement per row.

    Follows the schema used by evals/gen_vllm.py:load_samples:
      * default (MATH-DAPO, MATH-500, AIME24, AIME25, AMC23, Minerva,
        OlympiadBench): row["prompt"][0]["content"]
      * BRUMO25 / CMIMC25 / HMMT25: row["problem"]  (not used here)
    """
    df = pd.read_parquet(parquet_path)
    out = []
    for i in range(len(df)):
        try:
            out.append(df.at[i, "prompt"][0]["content"])
        except Exception:
            try:
                out.append(df.at[i, "problem"])
            except Exception:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True, help="MATH-DAPO train parquet")
    ap.add_argument("--eval_dir", required=True, help="Root containing {BENCH}/test.parquet")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--out", default="decontam_table.csv")
    args = ap.parse_args()

    print(f"[decontam] loading train {args.train_file}")
    train_texts = load_problem_texts(Path(args.train_file))
    train_ngrams: set[tuple[str, ...]] = set()
    for t in train_texts:
        train_ngrams |= ngrams(normalize(t), args.n)
    print(f"[decontam] train prompts: {len(train_texts)}, unique {args.n}-grams: {len(train_ngrams)}")

    rows = []
    for bench in EVAL_SETS:
        eval_path = Path(args.eval_dir) / bench / "test.parquet"
        if not eval_path.exists():
            print(f"[decontam] skip {bench}: {eval_path} not found")
            continue
        eval_texts = load_problem_texts(eval_path)
        touched = 0
        overlap_fracs = []
        for t in eval_texts:
            eng = ngrams(normalize(t), args.n)
            if not eng:
                overlap_fracs.append(0.0)
                continue
            inter = eng & train_ngrams
            if inter:
                touched += 1
            overlap_fracs.append(len(inter) / max(len(eng), 1))
        n_prob = len(eval_texts)
        rows.append(
            {
                "benchmark": bench,
                "n_problems": n_prob,
                "n_grams": args.n,
                "problems_with_any_overlap": touched,
                "problems_with_any_overlap_frac": round(touched / max(n_prob, 1), 4),
                "mean_overlap_frac_per_problem": round(
                    sum(overlap_fracs) / max(len(overlap_fracs), 1), 4
                ),
            }
        )
        print(f"[decontam] {bench}: {touched}/{n_prob} problems have any {args.n}-gram overlap")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "n_problems",
                "n_grams",
                "problems_with_any_overlap",
                "problems_with_any_overlap_frac",
                "mean_overlap_frac_per_problem",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[decontam] wrote {args.out}")


if __name__ == "__main__":
    main()
