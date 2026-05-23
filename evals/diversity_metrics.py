#!/usr/bin/env python3
"""
Compute trajectory-diversity measures from cached eval JSONL files.

Two complementary metrics, computed per problem then averaged:

1. distinct_4gram
   Fraction of unique 4-grams across the n responses for one problem.
   Range [0, 1]; higher = more diverse vocabulary / phrasing.
   Cheap, CPU-only.

2. logdet_diversity
   log det(I + alpha * K), where K is the n x n cosine-similarity
   kernel of sentence-transformer embeddings of the n responses.
   This is the DPP (determinantal point process) log-volume measure.
   Range [0, n*log(1+alpha)]; higher = more spread in embedding space.
   Needs sentence-transformers (and ~1 minute per checkpoint on CPU,
   ~5 sec on GPU).

Usage:
    # Per-checkpoint diversity report
    python evals/diversity_metrics.py \\
        --eval_dir eval_outputs_v2/02_ExTra_RegenOnly_Qwen3/step_250/ \\
        --benchmarks aime24 aime25

    # Cross-method comparison (writes a CSV)
    python evals/diversity_metrics.py \\
        --eval_dirs \\
            eval_outputs_v2/01b_GRPO_Baseline_3e6_Qwen3/step_250/ \\
            eval_outputs_v2/02_ExTra_RegenOnly_Qwen3/step_250/ \\
            eval_outputs_v2/05_ExTra_Full_OptionB_Qwen3_1e6_1nov_aws/step_250/ \\
        --output diversity_table.csv

    # Skip the embedding-based metric (faster)
    python evals/diversity_metrics.py --eval_dir <dir> --skip_logdet
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


# ----- distinct n-gram -----

def distinct_ngram(responses, n=4):
    """Fraction of unique n-grams across all responses for one problem."""
    distinct = set()
    total = 0
    for resp in responses:
        toks = str(resp).split()
        for i in range(len(toks) - n + 1):
            distinct.add(tuple(toks[i:i + n]))
            total += 1
    return len(distinct) / total if total > 0 else 0.0


# ----- logdet diversity (DPP) -----

def _load_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
    from sentence_transformers import SentenceTransformer
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [logdet] loading {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


def logdet_diversity(responses, embedder, alpha=1.0):
    """
    log det(I + alpha * K) where K_ij = cos(emb_i, emb_j).

    Cosine similarity is in [-1, 1]; we shift to a PSD form via the
    standard "I + alpha * cos_sim" reparameterization, which is the
    common DPP kernel used in diversity literature.
    """
    import numpy as np
    import torch
    if len(responses) < 2:
        return 0.0
    with torch.no_grad():
        embs = embedder.encode(
            list(responses), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
    # Cosine similarity (since vectors are normalized).
    K = embs @ embs.T            # n x n in [-1, 1]
    n = K.shape[0]
    M = np.eye(n) + alpha * K    # PSD for alpha small enough; usually fine
    # Numerical safety: add tiny ridge.
    M = M + 1e-6 * np.eye(n)
    sign, logabsdet = np.linalg.slogdet(M)
    if sign <= 0:
        # Fallback: clip eigenvalues to be safe.
        eigs = np.linalg.eigvalsh(M).clip(min=1e-8)
        return float(np.sum(np.log(eigs)))
    return float(logabsdet)


# ----- per-eval-dir aggregation -----

def load_jsonl(path):
    """Group rows by example_id; return list[dict(gt, responses)]."""
    by_id = defaultdict(lambda: {"gt": None, "responses": []})
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            i = int(d["example_id"])
            by_id[i]["gt"] = d.get("answer")
            by_id[i]["responses"].append(d["response"])
    return [by_id[i] for i in sorted(by_id)]


def diversity_for_jsonl(jsonl_path, embedder=None, n=4, alpha=1.0):
    """Average distinct_4gram and (optionally) logdet across problems."""
    problems = load_jsonl(jsonl_path)
    if not problems:
        return None
    n4_scores = [distinct_ngram(p["responses"], n=n) for p in problems]
    out = {
        "n_problems": len(problems),
        "n_samples_per_problem": len(problems[0]["responses"]),
        "distinct_4gram": sum(n4_scores) / len(n4_scores),
    }
    if embedder is not None:
        ld_scores = [logdet_diversity(p["responses"], embedder, alpha=alpha)
                     for p in problems]
        out["logdet_diversity"] = sum(ld_scores) / len(ld_scores)
    return out


def report(eval_dir, embedder=None, benchmarks=None):
    """Find all <bench>_*.jsonl in eval_dir and compute diversity per bench."""
    eval_dir = Path(eval_dir)
    files = sorted(eval_dir.glob("*.jsonl"))
    if benchmarks:
        wanted = {b.lower() for b in benchmarks}
        files = [f for f in files if f.stem.split("_")[0].lower() in wanted]
    rows = []
    for f in files:
        bench = f.stem.split("_")[0]
        print(f"  [{bench}] processing {f.name} ...")
        d = diversity_for_jsonl(f, embedder=embedder)
        if d is None:
            continue
        d["benchmark"] = bench
        d["eval_dir"] = str(eval_dir)
        rows.append(d)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", help="Single eval dir to process.")
    p.add_argument("--eval_dirs", nargs="+",
                   help="Multiple eval dirs (each becomes one row group in CSV).")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Restrict to these benchmarks (lowercase, e.g. aime24 math-500).")
    p.add_argument("--output", default=None, help="Write CSV to this path.")
    p.add_argument("--skip_logdet", action="store_true",
                   help="Skip the embedding-based logdet metric (faster).")
    p.add_argument("--alpha", type=float, default=1.0, help="DPP kernel scale.")
    p.add_argument("--ngram_n", type=int, default=4, help="n in distinct_n.")
    args = p.parse_args()

    if not args.eval_dir and not args.eval_dirs:
        p.error("specify --eval_dir or --eval_dirs")

    eval_dirs = args.eval_dirs if args.eval_dirs else [args.eval_dir]

    embedder = None if args.skip_logdet else _load_embedder()

    all_rows = []
    for d in eval_dirs:
        print(f"=== {d} ===")
        rows = report(d, embedder=embedder, benchmarks=args.benchmarks)
        all_rows.extend(rows)

    # Pretty print
    print()
    print(f"{'eval_dir':<60} {'benchmark':<14} {'n':>4} {'k':>3}  {'distinct_4gram':>14}", end='')
    if not args.skip_logdet:
        print(f"  {'logdet':>10}", end='')
    print()
    print('-' * 110)
    for r in all_rows:
        edir = Path(r['eval_dir']).parent.name + '/' + Path(r['eval_dir']).name
        line = (f"{edir[:60]:<60} {r['benchmark']:<14} "
                f"{r['n_problems']:>4} {r['n_samples_per_problem']:>3}  "
                f"{r['distinct_4gram']:>14.4f}")
        if 'logdet_diversity' in r:
            line += f"  {r['logdet_diversity']:>10.3f}"
        print(line)

    if args.output:
        import csv
        keys = ["eval_dir", "benchmark", "n_problems", "n_samples_per_problem",
                "distinct_4gram", "logdet_diversity"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in keys})
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
