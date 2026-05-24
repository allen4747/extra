#!/usr/bin/env python3
"""
Resampling pass@k bar-plot experiment for ExTra (Qwen3-1.7B).

For each MATH-500 level-5 problem, compares THREE prefix-selection
strategies against random sampling, at multiple k values:

  Arms:
    1. random          — generate K samples directly from the prompt
    2. raw_entropy     — pick the prefix with the LOWEST mean token
                         entropy among K candidate prefixes; resample
                         continuations from it
    3. smoothed_entropy — pick the prefix with the lowest semantic-
                          smoothed entropy (softmax over cosine sim,
                          temperature tau) -- this is ExTra's
                          regeneration heuristic

  Metrics (per arm): pass@1, pass@8, pass@16
                     where pass@k = 1[any of k samples is correct]

  Output:
    qwen3_outputs/resampling_passk_qwen3.json    — full numerical results
    qwen3_outputs/resampling_passk_qwen3.csv     — flat CSV for paper paste
    qwen3_outputs/resampling_passk_bars.pdf      — grouped bar plot
    qwen3_outputs/resampling_passk_bars.png      — same, raster
    qwen3_outputs/resampling_passk_bars.png.data.json
                                                 — sidecar plot data
                                                   for re-styling

Designed for an 8-GPU host with H200/A100 cards.  vLLM tensor_parallel
spans all 8 GPUs (Qwen3-1.7B has 16 heads so TP=8 divides evenly).
The HF scoring model shares cuda:0 with one vLLM TP slot.

Usage:
    cd analysis/
    pip install matplotlib scipy tqdm sentence-transformers
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        python resampling_passk_qwen3.py
"""

import os
import sys
import json
import argparse
import random
import re
from pathlib import Path

# CUDA_VISIBLE_DEVICES must be set before any cuda init.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Disable vLLM's torch.compile path (avoids "failed to get hash of
# compiled graph" failures on some vLLM/torch combos).
os.environ.setdefault("VLLM_USE_V1", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Robustness helpers (dual_log, atomic JSON, savefig sidecar).
try:
    import _qwen3_robust as robust  # noqa: F401
    HAVE_ROBUST = True
except Exception:
    HAVE_ROBUST = False

OUT_DIR = HERE / "qwen3_outputs"
OUT_DIR.mkdir(exist_ok=True)

# Install tiny shims for `parse_prediction` / `process_thoughts` if the
# matching modules aren't installed (matches the other Qwen3 wrappers).
try:
    import _qwen3_shims  # noqa: F401
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from datasets import load_dataset  # noqa: E402


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--n_problems", type=int, default=50,
                        help="Number of MATH-500 level-5 problems to use.")
    parser.add_argument("--n_per_arm", type=int, default=16,
                        help="Number of continuations per arm.  pass@k is "
                             "computed for k in {1,8,n_per_arm}.")
    parser.add_argument("--n_prefix_candidates", type=int, default=16,
                        help="Number of prefix candidates from which to pick "
                             "the best (used by raw and smoothed arms).")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Semantic smoothing temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_prefix", default="resampling_passk_qwen3")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip plotting; only write JSON / CSV.")
    args = parser.parse_args()

    # ---- Logging ----
    if HAVE_ROBUST:
        robust.dual_log(str(OUT_DIR))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ---- Pick TP size that divides Qwen3's 16 heads ----
    n_gpus = max(1, torch.cuda.device_count())
    valid_tps = [tp for tp in (n_gpus, 8, 4, 2, 1) if tp <= n_gpus and 16 % tp == 0]
    tp_size = valid_tps[0] if valid_tps else 1

    print(f"[config] model={args.model}")
    print(f"[config] n_gpus={n_gpus}, tensor_parallel_size={tp_size}")
    print(f"[config] n_problems={args.n_problems}, "
          f"n_per_arm={args.n_per_arm}, "
          f"n_prefix_candidates={args.n_prefix_candidates}, "
          f"tau={args.tau}")

    # ---- Load models ----
    print("[load] HF scoring model on cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    scoring_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0",
    )
    scoring_model.eval()

    print(f"[load] vLLM with TP={tp_size}")
    gen_llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.6,  # leave headroom for HF scoring model
    )

    # ---- Load problems ----
    print(f"[data] loading MATH-500 level-5, taking {args.n_problems} problems")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    pool = [{"problem": x["problem"], "answer": x["answer"], "level": x.get("level")}
            for x in ds if x.get("level") in [5]]
    random.shuffle(pool)
    problems = pool[:args.n_problems]
    print(f"[data] using {len(problems)} problems")

    # ---- Run experiment ----
    K_VALUES = sorted(set([1, min(8, args.n_per_arm), args.n_per_arm]))
    print(f"[config] reporting pass@k for k in {K_VALUES}")

    arm_hits = {arm: {k: [] for k in K_VALUES}
                for arm in ("random", "raw_entropy", "smoothed_entropy")}
    arm_failures = {arm: 0 for arm in arm_hits}
    per_problem = []

    for idx, p in enumerate(tqdm(problems, desc="problems")):
        prompt = p["problem"]
        gt = p["answer"]
        try:
            results = run_one_problem(
                gen_llm=gen_llm,
                scoring_model=scoring_model,
                tokenizer=tokenizer,
                prompt=prompt,
                gt=gt,
                n_per_arm=args.n_per_arm,
                n_prefix_candidates=args.n_prefix_candidates,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tau=args.tau,
            )
        except Exception as e:
            print(f"[problem {idx}] failed: {e}")
            for arm in arm_failures:
                arm_failures[arm] += 1
            continue

        # results[arm] = list[bool] of length n_per_arm
        problem_record = {"problem_idx": idx, "gt": gt}
        for arm, hits in results.items():
            problem_record[f"{arm}_hits"] = hits
            for k in K_VALUES:
                arm_hits[arm][k].append(int(any(hits[:k])))
        per_problem.append(problem_record)

    # ---- Aggregate ----
    summary = {
        "config": {
            "model": args.model,
            "n_problems_attempted": len(problems),
            "n_problems_succeeded": len(per_problem),
            "n_per_arm": args.n_per_arm,
            "n_prefix_candidates": args.n_prefix_candidates,
            "tau": args.tau,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "tp_size": tp_size,
            "k_values": K_VALUES,
            "arms": list(arm_hits.keys()),
        },
        "arm_failures_per_problem": arm_failures,
        "pass_rates": {
            arm: {f"pass@{k}": (float(np.mean(v)) if v else None)
                  for k, v in by_k.items()}
            for arm, by_k in arm_hits.items()
        },
        "n_used_per_arm": {
            arm: {f"pass@{k}": len(v) for k, v in by_k.items()}
            for arm, by_k in arm_hits.items()
        },
    }

    # ---- Save numerical results FIRST (before any plot) ----
    json_path = OUT_DIR / f"{args.output_prefix}.json"
    if HAVE_ROBUST:
        robust.safe_save_json(summary, str(json_path))
    else:
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[save] {json_path}")

    # Flat CSV for paper-paste convenience.
    csv_path = OUT_DIR / f"{args.output_prefix}.csv"
    with open(csv_path, "w") as f:
        f.write("arm," + ",".join(f"pass@{k}" for k in K_VALUES) + "\n")
        for arm in ("random", "raw_entropy", "smoothed_entropy"):
            row = [arm] + [f"{summary['pass_rates'][arm][f'pass@{k}']:.4f}"
                            if summary['pass_rates'][arm][f'pass@{k}'] is not None
                            else ""
                            for k in K_VALUES]
            f.write(",".join(row) + "\n")
    print(f"[save] {csv_path}")

    # Per-problem details for re-analysis.
    detail_path = OUT_DIR / f"{args.output_prefix}_per_problem.json"
    with open(detail_path, "w") as f:
        json.dump(per_problem, f, indent=2)
    print(f"[save] {detail_path}")

    # ---- Print summary table ----
    print()
    print("=" * 70)
    header = f"{'arm':<22}" + "".join(f"{f'pass@{k}':>10}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for arm in ("random", "raw_entropy", "smoothed_entropy"):
        line = f"{arm:<22}"
        for k in K_VALUES:
            v = summary["pass_rates"][arm][f"pass@{k}"]
            line += f"{v:>10.4f}" if v is not None else f"{'--':>10}"
        print(line)
    print()

    # ---- Bar plot ----
    if not args.no_plot:
        try:
            plot_grouped_bars(summary, args.output_prefix)
        except Exception as e:
            print(f"[plot] failed: {e}")

    print("\n[done] resampling pass@k experiment finished.")


# ----------------------------------------------------------------------
# Per-problem logic
# ----------------------------------------------------------------------

def _format_prompt(prompt, tokenizer):
    """Format the math problem with Qwen chat template + boxed-answer hint."""
    messages = [
        {"role": "system",
         "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _process_steps(text):
    """Newline-based step splitter, kept simple to avoid third-party deps."""
    return [line.strip() for line in text.split("\n") if line.strip()]


@torch.no_grad()
def _entropy_and_embedding(scoring_model, tokenizer, prompt_text, prefix_text):
    """Return (mean_token_entropy, sentence_embedding) for the prefix.

    - mean_token_entropy: mean of per-token Shannon entropy of the
      conditional distribution over the prefix tokens (computed with
      the HF scoring model).
    - sentence_embedding: last hidden state of the last prefix token
      (used for the semantic-similarity smoothing).
    """
    full = prompt_text + prefix_text
    full_ids = tokenizer(full, return_tensors="pt", truncation=True,
                         max_length=2048).input_ids.to(scoring_model.device)
    ctx_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                        max_length=2048).input_ids
    ctx_len = ctx_ids.shape[1]
    if ctx_len >= full_ids.shape[1]:
        # Empty prefix; return high entropy so it's not selected.
        d = scoring_model.config.hidden_size
        return float("inf"), torch.zeros(d, device=scoring_model.device)

    out = scoring_model(full_ids, output_hidden_states=True)
    # Token entropy over the prefix slice.
    logits = out.logits[0, ctx_len - 1: -1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    mean_ent = float(token_entropy.mean().item())

    # Sentence embedding = last hidden state at the last prefix token.
    last_hidden = out.hidden_states[-1][0, -1, :]
    return mean_ent, last_hidden.detach()


@torch.no_grad()
def run_one_problem(*, gen_llm, scoring_model, tokenizer, prompt, gt,
                    n_per_arm, n_prefix_candidates,
                    max_new_tokens, temperature, top_p, tau):
    """Return {arm_name: list[bool] of length n_per_arm}."""
    formatted = _format_prompt(prompt, tokenizer)

    # ---- (a) Random arm: K direct samples from the prompt ----
    sp = SamplingParams(temperature=temperature, top_p=top_p,
                        max_tokens=max_new_tokens, n=n_per_arm)
    out = gen_llm.generate([formatted], sp, use_tqdm=False)[0]
    random_responses = [comp.text for comp in out.outputs]

    # ---- Generate prefix-candidate trajectories ----
    sp_cand = SamplingParams(temperature=temperature, top_p=top_p,
                             max_tokens=max_new_tokens,
                             n=n_prefix_candidates)
    cand_out = gen_llm.generate([formatted], sp_cand, use_tqdm=False)[0]
    cand_responses = [comp.text for comp in cand_out.outputs]

    # For each candidate response, take its prefix (all but last step).
    # Build (full_prefix_string, prefix_text_only, mean_entropy, embedding).
    prefix_records = []
    for resp in cand_responses:
        steps = _process_steps(resp)
        if len(steps) < 2:
            continue
        # Use all-but-last step as the prefix.
        prefix_steps = steps[:-1]
        prefix_text = "\n".join(prefix_steps) + "\n"
        full_prefix = formatted + prefix_text
        try:
            mean_ent, emb = _entropy_and_embedding(
                scoring_model, tokenizer, formatted, prefix_text)
        except Exception:
            continue
        if not np.isfinite(mean_ent):
            continue
        prefix_records.append({
            "full_prefix": full_prefix,
            "prefix_text": prefix_text,
            "mean_entropy": mean_ent,
            "embedding": emb,
        })

    if not prefix_records:
        # No usable prefixes; fall back to random for the other two arms.
        return {
            "random": [_score(r, gt) for r in random_responses],
            "raw_entropy": [_score(r, gt) for r in random_responses],
            "smoothed_entropy": [_score(r, gt) for r in random_responses],
        }

    # ---- (b) Raw-entropy arm: pick the prefix with min mean entropy ----
    raw_idx = int(np.argmin([r["mean_entropy"] for r in prefix_records]))
    raw_best = prefix_records[raw_idx]["full_prefix"]

    # ---- (c) Smoothed-entropy arm: softmax over cosine similarity ----
    embs = torch.stack([r["embedding"] for r in prefix_records])  # [N, D]
    embs = F.normalize(embs.float(), p=2, dim=1)
    sim = embs @ embs.t()                            # [N, N] in [-1, 1]
    raw_scores = torch.tensor(
        [r["mean_entropy"] for r in prefix_records],
        dtype=torch.float32, device=embs.device,
    )
    weights = F.softmax(sim / max(tau, 1e-6), dim=1)  # [N, N]
    smoothed = (weights @ raw_scores).cpu().numpy()
    sm_idx = int(np.argmin(smoothed))
    sm_best = prefix_records[sm_idx]["full_prefix"]

    # ---- Generate K samples from each chosen prefix ----
    sp_cont = SamplingParams(temperature=temperature, top_p=top_p,
                             max_tokens=max_new_tokens, n=n_per_arm)
    raw_out = gen_llm.generate([raw_best], sp_cont, use_tqdm=False)[0]
    raw_responses = [comp.text for comp in raw_out.outputs]

    sm_out = gen_llm.generate([sm_best], sp_cont, use_tqdm=False)[0]
    sm_responses = [comp.text for comp in sm_out.outputs]

    return {
        "random": [_score(r, gt) for r in random_responses],
        "raw_entropy": [_score(r, gt) for r in raw_responses],
        "smoothed_entropy": [_score(r, gt) for r in sm_responses],
    }


# ----------------------------------------------------------------------
# Scoring (lightweight rule-based)
# ----------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def _score(response, gt):
    """Rule-based correctness check (kept simple and fast)."""
    if not isinstance(response, str):
        return False
    pred = ""
    matches = _BOXED_RE.findall(response)
    if matches:
        pred = matches[-1].strip()
    else:
        # Fallback: search for the GT string directly in the response.
        return _normalize(gt) in _normalize(response)
    return _normalize(pred) == _normalize(gt) or _normalize(gt) in _normalize(response)


def _normalize(s):
    return str(s).strip().lower().replace(" ", "")


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def plot_grouped_bars(summary, output_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if HAVE_ROBUST:
        try:
            robust.patch_pyplot_savefig(str(OUT_DIR))
        except Exception:
            pass

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "savefig.dpi": 300,
        "figure.dpi": 300,
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    arms = ("random", "raw_entropy", "smoothed_entropy")
    arm_labels = ("Random", "Raw Entropy", "Smoothed Entropy")
    arm_colors = ("#9aa0a6", "#E07A5F", "#2A9D8F")

    K_VALUES = summary["config"]["k_values"]
    n_arms = len(arms)
    bar_width = 0.25
    x = np.arange(len(K_VALUES))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, (arm, color, label) in enumerate(zip(arms, arm_colors, arm_labels)):
        ys = [summary["pass_rates"][arm][f"pass@{k}"] for k in K_VALUES]
        bars = ax.bar(x + (i - 1) * bar_width, ys, width=bar_width,
                      color=color, label=label, edgecolor="white", linewidth=0.6)
        for b, y in zip(bars, ys):
            if y is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2, y + 0.005,
                    f"{y:.2f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([f"pass@{k}" for k in K_VALUES])
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Resampling pass@k on MATH-500 (level 5, n={summary['config']['n_problems_succeeded']})")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()

    pdf_path = OUT_DIR / f"{output_prefix}_bars.pdf"
    png_path = OUT_DIR / f"{output_prefix}_bars.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"[plot] {pdf_path}")
    print(f"[plot] {png_path}")

    # Also dump the bar-plot's own data for easy re-styling.
    plot_data = {
        "k_values": K_VALUES,
        "arms": list(arms),
        "values": {arm: [summary["pass_rates"][arm][f"pass@{k}"] for k in K_VALUES]
                   for arm in arms},
        "colors": list(arm_colors),
        "labels": list(arm_labels),
        "n_problems": summary["config"]["n_problems_succeeded"],
    }
    sidecar = OUT_DIR / f"{output_prefix}_bars.plotdata.json"
    with open(sidecar, "w") as f:
        json.dump(plot_data, f, indent=2)
    print(f"[plot] {sidecar}")


if __name__ == "__main__":
    main()
