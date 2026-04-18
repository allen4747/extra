#!/usr/bin/env python3
"""
Evaluate pass@k and diversity metrics for ExTra checkpoints on MATH-500.

Handles FSDP checkpoints (any world_size) by merging shards and loading
into a HuggingFace model, then using vLLM for fast generation.

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval_passk_diversity.py \
        --checkpoint /external1/wenyang/checkpoints/ExTra_Research/GRPO-JustRL-Qwen2.5-1.5B/global_step_200/actor \
        --name GRPO-JustRL-step200 \
        --n_samples 16
"""

import argparse
import json
import os
import re
import glob
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


def merge_fsdp_checkpoint(ckpt_dir: str, output_dir: str, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Merge FSDP sharded checkpoint into a HuggingFace model directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Find all model shards
    shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model_world_size_*_rank_*.pt")))
    if not shard_files:
        raise FileNotFoundError(f"No model shards found in {ckpt_dir}")

    world_size = len(shard_files)
    print(f"Found {world_size} FSDP shard(s) in {ckpt_dir}")

    # Merge shards into a single state_dict
    merged_state_dict = {}
    for shard_file in shard_files:
        print(f"  Loading {os.path.basename(shard_file)}...")
        shard = torch.load(shard_file, map_location="cpu", weights_only=False)
        for key, value in shard.items():
            if key in merged_state_dict:
                # FSDP shards the first dimension — concatenate
                merged_state_dict[key] = torch.cat([merged_state_dict[key], value], dim=0)
            else:
                merged_state_dict[key] = value
        del shard

    # Load the base model config and create an empty model
    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # Load the merged weights
    missing, unexpected = model.load_state_dict(merged_state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    del merged_state_dict

    # Save as HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved HF model to {output_dir}")
    del model
    torch.cuda.empty_cache()
    return output_dir


def load_math500(val_file: str):
    """Load MATH-500 problems from parquet."""
    import pandas as pd
    df = pd.read_parquet(val_file)
    problems = []
    for _, row in df.iterrows():
        # verl format: prompt is in 'prompt' column as a chat-template list
        prompt = row.get("prompt", None)
        if prompt is None:
            continue
        # Extract the ground truth answer if available
        gt = row.get("reward_model", {}).get("ground_truth", row.get("answer", None))
        problems.append({"prompt": prompt, "ground_truth": gt, "uid": row.get("extra_info", {}).get("uid", str(len(problems)))})
    return problems


def generate_responses(model_path: str, problems: list, n_samples: int, max_tokens: int = 4096):
    """Generate n_samples responses per problem using vLLM."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading model from {model_path} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=6144,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        n=n_samples,
    )

    # Build prompts using chat template
    formatted_prompts = []
    for p in problems:
        prompt_data = p["prompt"]
        if isinstance(prompt_data, list):
            # Chat format: list of {"role": ..., "content": ...}
            text = tokenizer.apply_chat_template(prompt_data, tokenize=False, add_generation_prompt=True)
        elif isinstance(prompt_data, str):
            text = prompt_data
        else:
            text = str(prompt_data)
        formatted_prompts.append(text)

    print(f"Generating {n_samples} responses for {len(formatted_prompts)} problems...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        responses = [o.text for o in output.outputs]
        results.append({
            "uid": problems[i]["uid"],
            "ground_truth": problems[i]["ground_truth"],
            "responses": responses,
        })

    del llm
    torch.cuda.empty_cache()
    return results


def extract_answer(text: str) -> str:
    """Extract the final boxed answer from a response."""
    # Look for \boxed{...}
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for "answer is X" patterns
    m = re.search(r'(?:answer|result)\s*(?:is|=)\s*[:\s]*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def normalize_answer(ans: str) -> str:
    """Normalize an answer for comparison."""
    ans = ans.strip()
    # Remove surrounding $ signs
    ans = ans.strip("$").strip()
    # Remove \\text{} wrappers
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    # Remove spaces
    ans = ans.replace(" ", "")
    # Try to normalize fractions
    ans = ans.replace("\\frac", "frac")
    return ans.lower()


def check_correct(response: str, ground_truth) -> bool:
    """Check if a response's extracted answer matches the ground truth."""
    if ground_truth is None:
        return False
    pred = normalize_answer(extract_answer(response))
    gt = normalize_answer(str(ground_truth))
    if not pred:
        return False
    return pred == gt


def compute_pass_at_k(results: list, k_values: list = [1, 4, 8, 16]) -> dict:
    """Compute pass@k for each k value."""
    metrics = {}
    for k in k_values:
        pass_at_k = []
        for item in results:
            n = len(item["responses"])
            if n < k:
                continue
            correct = sum(1 for r in item["responses"] if check_correct(r, item["ground_truth"]))
            # pass@k = 1 - C(n-c, k) / C(n, k)
            if correct == 0:
                pass_at_k.append(0.0)
            elif correct >= n:
                pass_at_k.append(1.0)
            else:
                from math import comb
                pass_at_k.append(1.0 - comb(n - correct, k) / comb(n, k) if n >= k else 1.0)
        metrics[f"pass@{k}"] = float(np.mean(pass_at_k)) if pass_at_k else 0.0
        metrics[f"pass@{k}_n"] = len(pass_at_k)
    return metrics


def compute_diversity(results: list) -> dict:
    """Compute diversity metrics: avg pairwise cosine distance and log-det volume."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    cosine_dists = []
    logdet_vols = []
    correct_cosine_dists = []  # diversity among correct responses only

    for item in results:
        responses = item["responses"]
        if len(responses) < 2:
            continue

        embeddings = model.encode(responses, convert_to_tensor=True, normalize_embeddings=True)
        n = embeddings.shape[0]

        # All-response diversity
        sim = (embeddings @ embeddings.T).cpu().numpy()
        mask = ~np.eye(n, dtype=bool)
        cosine_dists.append(float(np.mean(1.0 - sim[mask])))

        # Log-det volume
        Z = embeddings.cpu().numpy()
        gram = Z @ Z.T + 1e-6 * np.eye(n)
        sign, logdet = np.linalg.slogdet(gram)
        if sign > 0:
            logdet_vols.append(float(logdet))

        # Diversity among correct responses only
        correct_idx = [i for i, r in enumerate(responses) if check_correct(r, item["ground_truth"])]
        if len(correct_idx) >= 2:
            correct_emb = embeddings[correct_idx]
            sim_c = (correct_emb @ correct_emb.T).cpu().numpy()
            mask_c = ~np.eye(len(correct_idx), dtype=bool)
            correct_cosine_dists.append(float(np.mean(1.0 - sim_c[mask_c])))

    return {
        "avg_cosine_distance": float(np.mean(cosine_dists)) if cosine_dists else None,
        "std_cosine_distance": float(np.std(cosine_dists)) if cosine_dists else None,
        "avg_logdet_volume": float(np.mean(logdet_vols)) if logdet_vols else None,
        "avg_correct_cosine_distance": float(np.mean(correct_cosine_dists)) if correct_cosine_dists else None,
        "n_problems_with_correct_diversity": len(correct_cosine_dists),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pass@k and diversity for ExTra checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to FSDP actor checkpoint directory")
    parser.add_argument("--name", required=True, help="Name for this evaluation run")
    parser.add_argument("--val_file", default=os.path.expanduser("~/data/math500/test.parquet"))
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="./eval_results")
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{args.name}.json")

    # Step 1: Convert FSDP checkpoint to HuggingFace format
    hf_dir = os.path.join(args.output_dir, f"{args.name}_hf")
    if os.path.exists(os.path.join(hf_dir, "config.json")):
        print(f"HF model already exists at {hf_dir}, skipping conversion.")
        model_path = hf_dir
    else:
        print("=== Step 1: Converting FSDP checkpoint to HuggingFace format ===")
        model_path = merge_fsdp_checkpoint(args.checkpoint, hf_dir, args.base_model)

    # Step 2: Load eval data
    print("\n=== Step 2: Loading MATH-500 ===")
    problems = load_math500(args.val_file)
    print(f"Loaded {len(problems)} problems")

    # Step 3: Generate responses
    print(f"\n=== Step 3: Generating {args.n_samples} responses per problem ===")
    results = generate_responses(model_path, problems, args.n_samples, args.max_tokens)

    # Step 4: Compute pass@k
    print("\n=== Step 4: Computing pass@k ===")
    k_values = [1, 4, 8, 16] if args.n_samples >= 16 else [1, 4, 8]
    k_values = [k for k in k_values if k <= args.n_samples]
    pass_metrics = compute_pass_at_k(results, k_values)
    for k, v in pass_metrics.items():
        if not k.endswith("_n"):
            print(f"  {k}: {v:.4f}")

    # Step 5: Compute diversity
    print("\n=== Step 5: Computing diversity metrics ===")
    div_metrics = compute_diversity(results)
    for k, v in div_metrics.items():
        print(f"  {k}: {v}")

    # Save all results
    all_results = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "n_problems": len(problems),
        **pass_metrics,
        **div_metrics,
    }
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Also save raw responses for later analysis
    raw_file = os.path.join(args.output_dir, f"{args.name}_responses.json")
    with open(raw_file, "w") as f:
        json.dump(results, f)
    print(f"Raw responses saved to {raw_file}")


if __name__ == "__main__":
    main()
