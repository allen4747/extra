"""
Prefix Value Estimation for Reasoning Trajectories

Goal: Discover metrics that correlate with the "value" of a partial trajectory,
where value = P(reaching correct answer | prefix).

Approach:
1. Generate full trajectories, split into steps
2. For each prefix (after each step), sample K continuations
3. Estimate pass rate from each prefix
4. Compute various metrics at each prefix
5. Correlate metrics with estimated pass rate
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import re
from scipy.stats import spearmanr, pearsonr
import torch.nn.functional as F
import random
import pickle
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Try to import vLLM for fast generation
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available, falling back to HuggingFace generate (much slower).")

from simplified_evaluator.eval import parse_prediction

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from openrlhf.trainer.ppo_utils.score import process_thoughts


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    n_problems: int = 30           # Number of problems to use
    n_initial_samples: int = 32     # Initial trajectories per problem
    n_continuations: int = 16       # Continuations per prefix for value estimation
    max_new_tokens_gen: int = 1024  # For initial generation
    max_new_tokens_cont: int = 512  # For continuations (shorter, since prefix exists)
    temperature: float = 0.7
    top_p: float = 0.9
    target_levels: list = field(default_factory=lambda: [3, 4])
    seed: int = 42
    # Which prefixes to evaluate (fraction of steps completed)
    # e.g., [0.25, 0.5, 0.75] means after 25%, 50%, 75% of steps
    prefix_fractions: list = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    save_dir: str = os.path.dirname(os.path.abspath(__file__))
    # vLLM settings
    use_vllm: bool = VLLM_AVAILABLE  # Auto-detect; set False to force HF
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9


# ============================================================
# Utility Functions
# ============================================================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name):
    print(f"Loading HF model {model_name} (for metric computation)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_vllm_model(model_name, config):
    """Load model via vLLM for fast batched generation."""
    print(f"Loading vLLM model {model_name} (for fast generation)...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        dtype="half",
        max_model_len=2048,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        seed=config.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return llm, tokenizer


def load_math_problems(target_levels, n=None):
    print(f"Loading problems with levels {target_levels} from HuggingFaceH4/MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = []
    for item in dataset:
        lvl = item.get("level", -1)
        if lvl in target_levels:
            problems.append({
                "problem": item["problem"],
                "answer": item["answer"],
                "level": lvl,
            })
    print(f"Loaded {len(problems)} problems.")
    if n is not None:
        random.shuffle(problems)
        problems = problems[:n]
    return problems


def format_prompt(tokenizer, problem_text):
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def check_correctness(response, gt):
    pred_ans = parse_prediction(response, gt, 'math')
    def clean(s):
        return s.strip().lower().replace(' ', '')
    return clean(pred_ans) == clean(gt) or clean(gt) in clean(response)


# ============================================================
# Step 1: Generate initial trajectories and find valid problems
# ============================================================
def generate_trajectories_hf(model, tokenizer, problem_text, n_samples, config):
    """Generate n_samples full reasoning trajectories using HuggingFace (fallback)."""
    formatted_prompt = format_prompt(tokenizer, problem_text)
    
    all_responses = []
    batch_size = min(n_samples, 8)
    
    for batch_start in range(0, n_samples, batch_size):
        curr_batch = min(batch_size, n_samples - batch_start)
        inputs = tokenizer([formatted_prompt] * curr_batch, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens_gen,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for i in range(curr_batch):
            response = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            all_responses.append(response)
        
        del outputs
        torch.cuda.empty_cache()
    
    return all_responses


def generate_trajectories_vllm(llm, tokenizer, problem_text, n_samples, config):
    """Generate n_samples full reasoning trajectories using vLLM (fast)."""
    formatted_prompt = format_prompt(tokenizer, problem_text)
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens_gen,
    )
    outputs = llm.generate([formatted_prompt], sampling_params=sampling_params)
    return [o.text for o in outputs[0].outputs]


def collect_valid_problems_vllm(llm, tokenizer, problems, config):
    """Collect problems where we have both correct and incorrect trajectories (vLLM)."""
    collected = []
    
    for p_idx, problem_data in enumerate(problems):
        if len(collected) >= config.n_problems:
            break
        
        prompt = problem_data["problem"]
        gt = problem_data["answer"]
        
        print(f"\nChecking Problem {p_idx+1}/{len(problems)}. Found: {len(collected)}/{config.n_problems}")
        
        try:
            responses = generate_trajectories_vllm(llm, tokenizer, prompt, config.n_initial_samples, config)
        except Exception as e:
            print(f"  Skipping due to error: {e}")
            continue
        
        correctness = [check_correctness(r, gt) for r in responses]
        n_correct = sum(correctness)
        n_incorrect = len(correctness) - n_correct
        
        if n_correct > 0 and n_incorrect > 0:
            print(f"  -> Valid! C:{n_correct}, I:{n_incorrect}")
            
            parsed_responses = []
            for r, c in zip(responses, correctness):
                steps = process_thoughts(r)
                if steps and isinstance(steps, list) and len(steps) >= 2:
                    parsed_responses.append({
                        "full_response": r,
                        "steps": steps,
                        "is_correct": c,
                        "extracted_answer": parse_prediction(r, gt, 'math'),
                    })
            
            if len([p for p in parsed_responses if p["is_correct"]]) > 0 and \
               len([p for p in parsed_responses if not p["is_correct"]]) > 0:
                collected.append({
                    "id": p_idx,
                    "problem": prompt,
                    "gt": gt,
                    "level": problem_data.get("level", -1),
                    "parsed_responses": parsed_responses,
                })
        else:
            print(f"  -> Invalid (C:{n_correct}, I:{n_incorrect})")
    
    return collected


def collect_valid_problems_hf(model, tokenizer, problems, config):
    """Collect problems where we have both correct and incorrect trajectories (HF fallback)."""
    collected = []
    
    for p_idx, problem_data in enumerate(problems):
        if len(collected) >= config.n_problems:
            break
        
        prompt = problem_data["problem"]
        gt = problem_data["answer"]
        
        print(f"\nChecking Problem {p_idx+1}/{len(problems)}. Found: {len(collected)}/{config.n_problems}")
        
        try:
            responses = generate_trajectories_hf(model, tokenizer, prompt, config.n_initial_samples, config)
        except RuntimeError as e:
            print(f"  Skipping due to error: {e}")
            torch.cuda.empty_cache()
            continue
        
        correctness = [check_correctness(r, gt) for r in responses]
        n_correct = sum(correctness)
        n_incorrect = len(correctness) - n_correct
        
        if n_correct > 0 and n_incorrect > 0:
            print(f"  -> Valid! C:{n_correct}, I:{n_incorrect}")
            
            # Parse steps for each response
            parsed_responses = []
            for r, c in zip(responses, correctness):
                steps = process_thoughts(r)
                if steps and len(steps) >= 2:  # Need at least 2 steps for meaningful prefixes
                    parsed_responses.append({
                        "full_response": r,
                        "steps": steps,
                        "is_correct": c,
                        "extracted_answer": parse_prediction(r, gt, 'math'),
                    })
            
            if len([p for p in parsed_responses if p["is_correct"]]) > 0 and \
               len([p for p in parsed_responses if not p["is_correct"]]) > 0:
                collected.append({
                    "id": p_idx,
                    "problem": prompt,
                    "gt": gt,
                    "level": problem_data.get("level", -1),
                    "parsed_responses": parsed_responses,
                })
        else:
            print(f"  -> Invalid (C:{n_correct}, I:{n_incorrect})")
    
    return collected


# ============================================================
# Step 2: Estimate prefix values via Monte Carlo sampling
# ============================================================
def continue_from_prefix_hf(model, tokenizer, formatted_prompt, prefix_text, n_continuations, config):
    """Generate multiple continuations from a given prefix (HF fallback)."""
    full_context = formatted_prompt + prefix_text
    inputs = tokenizer(full_context, return_tensors="pt").to(model.device)
    
    continuations = []
    batch_size = min(n_continuations, 8)
    
    for batch_start in range(0, n_continuations, batch_size):
        curr_batch = min(batch_size, n_continuations - batch_start)
        batched_inputs = {
            k: v.repeat(curr_batch, 1) for k, v in inputs.items()
        }
        
        with torch.no_grad():
            outputs = model.generate(
                **batched_inputs,
                max_new_tokens=config.max_new_tokens_cont,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        for i in range(curr_batch):
            cont = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            continuations.append(cont)
        
        del outputs
        torch.cuda.empty_cache()
    
    return continuations


def _build_prefix_requests(collected_data, tokenizer, config):
    """
    Build all prefix data point metadata and their corresponding prompts
    for batched continuation. Shared by both vLLM and HF paths.
    
    Returns:
        all_requests: list of dicts with metadata for each prefix data point
        prompts_for_generation: list of (formatted_prompt + prefix_text) strings
    """
    all_requests = []
    prompts_for_generation = []
    
    for item in collected_data:
        prompt = item["problem"]
        gt = item["gt"]
        formatted_prompt = format_prompt(tokenizer, prompt)
        
        correct_responses = [p for p in item["parsed_responses"] if p["is_correct"]]
        incorrect_responses = [p for p in item["parsed_responses"] if not p["is_correct"]]
        
        n_each = min(len(correct_responses), len(incorrect_responses), 4)
        random.shuffle(correct_responses)
        random.shuffle(incorrect_responses)
        selected = correct_responses[:n_each] + incorrect_responses[:n_each]
        
        for resp_data in selected:
            steps = resp_data["steps"]
            n_steps = len(steps)
            
            for frac in config.prefix_fractions:
                n_prefix_steps = max(1, int(frac * n_steps))
                if n_prefix_steps >= n_steps:
                    continue
                
                prefix_steps = steps[:n_prefix_steps]
                prefix_text = "\n".join(prefix_steps) + "\n"
                
                all_requests.append({
                    "problem_id": item["id"],
                    "problem": prompt,
                    "gt": gt,
                    "prefix_text": prefix_text,
                    "prefix_steps": prefix_steps,
                    "remaining_steps": steps[n_prefix_steps:],
                    "prefix_fraction": frac,
                    "n_prefix_steps": n_prefix_steps,
                    "n_total_steps": n_steps,
                    "full_trajectory_correct": resp_data["is_correct"],
                    "full_response": resp_data["full_response"],
                    "formatted_prompt": formatted_prompt,
                })
                prompts_for_generation.append(formatted_prompt + prefix_text)
    
    return all_requests, prompts_for_generation


def estimate_prefix_values_vllm(llm, tokenizer, collected_data, config):
    """
    Batched version: collect ALL prefix continuation requests first,
    then run them all through vLLM in one call for maximum throughput.
    """
    all_requests, prompts = _build_prefix_requests(collected_data, tokenizer, config)
    
    total_gens = len(prompts) * config.n_continuations
    print(f"Total continuation requests: {len(prompts)} prefixes "
          f"× {config.n_continuations} continuations = {total_gens} generations")
    
    # Single batched call — vLLM handles scheduling internally via SamplingParams(n=K)
    sampling_params = SamplingParams(
        n=config.n_continuations,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_new_tokens_cont,
    )
    
    all_outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    # Compute pass rates
    for req, output in zip(all_requests, all_outputs):
        gt = req["gt"]
        prefix_text = req["prefix_text"]
        continuations = [o.text for o in output.outputs]
        n_pass = sum(check_correctness(prefix_text + c, gt) for c in continuations)
        req["pass_rate"] = n_pass / len(continuations)
    
    return all_requests


def estimate_prefix_values_hf(model, tokenizer, collected_data, config):
    """
    HF fallback: iterate over each prefix and generate continuations sequentially.
    """
    all_requests, prompts = _build_prefix_requests(collected_data, tokenizer, config)
    
    print(f"Total continuation requests: {len(prompts)} prefixes "
          f"× {config.n_continuations} continuations (HF sequential)")
    
    for req, _ in tqdm(
        zip(all_requests, prompts), total=len(all_requests), desc="Estimating prefix values"
    ):
        gt = req["gt"]
        prefix_text = req["prefix_text"]
        formatted_prompt = req["formatted_prompt"]
        
        try:
            continuations = continue_from_prefix_hf(
                model, tokenizer, formatted_prompt, prefix_text,
                config.n_continuations, config
            )
        except RuntimeError as e:
            print(f"  Error during continuation: {e}")
            torch.cuda.empty_cache()
            req["pass_rate"] = 0.0
            continue
        
        n_pass = sum(check_correctness(prefix_text + c, gt) for c in continuations)
        req["pass_rate"] = n_pass / len(continuations)
    
    return all_requests


# ============================================================
# Step 3: Compute various metrics at each prefix
# ============================================================
def calculate_conditional_perplexity(model, tokenizer, context, completion):
    """Perplexity of completion conditioned on context."""
    # Tokenize separately then concatenate to get exact boundary
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    target_ids = input_ids.clone()
    target_ids[:, :context_len] = -100
    
    n_target_tokens = (target_ids != -100).sum().item()
    if n_target_tokens == 0:
        return float('inf')
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()


def calculate_semantic_similarity(model, tokenizer, context, completion):
    """Cosine similarity between hidden state at end of context and end of completion."""
    full_text = context + completion
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    context_inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
    context_len = context_inputs.input_ids.shape[1]
    full_len = inputs.input_ids.shape[1]
    
    if context_len >= full_len or context_len == 0:
        return 0.0
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        
        pre_state = last_hidden[0, context_len - 1, :]
        post_state = last_hidden[0, -1, :]
        
        similarity = F.cosine_similarity(pre_state, post_state, dim=0).item()
    
    return similarity


def calculate_token_entropy_stats(model, tokenizer, context, completion):
    """
    Token-level entropy statistics for the completion tokens.
    Returns: mean_entropy, max_entropy, entropy_std
    """
    full_text = context + completion
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    context_inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
    context_len = context_inputs.input_ids.shape[1]
    
    if context_len >= inputs.input_ids.shape[1]:
        return 0.0, 0.0, 0.0
    
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        # Logits for completion tokens
        logits = outputs.logits[0, context_len-1:-1, :]  # predict completion tokens
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return (
        token_entropy.mean().item(),
        token_entropy.max().item(),
        token_entropy.std().item() if len(token_entropy) > 1 else 0.0,
    )


def calculate_logit_lens_answer_prob(model, tokenizer, context, answer_token_str):
    """
    Project the last-token hidden state through the LM head.
    Return the probability assigned to the first token of the answer.
    """
    answer_tokens = tokenizer.encode(answer_token_str, add_special_tokens=False)
    if not answer_tokens:
        return 0.0
    target_token_id = answer_tokens[0]
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        answer_prob = probs[target_token_id].item()
    
    return answer_prob


# ============================================================
# New GT-aware metrics (scale-robust, information-theoretic)
# ============================================================

def calculate_gt_token_logprob_stats(model, tokenizer, context, gt_answer):
    """
    Compute log-probability statistics for ALL GT answer tokens autoregressively.
    Instead of just the first token, this measures how well the prefix
    predicts the entire GT answer sequence.
    
    Returns: sum_logprob, mean_logprob, min_logprob (weakest token)
    """
    gt_completion = "\nThe answer is " + gt_answer
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(gt_completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0, 0.0, 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]  # predictions for completion tokens
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log-probs of actual completion tokens
        target_ids = input_ids[0, context_len:]
        n_tokens = min(len(target_ids), log_probs.shape[0])
        token_log_probs = log_probs[torch.arange(n_tokens), target_ids[:n_tokens]]
    
    sum_lp = token_log_probs.sum().item()
    mean_lp = token_log_probs.mean().item()
    min_lp = token_log_probs.min().item()
    
    return sum_lp, mean_lp, min_lp


def calculate_gt_token_rank_stats(model, tokenizer, context, gt_answer):
    """
    For each GT answer token, compute its rank in the vocabulary distribution.
    Rank is much more robust to model scale than raw probability.
    
    Returns: mean_rank, max_rank (worst), fraction_in_top10
    """
    gt_completion = "\nThe answer is " + gt_answer
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(gt_completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0, 0.0, 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]
        
        target_ids = input_ids[0, context_len:]
        n_tokens = min(len(target_ids), logits.shape[0])
        
        # Compute ranks: sort descending, find position of target token
        ranks = []
        in_top_10 = 0
        for i in range(n_tokens):
            sorted_indices = torch.argsort(logits[i], descending=True)
            rank = (sorted_indices == target_ids[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            if rank <= 10:
                in_top_10 += 1
    
    ranks_arr = np.array(ranks)
    mean_rank = float(np.mean(ranks_arr))
    max_rank = float(np.max(ranks_arr))
    frac_top10 = in_top_10 / n_tokens if n_tokens > 0 else 0.0
    
    return mean_rank, max_rank, frac_top10


def calculate_mutual_information_prefix_gt(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    I(prefix; GT) = H(GT|prompt) - H(GT|prompt+prefix)
    
    Measures how much the prefix reduces uncertainty about the GT answer.
    Positive = prefix is informative about GT, Negative = prefix confuses.
    More robust than raw ppl because it's a *difference* in entropy.
    """
    gt_completion = "\nThe answer is " + gt_answer
    
    # H(GT | prompt) — entropy of GT tokens given only the prompt
    h_unconditional = _compute_conditional_token_entropy_mean(
        model, tokenizer, formatted_prompt, gt_completion
    )
    
    # H(GT | prompt + prefix) — entropy of GT tokens given prompt + prefix
    h_conditional = _compute_conditional_token_entropy_mean(
        model, tokenizer, formatted_prompt + prefix_text, gt_completion
    )
    
    # MI = reduction in entropy
    return h_unconditional - h_conditional, h_unconditional, h_conditional


def _compute_conditional_token_entropy_mean(model, tokenizer, context, completion):
    """
    Mean entropy over the vocabulary at each completion token position,
    conditioned on context. Lower entropy = model is more focused/certain.
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return token_entropy.mean().item()


def calculate_normalized_gt_logprob(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    Normalized GT log-probability:
        log P(GT | prompt + prefix) / log P(GT | prompt)
    
    Scale-invariant: values > 1 mean the prefix helps predict GT,
    < 1 means the prefix hurts. Normalizing by the unconditional 
    log-prob removes model-scale dependence.
    """
    gt_completion = "\nThe answer is " + gt_answer
    
    # Conditional log-prob (with prefix)
    cond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt + prefix_text, gt_completion
    )
    
    # Unconditional log-prob (without prefix)
    uncond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt, gt_completion
    )
    
    # Both are negative; ratio > 1 means prefix helps
    if uncond_lp == 0 or not np.isfinite(uncond_lp):
        return 1.0, cond_lp, uncond_lp
    
    ratio = cond_lp / uncond_lp  # Both negative, so ratio > 1 if |cond| < |uncond|
    return ratio, cond_lp, uncond_lp


def _compute_completion_logprob(model, tokenizer, context, completion):
    """Sum of log-probabilities of completion tokens given context."""
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = input_ids[0, context_len:]
        n_tokens = min(len(target_ids), log_probs.shape[0])
        token_log_probs = log_probs[torch.arange(n_tokens), target_ids[:n_tokens]]
    
    return token_log_probs.sum().item()


def calculate_pointwise_mutual_information(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    PMI(prefix, GT) = log P(GT|prefix) - log P(GT)
    
    Classic information-theoretic measure of association.
    Positive PMI = prefix is informative about GT.
    This is the log-domain version of the Bayes factor.
    """
    gt_completion = "\nThe answer is " + gt_answer
    
    cond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt + prefix_text, gt_completion
    )
    uncond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt, gt_completion
    )
    
    pmi = cond_lp - uncond_lp  # Positive if prefix makes GT more likely
    return pmi


def calculate_gt_answer_prob_proper(model, tokenizer, context, gt_answer):
    """
    Proper probability of the GT answer: exp(mean log-prob of GT tokens).
    This is the geometric mean of per-token probabilities, 
    which is length-normalized and more interpretable than raw sum.
    """
    gt_completion = "\nThe answer is " + gt_answer
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(gt_completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = input_ids[0, context_len:]
        n_tokens = min(len(target_ids), log_probs.shape[0])
        token_log_probs = log_probs[torch.arange(n_tokens), target_ids[:n_tokens]]
    
    # Geometric mean probability = exp(mean log-prob)
    return torch.exp(token_log_probs.mean()).item()


def calculate_prefix_gt_alignment(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    Cosine similarity between the hidden representation at the end of the prefix 
    and the hidden representation at the end of the GT completion (from prompt only).
    
    Measures whether the prefix is steering the model's internal state toward 
    the "correct answer" representation.
    """
    # Get hidden state at end of prefix
    prefix_context = formatted_prompt + prefix_text
    prefix_inputs = tokenizer(prefix_context, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    # Get hidden state at end of GT completion (from prompt, no prefix — the "target" state)
    gt_completion = "\nThe answer is " + gt_answer
    gt_context = formatted_prompt + gt_completion
    gt_inputs = tokenizer(gt_context, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        prefix_outputs = model(**prefix_inputs, output_hidden_states=True)
        gt_outputs = model(**gt_inputs, output_hidden_states=True)
        
        # Last-layer hidden state at last token position
        prefix_state = prefix_outputs.hidden_states[-1][0, -1, :]
        gt_state = gt_outputs.hidden_states[-1][0, -1, :]
        
        alignment = F.cosine_similarity(prefix_state, gt_state, dim=0).item()
    
    return alignment


def calculate_gt_entropy_reduction(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    Entropy reduction: how much does the prefix reduce the model's entropy
    over the vocabulary at the position where it predicts GT answer tokens?
    
    entropy_reduction = mean_entropy(GT positions | prompt) - mean_entropy(GT positions | prompt + prefix)
    
    This directly measures the "information gained about GT" from the prefix.
    """
    gt_completion = "\nThe answer is " + gt_answer
    
    # Entropy at GT token positions conditioned on prompt only
    h_no_prefix = _compute_conditional_token_entropy_mean(
        model, tokenizer, formatted_prompt, gt_completion
    )
    
    # Entropy at GT token positions conditioned on prompt + prefix
    h_with_prefix = _compute_conditional_token_entropy_mean(
        model, tokenizer, formatted_prompt + prefix_text, gt_completion
    )
    
    return h_no_prefix - h_with_prefix  # Positive = prefix reduces entropy


def calculate_gt_surprise_ratio(model, tokenizer, formatted_prompt, prefix_text, gt_answer):
    """
    Bayes factor in log-domain: how much does the prefix update the prior on GT?
    
    surprise_ratio = P(GT|prompt+prefix) / P(GT|prompt)
    
    Returned in log-domain for numerical stability.
    Equivalent to PMI but returned as a ratio for interpretability.
    Values > 0 (in log) mean prefix makes GT more likely.
    """
    gt_completion = "\nThe answer is " + gt_answer
    
    cond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt + prefix_text, gt_completion
    )
    uncond_lp = _compute_completion_logprob(
        model, tokenizer, formatted_prompt, gt_completion
    )
    
    # Log Bayes factor
    log_bf = cond_lp - uncond_lp
    return log_bf


def calculate_top_k_gt_overlap(model, tokenizer, context, gt_answer, k=50):
    """
    Among the top-K most probable next tokens at the end of the prefix,
    how many overlap with the GT answer token set?
    
    This is a discrete, rank-based metric that's inherently robust to model scale.
    """
    gt_completion = "\nThe answer is " + gt_answer
    gt_tokens = set(tokenizer.encode(gt_completion, add_special_tokens=False))
    
    if not gt_tokens:
        return 0.0
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        top_k_tokens = torch.topk(logits, k).indices.tolist()
    
    overlap = len(set(top_k_tokens) & gt_tokens)
    return overlap / len(gt_tokens)  # Fraction of GT tokens in top-K


def calculate_gt_logprob_concentration(model, tokenizer, context, gt_answer):
    """
    Measures how "concentrated" the probability mass is on GT answer tokens
    relative to the total entropy at each prediction position.
    
    concentration = mean(-log P(gt_token)) / mean(H(vocab distribution))
    
    Low ratio = GT tokens receive disproportionately high probability
    relative to how spread out the distribution is. Scale-robust because
    it normalizes by the model's own entropy level.
    """
    gt_completion = "\nThe answer is " + gt_answer
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    completion_ids = tokenizer.encode(gt_completion, add_special_tokens=False)
    
    if not completion_ids:
        return 0.0
    
    input_ids = torch.tensor([context_ids + completion_ids], device=model.device)
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, :2048]
    
    context_len = len(context_ids)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, context_len - 1:-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # Entropy at each position
        token_entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Negative log-prob of GT tokens
        target_ids = input_ids[0, context_len:]
        n_tokens = min(len(target_ids), log_probs.shape[0])
        gt_neg_logprobs = -log_probs[torch.arange(n_tokens), target_ids[:n_tokens]]
    
    mean_gt_nll = gt_neg_logprobs.mean().item()
    mean_entropy = token_entropy[:n_tokens].mean().item()
    
    if mean_entropy == 0:
        return 0.0
    
    # Lower ratio = GT is better predicted relative to model's uncertainty
    return mean_gt_nll / mean_entropy


def calculate_layer_consistency(model, tokenizer, context):
    """
    Cosine similarity between mid-layer and last-layer hidden states at the last token.
    High consistency might indicate the model has "settled" on a representation.
    """
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        n_layers = len(hidden_states) - 1
        
        mid_state = hidden_states[n_layers // 2][0, -1, :]
        last_state = hidden_states[-1][0, -1, :]
        
        consistency = F.cosine_similarity(mid_state, last_state, dim=0).item()
    
    return consistency


def calculate_relative_perplexity(model, tokenizer, formatted_prompt, prefix_text, completion):
    """
    How much does the prefix HELP predict the answer vs no prefix at all?
    relative_ppl = unconditional_ppl - conditional_ppl
    Positive = prefix helps predict the answer.
    """
    cond_ppl = calculate_conditional_perplexity(
        model, tokenizer, formatted_prompt + prefix_text, completion
    )
    uncond_ppl = calculate_conditional_perplexity(
        model, tokenizer, formatted_prompt, completion
    )
    return uncond_ppl - cond_ppl, cond_ppl, uncond_ppl


def calculate_self_consistency_perplexity(model, tokenizer, formatted_prompt, prefix_text, own_answer, gt_answer):
    """
    Measure how well the prefix predicts the model's own extracted answer vs the GT answer.
    The gap may indicate whether the reasoning is coherently leading somewhere (even if wrong).
    """
    context = formatted_prompt + prefix_text
    own_completion = "\nThe answer is " + own_answer
    gt_completion = "\nThe answer is " + gt_answer
    
    ppl_own = calculate_conditional_perplexity(model, tokenizer, context, own_completion)
    ppl_gt = calculate_conditional_perplexity(model, tokenizer, context, gt_completion)
    
    return ppl_own, ppl_gt, ppl_own - ppl_gt


def compute_all_metrics_for_prefix(model, tokenizer, data_point):
    """Compute all candidate metrics for a single prefix data point."""
    formatted_prompt = data_point["formatted_prompt"]
    prefix_text = data_point["prefix_text"]
    gt = data_point["gt"]
    
    context = formatted_prompt + prefix_text
    gt_completion = "\nThe answer is " + gt
    
    metrics = {}
    
    # 1. Conditional perplexity to GT answer
    metrics["cond_ppl_gt"] = calculate_conditional_perplexity(
        model, tokenizer, context, gt_completion
    )
    
    # 2. Semantic similarity (context end vs answer end)
    metrics["semantic_sim_gt"] = calculate_semantic_similarity(
        model, tokenizer, context, gt_completion
    )
    
    # 3. Token entropy statistics for the answer completion
    mean_ent, max_ent, std_ent = calculate_token_entropy_stats(
        model, tokenizer, context, gt_completion
    )
    metrics["answer_entropy_mean"] = mean_ent
    metrics["answer_entropy_max"] = max_ent
    metrics["answer_entropy_std"] = std_ent
    
    # 4. Relative perplexity (does prefix help predict answer?)
    rel_ppl, _, uncond_ppl = calculate_relative_perplexity(
        model, tokenizer, formatted_prompt, prefix_text, gt_completion
    )
    metrics["relative_ppl"] = rel_ppl
    metrics["uncond_ppl"] = uncond_ppl
    
    # 5. Logit lens: probability of first answer token
    metrics["logit_lens_answer_prob"] = calculate_logit_lens_answer_prob(
        model, tokenizer, context, gt
    )
    
    # 6. Layer consistency at prefix end
    metrics["layer_consistency"] = calculate_layer_consistency(
        model, tokenizer, context
    )
    
    # 7. Self-consistency: own answer vs GT answer perplexity
    full_response = data_point["full_response"]
    own_answer = parse_prediction(full_response, gt, 'math')
    if own_answer and own_answer.strip():
        ppl_own, ppl_gt_self, ppl_gap = calculate_self_consistency_perplexity(
            model, tokenizer, formatted_prompt, prefix_text, own_answer, gt
        )
        metrics["ppl_to_own_answer"] = ppl_own
        metrics["ppl_gap_own_vs_gt"] = ppl_gap
    else:
        metrics["ppl_to_own_answer"] = float('inf')
        metrics["ppl_gap_own_vs_gt"] = 0.0
    
    # 8. Token entropy of the PREFIX itself (is the reasoning confident?)
    if len(prefix_text.strip()) > 0:
        prefix_ent_mean, prefix_ent_max, prefix_ent_std = calculate_token_entropy_stats(
            model, tokenizer, formatted_prompt, prefix_text
        )
        metrics["prefix_entropy_mean"] = prefix_ent_mean
        metrics["prefix_entropy_max"] = prefix_ent_max
        metrics["prefix_entropy_std"] = prefix_ent_std
    else:
        metrics["prefix_entropy_mean"] = 0.0
        metrics["prefix_entropy_max"] = 0.0
        metrics["prefix_entropy_std"] = 0.0
    
    # 9. Combined metrics
    metrics["combined_neg_ppl_plus_sim"] = -metrics["cond_ppl_gt"] + metrics["semantic_sim_gt"]
    
    # 10. Perplexity ratio (conditional / unconditional)
    if uncond_ppl > 0:
        metrics["ppl_ratio"] = metrics["cond_ppl_gt"] / uncond_ppl
    else:
        metrics["ppl_ratio"] = 1.0
    
    # ================================================================
    # NEW METRICS: GT-aware, information-theoretic, scale-robust
    # ================================================================
    
    # 11. GT token log-probability statistics (all tokens, not just first)
    gt_lp_sum, gt_lp_mean, gt_lp_min = calculate_gt_token_logprob_stats(
        model, tokenizer, context, gt
    )
    metrics["gt_logprob_sum"] = gt_lp_sum
    metrics["gt_logprob_mean"] = gt_lp_mean
    metrics["gt_logprob_min"] = gt_lp_min  # weakest token
    
    # 12. GT token rank statistics (robust to scale)
    gt_rank_mean, gt_rank_max, gt_frac_top10 = calculate_gt_token_rank_stats(
        model, tokenizer, context, gt
    )
    metrics["gt_rank_mean"] = gt_rank_mean
    metrics["gt_rank_max"] = gt_rank_max
    metrics["gt_frac_in_top10"] = gt_frac_top10
    
    # 13. Mutual Information: I(prefix; GT) = H(GT|prompt) - H(GT|prompt+prefix)
    mi, h_uncond, h_cond = calculate_mutual_information_prefix_gt(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    metrics["mutual_info_prefix_gt"] = mi
    metrics["gt_entropy_conditional"] = h_cond
    metrics["gt_entropy_unconditional"] = h_uncond
    
    # 14. Normalized GT log-probability (scale-invariant ratio)
    norm_lp, cond_lp, uncond_lp_val = calculate_normalized_gt_logprob(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    metrics["normalized_gt_logprob"] = norm_lp
    
    # 15. Pointwise Mutual Information
    metrics["pmi_prefix_gt"] = calculate_pointwise_mutual_information(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    
    # 16. Proper GT answer probability (geometric mean, length-normalized)
    metrics["gt_answer_prob_geomean"] = calculate_gt_answer_prob_proper(
        model, tokenizer, context, gt
    )
    
    # 17. Prefix-GT hidden state alignment
    metrics["prefix_gt_alignment"] = calculate_prefix_gt_alignment(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    
    # 18. GT entropy reduction (information gained about GT from prefix)
    metrics["gt_entropy_reduction"] = calculate_gt_entropy_reduction(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    
    # 19. GT surprise ratio (Bayes factor in log-domain)
    metrics["gt_log_bayes_factor"] = calculate_gt_surprise_ratio(
        model, tokenizer, formatted_prompt, prefix_text, gt
    )
    
    # 20. Top-K GT token overlap
    metrics["top_k_gt_overlap"] = calculate_top_k_gt_overlap(
        model, tokenizer, context, gt, k=50
    )
    
    # 21. GT log-prob concentration (normalized by entropy)
    metrics["gt_logprob_concentration"] = calculate_gt_logprob_concentration(
        model, tokenizer, context, gt
    )
    
    return metrics


def compute_metrics_for_all_prefixes(model, tokenizer, prefix_data_points):
    """Compute metrics for all prefix data points."""
    print(f"\nComputing metrics for {len(prefix_data_points)} prefix data points...")
    
    for dp in tqdm(prefix_data_points, desc="Computing metrics"):
        try:
            metrics = compute_all_metrics_for_prefix(model, tokenizer, dp)
            dp["metrics"] = metrics
        except Exception as e:
            print(f"  Error computing metrics: {e}")
            dp["metrics"] = None
            torch.cuda.empty_cache()
    
    # Filter out failed ones
    prefix_data_points = [dp for dp in prefix_data_points if dp["metrics"] is not None]
    return prefix_data_points


# ============================================================
# Step 4: Analyze correlations
# ============================================================
def analyze_correlations(prefix_data_points):
    """
    Compute correlations between each metric and the estimated pass rate.
    Also compute correlations with binary full-trajectory correctness for comparison.
    
    We compute both:
    - Raw (pooled) correlations across all data points
    - Within-problem correlations (averaged), which control for problem difficulty
    """
    if not prefix_data_points:
        print("No data points to analyze.")
        return {}
    
    # Collect all metric names
    metric_names = list(prefix_data_points[0]["metrics"].keys())
    
    # Build arrays
    pass_rates = np.array([dp["pass_rate"] for dp in prefix_data_points])
    binary_labels = np.array([1.0 if dp["full_trajectory_correct"] else 0.0 for dp in prefix_data_points])
    prefix_fractions = np.array([dp["prefix_fraction"] for dp in prefix_data_points])
    problem_ids = np.array([dp["problem_id"] for dp in prefix_data_points])
    
    # Group data points by problem_id for within-problem analysis
    from collections import defaultdict
    problem_groups = defaultdict(list)
    for i, dp in enumerate(prefix_data_points):
        problem_groups[dp["problem_id"]].append(i)
    
    print("\n" + "=" * 120)
    print(f"{'Metric':<30} {'Pooled ρ':>10} {'Within-Prob ρ':>14} {'n_probs':>8} {'p (pooled)':>12} {'p (within)':>12}")
    print("=" * 120)
    
    results = {}
    
    for name in metric_names:
        values = np.array([dp["metrics"][name] for dp in prefix_data_points])
        
        # --- Pooled (raw) correlation ---
        valid = np.isfinite(values) & np.isfinite(pass_rates)
        if valid.sum() < 10:
            print(f"  {name:<30} {'(too few valid)':>18}")
            continue
        
        v = values[valid]
        pr = pass_rates[valid]
        bl = binary_labels[valid]
        
        corr_pr, p_pr = spearmanr(v, pr)
        corr_bl, p_bl = spearmanr(v, bl)
        
        # --- Within-problem correlation ---
        # For each problem, compute Spearman correlation between metric and pass rate,
        # then average (Fisher z-transform for proper averaging)
        within_corrs = []
        for pid, indices in problem_groups.items():
            idx = np.array(indices)
            v_prob = values[idx]
            pr_prob = pass_rates[idx]
            
            # Need valid, non-constant values
            valid_mask = np.isfinite(v_prob) & np.isfinite(pr_prob)
            if valid_mask.sum() < 4:
                continue
            v_sub = v_prob[valid_mask]
            pr_sub = pr_prob[valid_mask]
            
            # Skip if either is constant (no variance)
            if np.std(v_sub) < 1e-12 or np.std(pr_sub) < 1e-12:
                continue
            
            corr_within, _ = spearmanr(v_sub, pr_sub)
            if np.isfinite(corr_within):
                within_corrs.append(corr_within)
        
        if within_corrs:
            # Fisher z-transform for averaging correlations
            z_values = [np.arctanh(np.clip(c, -0.999, 0.999)) for c in within_corrs]
            mean_z = np.mean(z_values)
            mean_within_corr = np.tanh(mean_z)
            
            # Approximate p-value via t-test on z-values
            from scipy.stats import ttest_1samp
            if len(z_values) > 1:
                _, p_within = ttest_1samp(z_values, 0)
            else:
                p_within = 1.0
            n_probs_used = len(within_corrs)
        else:
            mean_within_corr = 0.0
            p_within = 1.0
            n_probs_used = 0
        
        results[name] = {
            "corr_pass_rate": corr_pr,
            "p_pass_rate": p_pr,
            "corr_binary": corr_bl,
            "p_binary": p_bl,
            "n_valid": int(valid.sum()),
            "within_problem_corr": mean_within_corr,
            "within_problem_p": p_within,
            "n_problems_used": n_probs_used,
        }
        
        print(f"  {name:<30} {corr_pr:>+10.4f} {mean_within_corr:>+14.4f} {n_probs_used:>8d} {p_pr:>12.2e} {p_within:>12.2e}")
    
    print("=" * 120)
    
    # Per-fraction analysis
    unique_fracs = sorted(set(prefix_fractions))
    if len(unique_fracs) > 1:
        print(f"\n{'Per-Prefix-Fraction Breakdown (within-problem correlations)':}")
        print("-" * 120)
        
        for frac in unique_fracs:
            mask = prefix_fractions == frac
            subset_indices = np.where(mask)[0]
            subset = [prefix_data_points[i] for i in subset_indices]
            if len(subset) < 5:
                continue
            
            pr_sub = np.array([dp["pass_rate"] for dp in subset])
            print(f"\n  Fraction={frac:.1f} (n={len(subset)}, mean_pass_rate={pr_sub.mean():.3f})")
            
            # Group this fraction's data by problem
            frac_problem_groups = defaultdict(list)
            for i, dp in enumerate(subset):
                frac_problem_groups[dp["problem_id"]].append(i)
            
            for name in metric_names:
                vals = np.array([dp["metrics"][name] for dp in subset])
                
                # Within-problem correlation at this fraction
                frac_within_corrs = []
                for pid, indices in frac_problem_groups.items():
                    idx = np.array(indices)
                    v_sub = vals[idx]
                    pr_sub_prob = pr_sub[idx]
                    valid_m = np.isfinite(v_sub) & np.isfinite(pr_sub_prob)
                    if valid_m.sum() < 3:
                        continue
                    v_s = v_sub[valid_m]
                    pr_s = pr_sub_prob[valid_m]
                    if np.std(v_s) < 1e-12 or np.std(pr_s) < 1e-12:
                        continue
                    c, _ = spearmanr(v_s, pr_s)
                    if np.isfinite(c):
                        frac_within_corrs.append(c)
                
                if frac_within_corrs:
                    z_vals = [np.arctanh(np.clip(c, -0.999, 0.999)) for c in frac_within_corrs]
                    mean_within = np.tanh(np.mean(z_vals))
                else:
                    mean_within = 0.0
                
                # Also compute pooled for this fraction
                valid_frac = np.isfinite(vals) & np.isfinite(pr_sub)
                if valid_frac.sum() < 5:
                    continue
                corr_pooled, _ = spearmanr(vals[valid_frac], pr_sub[valid_frac])
                
                if abs(mean_within) > 0.1 or abs(corr_pooled) > 0.1:
                    print(f"    {name:<28} pooled={corr_pooled:>+.4f}  within={mean_within:>+.4f}  (n_prob={len(frac_within_corrs)})")
    
    return results


# ============================================================
# Step 5: Visualization
# ============================================================
def plot_metric_vs_pass_rate(prefix_data_points, top_metrics, save_path):
    """Scatter plots of top metrics vs pass rate."""
    if not prefix_data_points or not top_metrics:
        return
    
    n_metrics = min(len(top_metrics), 8)
    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    pass_rates = np.array([dp["pass_rate"] for dp in prefix_data_points])
    prefix_fractions = np.array([dp["prefix_fraction"] for dp in prefix_data_points])
    
    for idx, (metric_name, metric_info) in enumerate(top_metrics[:n_metrics]):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        values = np.array([dp["metrics"][metric_name] for dp in prefix_data_points])
        valid = np.isfinite(values)
        
        # Color by prefix fraction
        scatter = ax.scatter(
            values[valid], pass_rates[valid],
            c=prefix_fractions[valid], cmap='viridis',
            alpha=0.5, s=20, edgecolors='none'
        )
        
        ax.set_xlabel(metric_name, fontsize=10)
        ax.set_ylabel("Pass Rate (Estimated Value)", fontsize=10)
        corr = metric_info.get("corr_pass_rate", 0)
        ax.set_title(f"{metric_name}\nρ={corr:+.4f}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label="Prefix Fraction")
    
    # Hide unused subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scatter plots to {save_path}")


def plot_pass_rate_distribution(prefix_data_points, save_path):
    """Plot the distribution of estimated pass rates to check quality."""
    pass_rates = [dp["pass_rate"] for dp in prefix_data_points]
    full_correct = [dp["full_trajectory_correct"] for dp in prefix_data_points]
    prefix_fracs = [dp["prefix_fraction"] for dp in prefix_data_points]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Overall distribution
    axes[0].hist(pass_rates, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Estimated Pass Rate")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Prefix Values")
    axes[0].axvline(np.mean(pass_rates), color='red', linestyle='--', label=f'Mean={np.mean(pass_rates):.3f}')
    axes[0].legend()
    
    # 2. Pass rate by full trajectory correctness
    correct_pr = [pr for pr, fc in zip(pass_rates, full_correct) if fc]
    incorrect_pr = [pr for pr, fc in zip(pass_rates, full_correct) if not fc]
    axes[1].hist(correct_pr, bins=15, alpha=0.6, label=f'Correct traj (n={len(correct_pr)})', color='green', edgecolor='black')
    axes[1].hist(incorrect_pr, bins=15, alpha=0.6, label=f'Incorrect traj (n={len(incorrect_pr)})', color='red', edgecolor='black')
    axes[1].set_xlabel("Estimated Pass Rate")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Pass Rate by Full Trajectory Correctness")
    axes[1].legend()
    
    # 3. Pass rate by prefix fraction
    unique_fracs = sorted(set(prefix_fracs))
    for frac in unique_fracs:
        frac_pr = [pr for pr, pf in zip(pass_rates, prefix_fracs) if pf == frac]
        axes[2].hist(frac_pr, bins=15, alpha=0.4, label=f'Frac={frac:.1f} (n={len(frac_pr)})', edgecolor='black')
    axes[2].set_xlabel("Estimated Pass Rate")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Pass Rate by Prefix Fraction")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved pass rate distribution to {save_path}")


def plot_value_trajectories(prefix_data_points, save_path):
    """
    For each problem, plot how pass rate changes along the trajectory
    for correct vs incorrect full trajectories.
    """
    # Group by problem_id and full_response
    from collections import defaultdict
    
    trajectories = defaultdict(lambda: defaultdict(list))
    for dp in prefix_data_points:
        key = (dp["problem_id"], dp["full_response"][:100])  # Use first 100 chars as key
        trajectories[dp["problem_id"]][(dp["full_trajectory_correct"], dp["full_response"][:100])].append(
            (dp["prefix_fraction"], dp["pass_rate"])
        )
    
    n_problems = min(len(trajectories), 6)
    problem_ids = list(trajectories.keys())[:n_problems]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for plot_idx, pid in enumerate(problem_ids):
        ax = axes[plot_idx]
        
        for (is_correct, resp_key), points in trajectories[pid].items():
            points.sort(key=lambda x: x[0])
            fracs, prs = zip(*points)
            color = 'green' if is_correct else 'red'
            alpha = 0.7 if is_correct else 0.4
            ax.plot(fracs, prs, 'o-', color=color, alpha=alpha, markersize=4)
        
        ax.set_xlabel("Prefix Fraction")
        ax.set_ylabel("Estimated Pass Rate")
        ax.set_title(f"Problem {pid}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', label='Correct trajectory'),
            Line2D([0], [0], color='red', label='Incorrect trajectory'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    for idx in range(n_problems, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Value (Pass Rate) Trajectories: Green=Correct, Red=Incorrect", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved value trajectories to {save_path}")


# ============================================================
# Main Pipeline
# ============================================================
def main():
    config = Config()
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Load problems
    problems = load_math_problems(config.target_levels, n=config.n_problems)
    
    # Determine backend for generation (Phase 1 & 2)
    use_vllm = config.use_vllm and VLLM_AVAILABLE
    
    if use_vllm:
        print("Using vLLM for generation (Phase 1 & 2)...")
        llm, tokenizer = load_vllm_model(config.model_name, config)
    else:
        print("Using HuggingFace for generation (Phase 1 & 2)...")
        model, tokenizer = load_model(config.model_name)
    
    # ---- Phase 1: Collect valid problems with trajectories ----
    collected_data_file = os.path.join(config.save_dir, "2b_prefix_collected_data.pkl")
    
    if os.path.exists(collected_data_file):
        print(f"Loading collected data from {collected_data_file}...")
        with open(collected_data_file, "rb") as f:
            collected_data = pickle.load(f)
        print(f"Loaded {len(collected_data)} problems.")
    else:
        print("Phase 1: Collecting valid problems with trajectories...")
        if use_vllm:
            collected_data = collect_valid_problems_vllm(llm, tokenizer, problems, config)
        else:
            collected_data = collect_valid_problems_hf(model, tokenizer, problems, config)
        with open(collected_data_file, "wb") as f:
            pickle.dump(collected_data, f)
        print(f"Saved {len(collected_data)} problems to {collected_data_file}")
    
    # ---- Phase 2: Estimate prefix values ----
    prefix_data_file = os.path.join(config.save_dir, "2b_prefix_value_data.pkl")
    
    if os.path.exists(prefix_data_file):
        print(f"Loading prefix value data from {prefix_data_file}...")
        with open(prefix_data_file, "rb") as f:
            prefix_data_points = pickle.load(f)
        print(f"Loaded {len(prefix_data_points)} prefix data points.")
    else:
        print("\nPhase 2: Estimating prefix values via Monte Carlo sampling...")
        if use_vllm:
            prefix_data_points = estimate_prefix_values_vllm(llm, tokenizer, collected_data, config)
        else:
            prefix_data_points = estimate_prefix_values_hf(model, tokenizer, collected_data, config)
        with open(prefix_data_file, "wb") as f:
            pickle.dump(prefix_data_points, f)
        print(f"Saved {len(prefix_data_points)} prefix data points to {prefix_data_file}")
    
    # ---- Free generation model, load HF model for metrics ----
    if use_vllm:
        print("\nFreeing vLLM model and loading HF model for metric computation...")
        del llm
        torch.cuda.empty_cache()
        import gc; gc.collect()
        # Load HF model for metric computation (needs hidden states, logits)
        model, tokenizer = load_model(config.model_name)
    else:
        # HF model already loaded from generation phase; reuse it
        print("\nReusing HF model for metric computation...")
    
    # ---- Phase 3: Compute metrics ----
    metrics_data_file = os.path.join(config.save_dir, "2b_prefix_metrics_data.pkl")
    
    if os.path.exists(metrics_data_file):
        print(f"Loading metrics data from {metrics_data_file}...")
        with open(metrics_data_file, "rb") as f:
            prefix_data_points = pickle.load(f)
        print(f"Loaded {len(prefix_data_points)} data points with metrics.")
    else:
        print("\nPhase 3: Computing metrics for all prefixes...")
        prefix_data_points = compute_metrics_for_all_prefixes(model, tokenizer, prefix_data_points)
        with open(metrics_data_file, "wb") as f:
            pickle.dump(prefix_data_points, f)
        print(f"Saved metrics data to {metrics_data_file}")
    
    # ---- Phase 4: Analyze correlations ----
    print("\nPhase 4: Analyzing correlations...")
    correlation_results = analyze_correlations(prefix_data_points)
    
    # Sort by absolute correlation with pass rate
    sorted_metrics = sorted(
        correlation_results.items(),
        key=lambda x: abs(x[1].get("within_problem_corr", 0)),
        reverse=True
    )
    
    print("\n\nTop Metrics by |Within-Problem Correlation with Pass Rate|:")
    print("-" * 80)
    for name, info in sorted_metrics[:10]:
        print(f"  {name:<30} within_ρ={info['within_problem_corr']:>+.4f}  pooled_ρ={info['corr_pass_rate']:>+.4f}  (p={info['within_problem_p']:.2e})")

    # ---- Phase 5: Visualize ----
    print("\nPhase 5: Generating visualizations...")
    
    plot_pass_rate_distribution(
        prefix_data_points,
        os.path.join(config.save_dir, "prefix_pass_rate_distribution.png")
    )
    
    plot_metric_vs_pass_rate(
        prefix_data_points,
        sorted_metrics,
        os.path.join(config.save_dir, "prefix_metric_vs_passrate.png")
    )
    
    plot_value_trajectories(
        prefix_data_points,
        os.path.join(config.save_dir, "prefix_value_trajectories.png")
    )
    
    # ---- Save summary ----
    summary = {
        "config": {
            "model_name": config.model_name,
            "n_problems": len(collected_data),
            "n_prefix_data_points": len(prefix_data_points),
            "n_continuations": config.n_continuations,
            "prefix_fractions": config.prefix_fractions,
        },
        "correlations": {
            name: {
                "corr_pass_rate": float(info["corr_pass_rate"]),
                "p_pass_rate": float(info["p_pass_rate"]),
                "corr_binary": float(info["corr_binary"]),
                "within_problem_corr": float(info["within_problem_corr"]),
                "within_problem_p": float(info["within_problem_p"]),
                "n_problems_used": int(info["n_problems_used"]),
            }
            for name, info in sorted_metrics
        },
    }
    
    summary_file = os.path.join(config.save_dir, "prefix_analysis_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_file}")
    
    print("\n✅ Analysis complete!")
    print(f"   Total prefix data points: {len(prefix_data_points)}")
    print(f"   Pass rate range: [{min(dp['pass_rate'] for dp in prefix_data_points):.3f}, "
          f"{max(dp['pass_rate'] for dp in prefix_data_points):.3f}]")
    if sorted_metrics:
        best_name, best_info = sorted_metrics[0]
        print(f"   Best metric: {best_name} (within_ρ={best_info['within_problem_corr']:+.4f}, pooled_ρ={best_info['corr_pass_rate']:+.4f})")


if __name__ == "__main__":
    main()