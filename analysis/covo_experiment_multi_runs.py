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
from scipy.stats import spearmanr, pointbiserialr
from torch.nn import functional as F

from simplified_evaluator.eval import parse_prediction

import random
import pickle

# Add the current directory to sys.path to import from openrlhf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from openrlhf.trainer.ppo_utils.score import process_thoughts
# except ImportError:
#     # Fallback if import fails
#     def process_thoughts(resp):
#         lines = [line.strip() for line in resp.split('\n') if line.strip()]
#         # Simple heuristic to split steps if they are long
#         return lines
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:3"
    )
    return model, tokenizer

def load_math_problems(n=None, target_levels=[4, 5]):
    print(f"Loading problems with levels {target_levels} from HuggingFaceH4/MATH-500...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    problems = []
    for item in dataset:
        lvl = item.get("level", -1)
        if lvl in target_levels:
            problems.append({
                "problem": item["problem"],
                "answer": item["answer"] # Extracted answer usually
            })
            
    print(f"Loaded {len(problems)} problems.")
    if n is not None and len(problems) > n:
        import random
        random.shuffle(problems)
        # problems = problems[:n] # Don't slice here, return all so we can search for valid ones
        print(f"Will search for {n} valid problems from {len(problems)} candidates.")
        
    return problems

def generate_responses(model, tokenizer, prompt, n=5):
    # Format prompt for Qwen
    messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
    formatted_prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = tokenizer([formatted_prompt] * n, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    responses = []
    sequences = outputs
    for i in range(len(sequences)):
        # Decode only the new tokens
        response = tokenizer.decode(sequences[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(response)
    
    return responses


def check_correctness(response, gt):
    # Very basic check: extracting boxed answer from both
    pred_ans = parse_prediction(response, gt, 'math')
    
    # Clean up
    def clean(s):
        s = s.strip().lower()
        s = s.replace(' ', '')
        return s
        
    return clean(pred_ans) == clean(gt) or clean(gt) in clean(response)

def calculate_conditional_perplexity(model, tokenizer, prompt, context, completion):
    full_text = prompt + context + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    target_ids = inputs.input_ids.clone()
    
    # Mask context
    context_len = tokenizer(prompt + context, return_tensors="pt").input_ids.shape[1]
    target_ids[:, :context_len] = -100
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, labels=target_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def calculate_semantic_surprisal(model, tokenizer, prompt, context, completion):
    """
    Calculates the Cosine Similarity between the hidden state at the end of the Context
    and the hidden state at the end of the Completion (Answer).
    
    High Similarity = Low Surprisal = The model's reasoning state aligns with the answer.
    """
    full_text = prompt + context + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    # Identify where the context ends and completion begins
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    context_len = tokenizer(prompt+context, return_tensors="pt").input_ids.shape[1]
    full_len = inputs.input_ids.shape[1]
    
    if context_len >= full_len:
        return 0.0, 0.0 # Safety fallback
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1] # [1, seq_len, hidden_dim]
        
        # Use the last token of the context to represent the reasoning state so far
        pre_answer_state = last_hidden_state[0, context_len - 1, :]
        post_answer_state = last_hidden_state[0, -1, :]
        
        similarity = torch.nn.functional.cosine_similarity(
            pre_answer_state, post_answer_state, dim=0
        ).item()
        
    return similarity

def interpolate_curve(values, target_len=10):
    if not values:
        return np.zeros(target_len)
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, values)

def compute_trajectory_metrics(model, tokenizer, prompt, response, answer_content):
    steps = process_thoughts(response)
    if not steps: return None
    
    formatted_prompt = f"<|im_start|>{prompt}\n"
    current_context = ""
    
    traj_prob = []
    traj_emb = []
    traj_ent = []
    traj_ctx_l1 = []
    traj_ctx_cos = []
    
    # Consistency with ITSELF (Current Answer)
    completion = "\nThe answer is \\boxed{" + answer_content +"}"
    # completion = "\n" + steps[-1]
    for step in steps:
        current_context += step + "\n"
        
        d_prob = calculate_conditional_perplexity(model, tokenizer, formatted_prompt, current_context, completion)
        # _, diff1, diff2 = calculate_embedding_distance(model, tokenizer, current_context, completion)
        ctx_cos = calculate_semantic_surprisal(model, tokenizer, formatted_prompt, current_context, completion)
        
        traj_prob.append(d_prob)
        # traj_emb.append(diff1)
        # traj_ent.append(diff2)
        traj_ctx_l1.append(ctx_cos)
        traj_ctx_cos.append(ctx_cos)

    # if not traj_prob: return None
        
    # Normalize per trajectory
    max_prob = max(traj_prob) if max(traj_prob) > 0 else 1.0
    norm_traj_prob = [v / max_prob for v in traj_prob]
    
    
    # min_l1 = max(traj_ctx_l1) if max(traj_ctx_l1) > 0 else 1.0
    # norm_traj_ctx_l1 = [v / min_l1 for v in traj_ctx_l1]
    # min_cos = min(traj_ctx_cos)
    # norm_traj_ctx_cos = [v / min_cos for v in traj_ctx_cos]

    
    # Combined
    norm_traj_comb = [v + k for (k,v) in zip(norm_traj_prob, traj_ctx_cos)]
    # norm_traj_comb = norm_traj_prob
    
    # Interpolate to fixed length
    norm_prob = interpolate_curve(norm_traj_prob)
    # norm_emb = interpolate_curve(norm_traj_emb)
    # norm_ent = interpolate_curve(norm_traj_ent)
    norm_comb = interpolate_curve(norm_traj_comb)
    norm_ctx_l1 = interpolate_curve(traj_ctx_l1)
    norm_ctx_cos = interpolate_curve(traj_ctx_cos)
    
    return {
        "prob": norm_prob, 
        # "emb": norm_emb,
        # "ent": norm_ent,
        "comb": norm_comb,
        "ctx_l1": norm_ctx_l1,
        "ctx_cos": norm_ctx_cos
    }

def collect_data(model, tokenizer, problems, target_n=50, n_samples=64):
    collected_data = []
    
    # Iterate over all available problems
    for p_idx, problem_data in enumerate(problems):
        if len(collected_data) >= target_n:
            break
            
        prompt = problem_data["problem"]
        gt = problem_data["answer"]
        
        print(f"\nChecking Problem {p_idx+1}/{len(problems)}. Found: {len(collected_data)}/{target_n}")
        try:
            responses = generate_responses(model, tokenizer, prompt, n=n_samples)
        except RuntimeError as e: # Handle potential OOM
            print(f"Skipping problem due to error: {e}")
            torch.cuda.empty_cache()
            continue

        correctness = [check_correctness(r, gt) for r in responses]
        n_correct = sum(correctness)
        n_incorrect = len(correctness) - n_correct
        
        if n_correct > 3 and n_incorrect > 3:
            print(f"  -> Valid! C:{n_correct}, I:{n_incorrect}")
            collected_data.append({
                "id": p_idx,
                "problem": prompt,
                "gt": gt,
                "responses": responses,
                "correctness": correctness
            })
        else:
            print(f"  -> Invalid (C:{n_correct}, I:{n_incorrect})")
            
    return collected_data

def process_metrics_and_balance(model, tokenizer, collected_data):
    processed_results = []
    # all_metrics_flat = {"prob": [], "emb": [], "ent": [], "comb": [], "labels": []}
    all_metrics_flat = {"prob": [], "ctx_l1": [], "ctx_cos": [], "comb": [], "labels": []}
    
    for idx, item in enumerate(collected_data):
        # print(f"Processing metrics for Problem {idx+1}/{len(collected_data)}")
        responses = item["responses"]
        correctness = item["correctness"]
        prompt = item["problem"]
        gt = item["gt"]
        
        # Balancing Logic
        c_indices = [i for i, c in enumerate(correctness) if c]
        i_indices = [i for i, c in enumerate(correctness) if not c]
        
        n_min = min(len(c_indices), len(i_indices))
        
        # Sample from majority
        random.shuffle(c_indices)
        random.shuffle(i_indices)
        
        selected_indices = c_indices[:n_min] + i_indices[:n_min]
        
        current_correct_trajs = []
        current_incorrect_trajs = []
        
        for r_idx in tqdm(selected_indices, desc="Trajectories", leave=False):
            r = responses[r_idx]
            is_correct = correctness[r_idx]
            # final_answer = extract_content_from_boxed(r)
            final_answer = gt
            
            metrics = compute_trajectory_metrics(model, tokenizer, prompt, r, final_answer)
            if not metrics: continue
            
            if is_correct:
                current_correct_trajs.append(metrics)
            else:
                current_incorrect_trajs.append(metrics)
                
            # Aggregate for correlation (Used "average metric value along the trajectory")

            # for key in ["prob", "ctx_l1", "ctx_cos", "comb"]:
            #     avg_val = np.mean(metrics[key])
            #     all_metrics_flat[key].append(avg_val)
            # all_metrics_flat["labels"].append(1 if is_correct else 0)

            # Use every value for correlation
            for key in ["prob", "ctx_l1", "ctx_cos", "comb"]:
                all_metrics_flat[key].extend(metrics[key])
            all_metrics_flat["labels"].extend([1 if is_correct else 0] * len(metrics["prob"]))
            
        processed_results.append({
            "id": item["id"],
            "correct": current_correct_trajs,
            "incorrect": current_incorrect_trajs
        })
        
    return processed_results, all_metrics_flat

def plot_per_problem_results(results, save_path="math500_analysis.png"):
    # metrics = ["prob", "emb", "ent", "comb"]
    metrics = ["prob", "ctx_l1", "ctx_cos", "comb"]
    titles = ["Conditional Perplexity", "Embedding Distance 1", "Embedding Distance 2", "Combined Metric"]
    
    n_problems = len(results)
    if n_problems == 0:
        print("No problems to plot.")
        return

    # Create a figure with a row for each problem and 4 columns for metrics
    plt.figure(figsize=(25, 5 * n_problems))
    x_axis = np.linspace(0, 100, 10)
    
    for i, res in enumerate(results):
        correct_trajs = res["correct"]
        incorrect_trajs = res["incorrect"]
        p_id = res["id"]
        
        for j, metric in enumerate(metrics):
            plt.subplot(n_problems, 4, i * 4 + j + 1)
            
            c_data = [t[metric] for t in correct_trajs]
            i_data = [t[metric] for t in incorrect_trajs]
            
            if c_data:
                c_mean = np.mean(c_data, axis=0)
                c_std = np.std(c_data, axis=0)
                plt.plot(x_axis, c_mean, 'g-', label="Correct", linewidth=2)
                plt.fill_between(x_axis, c_mean - c_std, c_mean + c_std, color='g', alpha=0.1)
                
            if i_data:
                i_mean = np.mean(i_data, axis=0)
                i_std = np.std(i_data, axis=0)
                plt.plot(x_axis, i_mean, 'r--', label="Incorrect", linewidth=2)
                plt.fill_between(x_axis, i_mean - i_std, i_mean + i_std, color='r', alpha=0.1)
            
            plt.title(f"Prob ID {p_id}: {titles[j]}")
            plt.xlabel("Progress (%)")
            if i == 0 and j == 0:
                plt.legend()
            plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    print(f"Saving plot to {save_path}")
    plt.savefig(save_path)

def plot_subset_results(results, n_plot=5, save_path="math500_analysis_subset.png"):
    if len(results) == 0: return
    subset = random.sample(results, min(n_plot, len(results)))
    plot_per_problem_results(subset, save_path)

def get_ranks(x):
    x = np.array(x)
    n = len(x)
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(n)
    return ranks

def manual_spearmanr(x, y):
    x_rank = get_ranks(x)
    y_rank = get_ranks(y)
    return np.corrcoef(x_rank, y_rank)[0, 1]

if __name__ == "__main__":
    model, tokenizer = load_model()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Load all problems to ensure we can find enough
    problems = load_math_problems(n=None)

    # data_file = os.path.join(os.path.dirname(__file__), "new_collected_data.pkl") # 0.5B
    data_file = os.path.join(os.path.dirname(__file__), "new_collected_data_2b.pkl") # 2.5B
    data = []

    if os.path.exists(data_file):
        print(f"Loading data from {data_file}...")
        try:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} problems from file.")
        except Exception as e:
            print(f"Error loading data: {e}")

    if not data:
        print("Collecting new data...")
        # 1. Collect Data (Find 100 items with 64 samples)
        data = collect_data(model, tokenizer, problems, target_n=100, n_samples=64)
        print(f"Collected total {len(data)} valid problems.")
        
        with open(data_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved collected data to {data_file}")
    
    # 2. Process Metrics & Balance

    # do this for 3 runs to average out randomness, print out the mean correlation and std
    all_run_metrics = []
    n_runs = 3
    for run_idx in range(n_runs):
        # randomly Sample 50 problems for analysis
        sampled_data = random.sample(data, 50)

        print(f"\n=== Processing Metrics: Run {run_idx+1}/{n_runs} ===")
        processed_results, all_metrics_flat = process_metrics_and_balance(model, tokenizer, sampled_data)
        all_run_metrics.append(all_metrics_flat)
        
        # 3. Plot per-problem results
        # plot_per_problem_results(processed_results, save_path=f"math500_analysis_run{run_idx+1}.png")
        # plot_subset_results(processed_results, n_plot=5, save_path=f"math500_analysis_subset_run{run_idx+1}.png")

    # 4. Correlation Analysis across runs
    metrics = ["prob", "ctx_l1", "ctx_cos", "comb"]
    for metric in metrics:
        corrs = []
        for run_idx in range(n_runs):
            vals = all_run_metrics[run_idx][metric]
            labels = all_run_metrics[run_idx]["labels"]
            if len(set(labels)) < 2:
                print(f"Run {run_idx+1}: Not enough label variety for metric {metric}. Skipping.")
                continue
            corr, _ = pointbiserialr(labels, vals)
            corrs.append(corr)
            print(f"Run {run_idx+1}: Correlation between {metric} and correctness: {corr:.4f}")
        if corrs:
            mean_corr = np.mean(corrs)
            std_corr = np.std(corrs)
            print(f"\nOverall Correlation for {metric}: Mean={mean_corr:.4f}, Std={std_corr:.4f}\n")
    
    print("Experiment completed.")

    # results, all_metrics = process_metrics_and_balance(model, tokenizer, data)
    
    # # 3. Plot 5 random
    # plot_subset_results(results, n_plot=5, save_path=os.path.join(os.path.dirname(__file__), "math50_analysis_plot.png"))
    
    # # 4. Rank Correlation
    # print("\nRank Correlations (Metric Average vs Correctness Label):")
    # # for key in ["prob", "emb", "ent", "comb"]:
    # for key in ["prob", "ctx_l1", "ctx_cos", "comb"]:
    #     try:
    #         corr, _ = spearmanr(all_metrics[key], all_metrics["labels"])
    #     except Exception:
    #         corr = manual_spearmanr(all_metrics[key], all_metrics["labels"])
    #     print(f"  {key}: {corr:.4f}")
    # print("Pointbiserialr Correlation:")
    # for key in ["prob", "ctx_l1", "ctx_cos", "comb"]:
    #     corr, _ = pointbiserialr(all_metrics[key], all_metrics["labels"])
    #     print(f"  {key}: {corr:.4f}")
