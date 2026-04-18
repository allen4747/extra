import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import re
from scipy.stats import spearmanr
import pickle

# --- SETUP & LOADING ---
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct"):
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def load_math_problems(n=None, target_levels=[4]):
    print(f"Loading problems with levels {target_levels}...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    problems = []
    for item in dataset:
        if item.get("level", -1) in target_levels:
            problems.append({
                "problem": item["problem"],
                "solution": item["solution"], # Contains full reasoning trace
                "answer": item["answer"]
            })
    print(f"Loaded {len(problems)} problems.")
    return problems

def process_thoughts(resp):
    return [line.strip() for line in resp.split('\n') if line.strip()]

def extract_content_from_boxed(text):
    matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if matches: return matches[-1]
    return text.strip().split('\n')[-1]

def check_correctness(response, ground_truth):
    pred_ans = extract_content_from_boxed(response)
    def clean(s): return s.strip().lower().replace(' ', '').replace('$', '')
    return clean(pred_ans) == clean(ground_truth) or clean(ground_truth) in clean(response)

# --- NEW METRIC: FIXED ANCHOR SIMILARITY ---

def get_last_hidden_state(model, tokenizer, text):
    """Helper to get the final token's hidden state."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # Truncate if too long (rare but possible)
    if inputs.input_ids.shape[1] > model.config.max_position_embeddings:
        inputs.input_ids = inputs.input_ids[:, -model.config.max_position_embeddings:]
        inputs.attention_mask = inputs.attention_mask[:, -model.config.max_position_embeddings:]
        
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the last layer (or -2 for slightly more semantic, less lexical rep)
        # We use -1 here for strongest signal on "next token" prediction alignment
        return outputs.hidden_states[-1][:, -1, :] # [1, Dim]

def compute_anchor_metrics(model, tokenizer, prompt, response, gold_solution):
    """
    Computes similarity of each step to the 'Gold Anchor' state.
    """
    steps = process_thoughts(response)
    if not steps: return None
    
    # 1. Compute Fixed Gold Anchor (The "Correct State of Mind")
    # We use Prompt + Gold Solution to establish the target vector.
    formatted_prompt = f"<|im_start|>{prompt}\n"
    gold_text = formatted_prompt + gold_solution
    anchor_embedding = get_last_hidden_state(model, tokenizer, gold_text)
    
    current_context = formatted_prompt
    traj_sim = []
    
    # 2. Compare trajectory steps to Anchor
    for step in steps:
        current_context += step + "\n"
        
        # Get state of current partial reasoning
        current_embedding = get_last_hidden_state(model, tokenizer, current_context)
        
        # Calculate Cosine Similarity to Anchor
        sim = F.cosine_similarity(current_embedding, anchor_embedding).item()
        traj_sim.append(sim)
        
    # Interpolate for plotting
    def interpolate(vals, n=10):
        if not vals: return np.zeros(n)
        return np.interp(np.linspace(0,1,n), np.linspace(0,1,len(vals)), vals)
        
    return {
        "anchor_sim": interpolate(traj_sim),
        "raw_sim": traj_sim
    }

# --- MAIN LOOP ---

def collect_and_evaluate(model, tokenizer, problems, target_n=20, n_samples=8):
    collected_data = []
    all_metrics = {"mean_sim": [], "final_sim": [], "labels": []}
    
    print(f"Collecting data using Fixed Anchor Similarity...")
    
    for p_idx, problem_data in enumerate(problems):
        if len(collected_data) >= target_n: break
        
        prompt = problem_data["problem"]
        gt_short = problem_data["answer"]
        gt_full_sol = problem_data["solution"] # IMPORTANT: We need the full solution text
        
        try:
            # Generate samples
            responses = []
            inputs = tokenizer([f"<|im_start|>{prompt}\n"]*n_samples, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
            for seq in out:
                responses.append(tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True))
        except RuntimeError: 
            torch.cuda.empty_cache()
            continue

        correctness = [check_correctness(r, gt_short) for r in responses]
        
        if sum(correctness) > 0 and sum(correctness) < len(correctness):
            print(f"Problem {p_idx}: Contrast Found (C:{sum(correctness)}/I:{len(correctness)-sum(correctness)})")
            
            p_data = {"correct": [], "incorrect": []}
            
            for r_idx, r in enumerate(responses):
                is_correct = correctness[r_idx]
                
                # Compute Metrics against the FULL Gold Solution
                metrics = compute_anchor_metrics(model, tokenizer, prompt, r, gt_full_sol)
                if not metrics: continue
                
                # Store for Correlation
                # We track both Mean (whole path quality) and Final (end state quality)
                all_metrics["mean_sim"].append(np.mean(metrics["raw_sim"]))
                all_metrics["final_sim"].append(metrics["raw_sim"][-1])
                all_metrics["labels"].append(1 if is_correct else 0)
                
                if is_correct: p_data["correct"].append(metrics)
                else: p_data["incorrect"].append(metrics)
            
            collected_data.append({"id": p_idx, "trajs": p_data})
            
    return collected_data, all_metrics

def plot_anchor_results(results):
    if not results: return
    plt.figure(figsize=(10, 6))
    
    c_data, i_data = [], []
    for item in results:
        c_data.extend([t["anchor_sim"] for t in item["trajs"]["correct"]])
        i_data.extend([t["anchor_sim"] for t in item["trajs"]["incorrect"]])
    
    x = np.linspace(0, 100, 10)
    
    if c_data:
        mu = np.mean(c_data, 0)
        plt.plot(x, mu, 'g-', label="Correct Trajectory", linewidth=2)
        plt.fill_between(x, mu - np.std(c_data,0)*0.1, mu + np.std(c_data,0)*0.1, color='g', alpha=0.1)
        
    if i_data:
        mu = np.mean(i_data, 0)
        plt.plot(x, mu, 'r--', label="Incorrect Trajectory", linewidth=2)
        plt.fill_between(x, mu - np.std(i_data,0)*0.1, mu + np.std(i_data,0)*0.1, color='r', alpha=0.1)
        
    plt.title("Fixed Anchor Similarity\n(Distance to Ground Truth Solution Embedding)")
    plt.xlabel("Reasoning Progress (%)")
    plt.ylabel("Cosine Similarity to Gold State")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("anchor_similarity.png")
    print("Plot saved.")

if __name__ == "__main__":
    torch.manual_seed(42)
    model, tokenizer = load_model()
    problems = load_math_problems(target_levels=[4]) # Harder problems show better contrast
    
    results, metrics = collect_and_evaluate(model, tokenizer, problems, target_n=20)
    plot_anchor_results(results)
    
    print("\nSpearman Correlations:")
    if metrics["labels"]:
        c_mean, _ = spearmanr(metrics["mean_sim"], metrics["labels"])
        c_final, _ = spearmanr(metrics["final_sim"], metrics["labels"])
        print(f"  Mean Path Similarity:  {c_mean:.4f}")
        print(f"  Final State Similarity: {c_final:.4f}")