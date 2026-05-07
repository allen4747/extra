import os
# Must set CUDA_VISIBLE_DEVICES before any CUDA/torch init
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
# Add vLLM imports
from vllm import LLM, SamplingParams
from datasets import load_dataset
import re
import random
from tqdm import tqdm
import torch.nn.functional as F

from simplified_evaluator.eval import parse_prediction

# Add experiment directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl"))
from verl.trainer.ppo.metric_utils import process_thoughts, process_thoughts_reasoning

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REASONING_SPLIT_MODE = "paragraph"  # "line" or "paragraph"
MAX_TOKENS = 15360

def load_models(model_name=MODEL_NAME):
    print(f"Loading scoring model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load HF model for scoring (hidden states access)
    # CUDA_VISIBLE_DEVICES=4,5,6,7 is set at the top of the file,
    # so cuda:0 = physical GPU 4
    scoring_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda:0"
    )
    scoring_model.eval()

    print(f"Loading vLLM generation model {model_name}...")
    generation_llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        dtype="float16",
        tensor_parallel_size=1,
        max_model_len=MAX_TOKENS + 2048,
    )
    return scoring_model, tokenizer, generation_llm

def load_math_problems(n=None, target_levels=[5]):
    print(f"Loading problems with levels {target_levels} from HuggingFaceH4/MATH-500...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []
    
    problems = []
    for item in dataset:
        lvl = item.get("level", -1)
        if lvl in target_levels:
            problems.append({
                "problem": item["problem"],
                "solution": item["solution"], 
                "answer": item["answer"]
            })
            
    print(f"Loaded {len(problems)} problems.")
    if n is not None:
        if len(problems) > n:
            random.seed(42)
            random.shuffle(problems)
            problems = problems[:n]
            print(f"Selected {n} problems.")
    return problems

def generate_responses(llm, tokenizer, prompt, n=5):
    content = prompt + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    messages = [{"role": "user", "content": content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        n=n
    )
    
    # vLLM generation
    outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
    
    responses = []
    # vLLM returns one RequestOutput per prompt. We have 1 prompt.
    for output in outputs:
        for comp in output.outputs:
            responses.append(comp.text)
    
    return responses, formatted_prompt

def generate_from_prefix(llm, tokenizer, prefix, n=5):
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_TOKENS,
        n=n
    )
    
    # vLLM generation
    outputs = llm.generate([prefix], sampling_params, use_tqdm=False)
    
    responses = []
    for output in outputs:
        for comp in output.outputs:
            responses.append(comp.text)
    
    return responses


def check_correctness(response, ground_truth):
    pred_ans = parse_prediction(response, ground_truth, 'math')
    
    gt_ans = ground_truth
    
    def clean(s):
        s = s.strip().lower()
        s = s.replace(' ', '')
        if not s: return "EMPTY_STRING"
        return s
        
    return clean(pred_ans) == clean(gt_ans) or clean(gt_ans) in clean(response)

def calculate_conditional_perplexity(model, tokenizer, context, completion):
    full_text = context + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    target_ids = inputs.input_ids.clone()
    
    context_len = tokenizer(context, return_tensors="pt").input_ids.shape[1]
    
    # Boundary check
    if context_len >= inputs.input_ids.shape[1]:
        return float('inf')

    target_ids[:, :context_len] = -100
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, labels=target_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

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
    
    return token_entropy.max().item()

def calculate_semantic_similarity(model, tokenizer, context, completion):
    """
    Calculates the Cosine Similarity between the hidden state at the end of the Context
    and the hidden state at the end of the Completion (Answer).
    
    High Similarity = Low Surprisal = The model's reasoning state aligns with the answer.
    """
    full_text = context + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    # Identify where the context ends and completion begins
    context_inputs = tokenizer(context, return_tensors="pt")
    context_len = context_inputs.input_ids.shape[1]
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
        
    return similarity, pre_answer_state.cpu()

def run_experiment():
    # Load both models
    scoring_model, tokenizer, gen_llm = load_models()
    # Limit to 50 level-5 problems
    problems = load_math_problems(n=50, target_levels=[5])
    
    random_successes = []
    resampling_successes = []
    
    for idx, p_data in enumerate(tqdm(problems, desc="Evaluating")):
        prompt = p_data["problem"]
        gt = p_data["answer"]
        
        # 1. Random Sampling: Generate 32 samples total.
        #    Split into sets A (16) and B (16)
        
        try:
            # Pass gen_llm instead of model
            batch_a, formatted_prompt = generate_responses(gen_llm, tokenizer, prompt, n=16)
        except Exception as e:
            print(f"Error generating Batch A: {e}")
            torch.cuda.empty_cache()
            continue
            
        # Analyze Batch A to find best prefix
        
        completion = "\nThe answer is " + gt

        # Store the ppl and surprisal for each context
        context_dict = {}

        for resp in batch_a:
            # ans = extract_content_from_boxed(resp)
            steps = process_thoughts_reasoning(resp, mode=REASONING_SPLIT_MODE)
            if not steps:
                # If no steps, treat whole response as one step?
                steps = [resp]

            current_context = formatted_prompt + "<think>\n"
            prefix_text = ""
            for step in steps[:-1]:  # Exclude last step to avoid overfitting
                current_context += step + "\n"
                prefix_text += step + "\n"
                # Use scoring_model for calculations
                # ppl = calculate_conditional_perplexity(scoring_model, tokenizer, current_context, completion)
                _, emb = calculate_semantic_similarity(scoring_model, tokenizer, current_context, completion)
                
                # Check for valid embedding return
                # if emb is not None:
                #     context_dict[current_context] = (ppl, similarity, emb)
                # entropy = calculate_token_entropy_stats(scoring_model, tokenizer, current_context, completion)
                entropy = calculate_token_entropy_stats(scoring_model, tokenizer, formatted_prompt, prefix_text)
                context_dict[current_context] = (entropy, emb)

        # Normalize the ppl scores by dividing by max ppl
        if not context_dict:
            print(f"No valid contexts found in Batch A for problem {idx}. Skipping.")
            continue

        # --- New Semantic Consistency Logic ---
        context_list = list(context_dict.keys())
        
        # 1. Prepare Embeddings for pairwise comparison
        # Stack embeddings: [num_contexts, hidden_dim]
        all_embeddings = torch.stack([context_dict[k][1] for k in context_list])
        # Normalize for cosine similarity
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
        
        # Compute Similarity Matrix: [num_contexts, num_contexts]
        sim_matrix = torch.mm(all_embeddings, all_embeddings.t())
        
        raw_scores = torch.tensor([context_dict[k][0] for k in context_list], dtype=torch.float32)
        tau = 0.1
        weights = torch.nn.functional.softmax(sim_matrix.to(dtype=torch.float32) / tau, dim=1)
        smoothed_scores = torch.mv(weights, raw_scores)

        candidates = []
        for i, ctx in enumerate(context_list):
            candidates.append((ctx, smoothed_scores[i].item()))
        
        candidates.sort(key=lambda x: x[1])

        best_prefix = candidates[0][0]
        best_ppl = candidates[0][1] # Storing score for logging
        # ----------------------------------------

        # --- New Entropy Minimization Logic ---
        # entropy_values = list(context_dict.values())
        # context_list = list(context_dict.keys())
        # candidates = list(zip(context_list, entropy_values))
        # candidates.sort(key=lambda x: x[1])  # Minimize entropy
        # best_prefix = candidates[0][0]
        # best_ppl = candidates[0][1]  # Storing score for logging

        if best_prefix is None:
            best_prefix = formatted_prompt
            
        # Generate Batch C (Resampled from prefix) (16 samples)
        try:
            # Pass gen_llm
            batch_c = generate_from_prefix(gen_llm, tokenizer, best_prefix, n=16)
        except Exception as e:
            print(f"Error generating Batch C: {e}")
            torch.cuda.empty_cache()
            batch_c = []
            
        # Generate Batch B (Random Continued) (16 samples)
        try:
            # Pass gen_llm
            batch_b, _ = generate_responses(gen_llm, tokenizer, prompt, n=16)
        except Exception as e:
            print(f"Error generating Batch B: {e}")
            torch.cuda.empty_cache()
            batch_b = []
            
        # Evaluation
        # Baseline Pass@32: Check Batch A + Batch B
        baseline_hits = [check_correctness(r, gt) for r in (batch_a + batch_b)]
        baseline_pass = any(baseline_hits)
        random_successes.append(1 if baseline_pass else 0)
        
        # Resampling Pass@32: Check Batch A + Batch C
        resampling_hits = [check_correctness(r, gt) for r in (batch_a + batch_c)]
        resampling_pass = any(resampling_hits)
        resampling_successes.append(1 if resampling_pass else 0)
        
        print(f"Prob {idx} | Baseline: {baseline_pass} | Resampling: {resampling_pass} | Best PPL: {best_ppl:.4f}")

    if random_successes:
        print(f"\nFinal Results on {len(random_successes)} problems:")
        print(f"Random Sampling Pass@32: {np.mean(random_successes):.4f}")
        print(f"Resampling Pass@32:      {np.mean(resampling_successes):.4f}")
    else:
        print("No problems evaluated.")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("Interrupted by user.")
