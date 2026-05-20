
from utils import grade_answer_verl
from transformers import AutoTokenizer
import json
import pandas as pd
from pathlib import Path
import re
from vllm import LLM, SamplingParams

CV_PROMPT = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly. 
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT 
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>: 
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval_dir', type=str, required=True, help='Evaluation directory')
parser.add_argument('--no_verifier', action='store_true',
                    help='Skip the CompassVerifier-3B fallback (rule-based only).')
parser.add_argument('--verifier_model', type=str,
                    default='opencompass/CompassVerifier-3B',
                    help='HF id of the model used to grade responses that the rule-based '
                         'verifier cannot match.')
args = parser.parse_args()

EVAL_DIR = Path(args.eval_dir)
OUTPUT_FILE = EVAL_DIR / "grading_results.json"

# Model-based fallback verifier.  Loaded lazily on first need so the script
# is fast when nothing falls back (e.g. small-scale debug runs).
USE_VERIFIER = not args.no_verifier
VERIFIER_MODEL_NAME = args.verifier_model
model_tokenizer = None
vllm_model = None
sampling_params = None


def _load_verifier():
    """Load the CompassVerifier-3B model + tokenizer on demand."""
    global model_tokenizer, vllm_model, sampling_params
    if vllm_model is not None:
        return
    print(f"[verifier] loading fallback judge: {VERIFIER_MODEL_NAME}")
    model_tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL_NAME)
    vllm_model = LLM(
        model=VERIFIER_MODEL_NAME,
        tensor_parallel_size=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

length_tokenizer = None

def get_len(seq):
    return len(length_tokenizer.encode(seq))

def get_diverse_score(sequences, n=4):
    """
    calculate the Distinct-n score。

    sequences: List[str] response list
    n: int, n-gram default=4
    """
    distinct_ngrams = set()
    total_ngrams = 0

    for seq in sequences:
        # more accurate n-gram
        # tokens = nltk.word_tokenize(seq)
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            distinct_ngrams.add(ngram)
            total_ngrams += 1

    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0

def process_jsonl_file(file_name):
    """
    Process a JSONL file and dynamically handle the number of problems.
    """
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            id = int(data["example_id"])
            while len(results) <= id:  # Ensure the list is large enough
                results.append({"gt": None, "responses": [], "question": ""})
            gt = data["answer"]
            response = data["response"]
            results[id]["gt"] = gt
            results[id]["responses"].append(response)
            # Carry the prompt forward so the model-based verifier can see
            # the original question (gen_vllm stores it under 'prompt').
            if not results[id].get("question"):
                results[id]["question"] = data.get("prompt", "")
    return results

def parse_hyperparameters_from_filename(filename):
    """
    Parse hyperparameters from the filename.
    Example filename format: {taskname}_t{temperature}_p{top_p}_n{n}-MNT{max_tokens}.jsonl
    """
    match = re.search(r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)",
                      filename)
    return match.groupdict() if match else {}

def grade_file(file_path):
    """
    Grade a single file and return the results.
    """
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None

    task_name = file_path.stem.split("_")[0]
    hyperparams["task_name"] = task_name

    if "parquet" in str(file_path):
        df = pd.read_parquet(file_path)
        num_pred = len(df["responses"][0])
    else:
        df = process_jsonl_file(file_path)
        num_pred = len(df[0]["responses"])

    results = {
        "hyperparameters": hyperparams,
        "mean_score": 0,
        "distinct_4gram": 0,
        "best_score": 0,
        "solve_none": 0,
        "solve_all": 0,
        "avg_output_length": 0,
        "format_error_rollouts": 0,
    }

    diverse = []
    avg_scores = []
    best = []
    solve_none = 0
    solve_all = 0
    without_boxed = 0
    response_lengths = []
    incorrect_data = []  # List to store incorrect responses and ground truths

    all_model_inputs = []  # Collect all prompts for batch processing
    all_responses = []  # Keep track of responses for mapping back
    all_questions = []  # Keep track of questions for mapping back
    all_ground_truths = []  # Keep track of ground truths for mapping back
    rule_based_scores = []  # Store rule-based scores for fallback logic

    for i in range(len(df)):
        if "jsonl" in str(file_path):
            responses = df[i]["responses"]
            gt = df[i]["gt"]
            question = df[i].get("question", "")  # Assuming question is part of the data
        else:
            responses = df["responses"][i]
            gt = df["reward_model"][i]["ground_truth"]
            question = df["reward_model"][i].get("question", "")

        responses_list = [str(response) for response in responses]
        if length_tokenizer:
            response_lengths += [get_len(response) for response in responses_list]
        else:
            response_lengths = [0]
        not_formated = ["boxed" not in response for response in responses_list]
        without_boxed += sum(not_formated)

        # First, use the rule-based verifier
        for response in responses_list:
            rule_score = grade_answer_verl(response, gt)
            rule_based_scores.append(rule_score)
            if not rule_score:  # If rule-based verifier fails, prepare for model-based verifier
                model_input = CV_PROMPT.format(
                    question=question,
                    gold_answer=gt,
                    llm_response=response
                )
                all_model_inputs.append(model_input)
                all_responses.append(response)
                all_questions.append(question)
                all_ground_truths.append(gt)

        diverse.append(get_diverse_score(responses_list))

    if all_model_inputs:
        if USE_VERIFIER:
            _load_verifier()
            # Wrap each prompt in the verifier's chat template
            wrapped = [
                model_tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in all_model_inputs
            ]
            outputs = vllm_model.generate(wrapped, sampling_params)
            model_based_scores = []
            for out in outputs:
                judgement = out.outputs[0].text.strip()
                # The verifier emits a single letter A / B / C.  Treat A as
                # correct; any other letter (including C = INVALID) as wrong.
                model_based_scores.append(judgement.startswith("A"))
        else:
            # Conservative fallback when the verifier is disabled.
            model_based_scores = [False] * len(all_model_inputs)

        # Combine rule-based and model-based scores
        model_idx = 0
        final_scores = []
        for rule_score in rule_based_scores:
            if rule_score:  # If rule-based verifier passed
                final_scores.append(rule_score)
            else:  # Use model-based verifier score
                final_scores.append(model_based_scores[model_idx])
                model_idx += 1
    else:
        final_scores = rule_based_scores

    # Calculate metrics
    avg_scores = [sum(final_scores[i:i + num_pred]) / num_pred for i in range(0, len(final_scores), num_pred)]
    best = [max(final_scores[i:i + num_pred]) for i in range(0, len(final_scores), num_pred)]

    solve_none = sum(1 for avg_score in avg_scores if avg_score == 0)
    solve_all = sum(1 for avg_score in avg_scores if avg_score == 1)

    results["mean_score"] = sum(avg_scores) / len(avg_scores)
    results["distinct_4gram"] = sum(diverse) / len(diverse)
    results["best_score"] = sum(best) / len(best)
    results["solve_none"] = solve_none
    results["solve_all"] = solve_all
    results["avg_output_length"] = sum(response_lengths) / len(response_lengths)
    results["format_error_rollouts"] = without_boxed

    # Save incorrect responses and ground truths to a separate file
    # incorrect_file = EVAL_DIR / f"{file_path.stem}_incorrect_data.json"
    # with incorrect_file.open("w", encoding="utf-8") as f:
    #     json.dump(incorrect_data, f, indent=4)

    return results

def main():
    all_results = []
    for file_path in EVAL_DIR.glob("*.jsonl"):
        print(f"Processing file: {file_path}")
        file_result = grade_file(file_path)
        if file_result:
            all_results.append(file_result)

    # Save results to JSON
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"Grading results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

