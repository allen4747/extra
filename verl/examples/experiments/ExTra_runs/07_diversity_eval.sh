#!/usr/bin/env bash
# Evaluate output diversity metrics (avg pairwise cosine distance, log-det volume)
# for a trained checkpoint on MATH-500.
#
# Usage:
#   CHECKPOINT_PATH=/path/to/ckpt bash 07_diversity_eval.sh
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   VAL_FILE              (default: $HOME/data/math500/test.parquet)
#   N_SAMPLES             (default: 8)
#   OUTPUT_DIR            (default: ./diversity_outputs/<checkpoint_name>)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_DEBUG_POST_MORTEM=1

CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH must be set}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
N_SAMPLES="${N_SAMPLES:-8}"
CKPT_NAME="$(basename "$CHECKPOINT_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-./diversity_outputs/${CKPT_NAME}}"
ROLLOUT_DATA_DIR="${OUTPUT_DIR}/rollouts"
mkdir -p "$ROLLOUT_DATA_DIR"

echo "=== Diversity Metric Evaluation ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Dataset:    $VAL_FILE"
echo "N samples:  $N_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Generate N responses per problem and save rollout data
echo "[1/2] Generating $N_SAMPLES responses per problem..."
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.curiosity.enable=False \
  algorithm.guided_resampling.enable=False \
  data.val_files="$VAL_FILE" \
  data.train_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$CHECKPOINT_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.n="$N_SAMPLES" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=True \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/generation.log"

# Step 2: Compute diversity metrics from saved rollout data
echo "[2/2] Computing diversity metrics..."
python3 - <<'PYEOF'
import os, glob, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

output_dir = os.environ.get("OUTPUT_DIR", "./diversity_outputs")
rollout_dir = os.path.join(output_dir, "rollouts")
ckpt_name = os.environ.get("CKPT_NAME", "checkpoint")

print(f"Loading rollout data from {rollout_dir} ...")

# Collect all response texts grouped by prompt
# (Exact loading depends on verl's rollout_data_dir format;
#  adapt the loader below to match the saved file structure)
response_groups = {}  # prompt_id -> list of response strings
data_files = glob.glob(os.path.join(rollout_dir, "*.pkl")) + \
             glob.glob(os.path.join(rollout_dir, "*.json"))

if not data_files:
    print(f"WARNING: No rollout data files found in {rollout_dir}.")
    print("Ensure trainer.rollout_data_dir is supported by your verl version.")
    exit(0)

for fpath in data_files:
    if fpath.endswith(".pkl"):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
    else:
        with open(fpath) as f:
            data = json.load(f)
    # data is expected to be a list of dicts with keys: prompt_id, responses
    for item in data:
        pid = item.get("prompt_id", item.get("uid", str(hash(item.get("prompt", "")))))
        responses = item.get("responses", item.get("completions", []))
        if pid not in response_groups:
            response_groups[pid] = []
        response_groups[pid].extend(responses)

print(f"Found {len(response_groups)} prompts with rollout data.")

# Embed all responses
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

avg_cosine_distances = []
logdet_volumes = []

for pid, responses in response_groups.items():
    if len(responses) < 2:
        continue
    embeddings = model.encode(responses, convert_to_tensor=True, normalize_embeddings=True)
    # (1) Average pairwise cosine distance
    n = embeddings.shape[0]
    sim_matrix = (embeddings @ embeddings.T).cpu().numpy()
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = sim_matrix[mask]
    avg_dist = float(np.mean(1.0 - pairwise_sims))
    avg_cosine_distances.append(avg_dist)
    # (2) Log-det volume
    Z = embeddings.cpu().numpy()  # (n, d)
    gram = Z @ Z.T + 1e-6 * np.eye(n)
    sign, logdet = np.linalg.slogdet(gram)
    if sign > 0:
        logdet_volumes.append(float(logdet))

results = {
    "checkpoint": ckpt_name,
    "n_prompts": len(response_groups),
    "avg_pairwise_cosine_distance": float(np.mean(avg_cosine_distances)) if avg_cosine_distances else None,
    "avg_logdet_volume": float(np.mean(logdet_volumes)) if logdet_volumes else None,
    "std_pairwise_cosine_distance": float(np.std(avg_cosine_distances)) if avg_cosine_distances else None,
}

print("\n=== Diversity Results ===")
for k, v in results.items():
    print(f"  {k}: {v}")

out_file = os.path.join(output_dir, "diversity_results.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_file}")
PYEOF

echo ""
echo "=== Diversity evaluation complete. Results in $OUTPUT_DIR/diversity_results.json ==="
