#!/usr/bin/env bash
# Evaluate pass@1 and pass@8 for a trained checkpoint.
#
# Usage:
#   CHECKPOINT_PATH=/path/to/ckpt bash 06_passk_eval.sh
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   VAL_FILE              (default: $HOME/data/math500/test.parquet)
#   GSM8K_FILE            (default: $HOME/data/gsm8k/test.parquet)
#   N_SAMPLES             (default: 8, used for pass@8)
#   OUTPUT_DIR            (default: ./eval_outputs/<checkpoint_name>)

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_DEBUG_POST_MORTEM=1

CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH must be set to the trained model/checkpoint directory}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
GSM8K_FILE="${GSM8K_FILE:-$HOME/data/gsm8k/test.parquet}"
N_SAMPLES="${N_SAMPLES:-8}"
CKPT_NAME="$(basename "$CHECKPOINT_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/${CKPT_NAME}}"
mkdir -p "$OUTPUT_DIR"

echo "=== Pass@k Evaluation ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "N samples:  $N_SAMPLES (pass@1 + pass@$N_SAMPLES)"
echo "MATH-500:   $VAL_FILE"
echo "GSM8K:      $GSM8K_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

# --- MATH-500 evaluation ---
echo "[1/2] Evaluating on MATH-500..."
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
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Eval' \
  trainer.experiment_name="passk_math500_${CKPT_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=True \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/math500_eval.log"

# --- GSM8K evaluation ---
echo "[2/2] Evaluating on GSM8K..."
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.curiosity.enable=False \
  algorithm.guided_resampling.enable=False \
  data.val_files="$GSM8K_FILE" \
  data.train_files="$GSM8K_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=4096 \
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
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Eval' \
  trainer.experiment_name="passk_gsm8k_${CKPT_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_training_steps=1 \
  trainer.val_before_train=True \
  "$@" \
  2>&1 | tee "$OUTPUT_DIR/gsm8k_eval.log"

echo ""
echo "=== Evaluation complete. Logs saved to $OUTPUT_DIR ==="
