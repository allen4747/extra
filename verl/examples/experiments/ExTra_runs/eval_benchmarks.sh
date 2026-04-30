#!/usr/bin/env bash

# Evaluate pass@1 (avg) and pass@16 for a trained checkpoint on multiple datasets.
# Usage: CHECKPOINT_PATH=/path/to/ckpt bash 22_eval_benchmarks.sh

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export RAY_DEBUG_POST_MORTEM=1

CHECKPOINT_PATH="${CHECKPOINT_PATH:?ERROR: CHECKPOINT_PATH must be set to the trained model/checkpoint directory}"
CKPT_NAME="$(basename "$CHECKPOINT_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_outputs/${CKPT_NAME}}"
mkdir -p "$OUTPUT_DIR"

DATASETS=("MATH-500" "AMC23" "AIME24" "AIME25")
DATA_DIR="/home/wenyang/my_efs/datasets"

echo "=== Benchmark Evaluation ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output dir: $OUTPUT_DIR"

for DATASET in "${DATASETS[@]}"; do
  VAL_FILE="${DATA_DIR}/${DATASET}/test.parquet"
  
  if [ ! -f "$VAL_FILE" ]; then
    echo "Warning: $VAL_FILE not found, skipping $DATASET"
    continue
  fi

  echo "----------------------------------------"
  echo "Evaluating on $DATASET..."
  echo "----------------------------------------"

  # --- pass@1 (avg) ---
  echo "[1/2] Running pass@1 (avg) for $DATASET..."
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='ExTra_Eval' \
    trainer.experiment_name="pass1_${DATASET}_${CKPT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    "$@" \
    2>&1 | tee "$OUTPUT_DIR/${DATASET}_pass1_eval.log"

  # --- pass@16 ---
  echo "[2/2] Running pass@16 for $DATASET..."
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
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='ExTra_Eval' \
    trainer.experiment_name="pass16_${DATASET}_${CKPT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=True \
    "$@" \
    2>&1 | tee "$OUTPUT_DIR/${DATASET}_pass16_eval.log"

done

echo "Done evaluating on all dat

