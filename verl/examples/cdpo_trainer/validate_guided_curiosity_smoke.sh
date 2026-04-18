#!/usr/bin/env bash
set -euo pipefail

# Simple smoke experiment to validate:
# 1) pass criterion: sequence_reward > 0
# 2) GRPO-style positive-gated novelty advantage
# 3) guided regeneration forward pass in main loop
#
# Usage:
#   bash examples/cdpo_trainer/validate_guided_curiosity_smoke.sh
# Optional overrides:
#   CUDA_VISIBLE_DEVICES=0 MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
#   TRAIN_FILE=$HOME/data/gsm8k/train.parquet VAL_FILE=$HOME/data/gsm8k/test.parquet \
#   bash examples/cdpo_trainer/validate_guided_curiosity_smoke.sh

export CUDA_VISIBLE_DEVICES=4,5,6,7
export RAY_DEBUG_POST_MORTEM=1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
# VAL_FILE="${VAL_FILE:-$HOME/data/aime-2024.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "[ERROR] TRAIN_FILE not found: $TRAIN_FILE"
  exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "[ERROR] VAL_FILE not found: $VAL_FILE"
  exit 1
fi

EXP_NAME="validate_guided_curiosity_$(date +%m%d_%H%M%S)"

echo "[INFO] Running smoke validation experiment"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] TRAIN_FILE=$TRAIN_FILE"
echo "[INFO] VAL_FILE=$VAL_FILE"
echo "[INFO] EXP_NAME=$EXP_NAME"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=True \
  algorithm.curiosity.novelty_reward_scale=0.1 \
  algorithm.curiosity.max_rollouts_per_prompt=64 \
  algorithm.curiosity.max_prefixes_per_prompt=128 \
  algorithm.guided_resampling.enable=True \
  algorithm.guided_resampling.tau=0.1 \
  algorithm.guided_resampling.regen_batch_size=1 \
  algorithm.guided_resampling.max_queue_size=6 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=80 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name='verl_cdpo_validation' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=20 \
  trainer.val_before_train=False \
  "$@"

echo "[INFO] Validation run finished. Check console metrics:"
echo "  - exploration/novelty_reward_mean"
echo "  - exploration/hard_prompt_count"
echo "  - exploration/guided_queue_size"
echo "  - exploration/guided_regen_batch_size"
