#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEBUG_POST_MORTEM=1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
# VAL_FILE="${VAL_FILE:-$HOME/data/aime-2024.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
# EXP_NAME="validate_guided_curiosity_$(date +%m%d_%H%M%S)"
EXP_NAME='ExTra-Qwen2.5-1.5B-CuriosityOnly1'


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
  algorithm.guided_resampling.enable=False \
  algorithm.guided_resampling.tau=0.1 \
  algorithm.guided_resampling.regen_batch_size=128 \
  algorithm.guided_resampling.max_queue_size=512 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2000 \
  data.max_response_length=3000 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='dapo_qw2.5_1.5b_extra_test1' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  trainer.total_training_steps=300 \
  trainer.val_before_train=False \
  "$@"

# trainer.total_training_steps=20 \

echo "[INFO] Validation run finished. Check console metrics:"
echo "  - exploration/novelty_reward_mean"
echo "  - exploration/hard_prompt_count"
echo "  - exploration/guided_queue_size"
echo "  - exploration/guided_regen_batch_size"
