#!/usr/bin/env bash
# ExTra Full (curiosity + resampling) on Qwen3-1.7B — 4 GPUs.
# Novelty reward scale=0.01 (conservative to avoid entropy explosion).

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
EXP_NAME="03_ExTra_Full_Qwen3"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.curiosity.enable=True \
  algorithm.curiosity.novelty_reward_scale=0.01 \
  algorithm.curiosity.max_rollouts_per_prompt=16 \
  algorithm.curiosity.max_prefixes_per_prompt=128 \
  algorithm.guided_resampling.enable=True \
  algorithm.guided_resampling.tau=0.1 \
  algorithm.guided_resampling.regen_batch_size=16 \
  algorithm.guided_resampling.max_queue_size=512 \
  algorithm.guided_resampling.warmup_steps=30 \
  algorithm.guided_resampling.reasoning_split_mode=paragraph \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=512 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=65536 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=10 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.enable=False \
  reward_model.reward_manager=dapo \
  trainer.val_before_train=True \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Qwen' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=10 \
  "$@"
