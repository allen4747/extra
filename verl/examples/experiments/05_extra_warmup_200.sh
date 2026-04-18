#!/usr/bin/env bash

export RAY_TMPDIR=/external1/wenyang/ray_tmp
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RAY_DEBUG_POST_MORTEM=1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
EXP_NAME="ExTra-Warmup-200-Late"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=True \
  algorithm.curiosity.novelty_reward_scale=0.1 \
  algorithm.guided_resampling.enable=True \
  algorithm.guided_resampling.tau=0.1 \
  algorithm.guided_resampling.regen_batch_size=128 \
  algorithm.guided_resampling.warmup_steps=200 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=1024 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  trainer.project_name='ExTra_Timing_Study' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.total_epochs=10 \
  trainer.save_freq=50 \
  trainer.default_local_dir="/external1/wenyang/checkpoints/ExTra_Timing_Study/$EXP_NAME" \
  trainer.save_start_step=100 \
  "$@"
