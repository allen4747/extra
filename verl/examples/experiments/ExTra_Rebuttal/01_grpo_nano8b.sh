#!/usr/bin/env bash
# EMNLP rebuttal — GRPO baseline on Llama-3.1-Nemotron-Nano-8B-v1.
#
# One node, 8 x H100 (80 GB each). 150 training steps.
# Answers reviewers XB9Q W6 / xvYm W3: scale generalization beyond 1-2B.
#
# Memory budget (per GPU, ~80 GB):
#   * FSDP-8 sharded params+grads+optimizer:  ~18-22 GB
#   * vLLM engine (TP=2, so half a model):     ~8 GB
#   * Activations (grad-checkpoint, mbs=4):    ~5-10 GB
#   * KV cache + workspace (GMU=0.80):        ~40-45 GB
#   => No offloading required. If a mid-training OOM appears (rare, driven by
#      long-tail sequences), the RUNBOOK "Fallback tier 1/2/3" tells you which
#      knob to flip.
#
# Env-var overrides:
#   MODEL_PATH, TRAIN_FILE, VAL_FILE, EXP_NAME, CKPT_ROOT, TOTAL_STEPS

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH="${MODEL_PATH:-nvidia/Llama-3.1-Nemotron-Nano-8B-v1}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/my_efs/datasets/AIME24/test.parquet}"
EXP_NAME="${EXP_NAME:-01_GRPO_NanoNemotron_8B}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/my_efs/checkpoints/ExTra_Rebuttal}"
TOTAL_STEPS="${TOTAL_STEPS:-150}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=False \
  algorithm.guided_resampling.enable=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=6 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=31744 \
  actor_rollout_ref.rollout.val_kwargs.n=16 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.enable=False \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward_model.reward_kwargs.max_resp_len=8192 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=True \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Rebuttal' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.save_start_step=50 \
  trainer.test_freq=25 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.default_local_dir="$CKPT_ROOT/$EXP_NAME" \
  "$@"
