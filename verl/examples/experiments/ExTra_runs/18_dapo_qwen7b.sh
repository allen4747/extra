#!/usr/bin/env bash

export HF_HUB_CACHE="/home/wenyang/my_efs/models"

# Auto-detect 4 free GPUs (< 2GB used) every 30 mins
find_4_free_gpus() {
    local free_gpus=()
    for i in {0..7}; do
        used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used" -lt 2000 ]; then
            free_gpus+=("$i")
        fi
        if [ "${#free_gpus[@]}" -eq 4 ]; then
            local IFS=","
            echo "${free_gpus[*]}"
            return 0
        fi
    done
    return 1
}

echo "Looking for 4 free GPUs..."
while true; do
    FREE_GPUS=$(find_4_free_gpus)
    if [ $? -eq 0 ]; then
        echo "Found 4 free GPUs ($FREE_GPUS), starting training."
        break
    fi
    echo "Not enough free GPUs available. Retrying in 30 minutes..."
    sleep 1800
done

export HF_HUB_CACHE="/home/wenyang/my_efs/models"
export CUDA_VISIBLE_DEVICES=$FREE_GPUS

MODEL_PATH="${MODEL_PATH:-/home/wenyang/my_efs/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
EXP_NAME="DAPO-JustRL-Qwen2.5-7B-LoRA"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=False \
  algorithm.guided_resampling.enable=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2000 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12192 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=18288 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18288 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
  +reward_model.reward_kwargs.max_resp_len=4096 \
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Research' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.save_start_step=150 \
  trainer.test_freq=10 \
  trainer.total_training_steps=300 \
  trainer.default_local_dir="/home/wenyang/my_efs/checkpoints/ExTra_Research/$EXP_NAME" \
  trainer.val_before_train=False \
  "$@"
