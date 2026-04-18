#!/usr/bin/env bash

export RAY_TMPDIR=/external1/wenyang/ray_tmp

# Auto-detect two free GPUs with hold-and-wait:
# When only one GPU is free, occupy it immediately to prevent others from
# grabbing it, then wait for a second GPU to become available.
find_free_gpu() {
    for i in {0..7}; do
        used=$(nvidia-smi -i "$i" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used" -lt 2000 ]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

occupy_gpu() {
    CUDA_VISIBLE_DEVICES=$1 python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(40000, 40000, device=d)
print(f'Holding GPU $1', flush=True)
while True:
    time.sleep(30)
" &
    echo $!
}

GPU1="" ; GPU2="" ; HOLD_PID=""

echo "Looking for two free GPUs..."
while true; do
    if [ -z "$GPU1" ]; then
        GPU1=$(find_free_gpu)
        if [ $? -eq 0 ]; then
            echo "Found first free GPU $GPU1, occupying it..."
            HOLD_PID=$(occupy_gpu "$GPU1")
            sleep 2
        else
            GPU1=""
            echo "No free GPU yet. Retrying in 90s..."
            sleep 90
            continue
        fi
    fi
    GPU2=$(find_free_gpu)
    if [ $? -eq 0 ] && [ "$GPU2" != "$GPU1" ]; then
        echo "Found second free GPU $GPU2."
        break
    fi
    GPU2=""
    echo "Waiting for a second free GPU (holding GPU $GPU1)... Retrying in 90s."
    sleep 90
done

if [ -n "$HOLD_PID" ]; then
    kill "$HOLD_PID" 2>/dev/null
    wait "$HOLD_PID" 2>/dev/null
    sleep 2
fi
echo "Starting training on GPUs $GPU1,$GPU2"
export CUDA_VISIBLE_DEVICES=$GPU1,$GPU2

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$HOME/data/math_dapo/train.parquet}"
VAL_FILE="${VAL_FILE:-$HOME/data/math500/test.parquet}"
EXP_NAME="ExTra-JustRL-Qwen2.5-1.5B"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=True \
  algorithm.curiosity.novelty_reward_scale=0.1 \
  algorithm.curiosity.max_rollouts_per_prompt=16 \
  algorithm.curiosity.max_prefixes_per_prompt=128 \
  algorithm.guided_resampling.enable=True \
  algorithm.guided_resampling.tau=0.1 \
  algorithm.guided_resampling.regen_batch_size=32 \
  algorithm.guided_resampling.max_queue_size=512 \
  algorithm.guided_resampling.warmup_steps=10 \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=2000 \
  data.max_response_length=4096 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Research' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.save_start_step=100 \
  trainer.test_freq=10 \
  trainer.total_training_steps=300 \
  trainer.default_local_dir="/external1/wenyang/checkpoints/ExTra_Research/$EXP_NAME" \
  trainer.val_before_train=False \
  "$@"