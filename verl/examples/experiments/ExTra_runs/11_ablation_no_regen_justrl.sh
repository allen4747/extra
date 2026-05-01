#!/usr/bin/env bash
# Ablation: Curiosity only (no regeneration) — JustRL config
# FIX: use ppo_mini_batch_size=64 and 2 GPUs to match scripts 08/09
#      (previous version used mini_batch=256 on 1 GPU → 128 grad accum
#       steps which diluted the signal and caused training collapse)

export RAY_TMPDIR=/external1/wenyang/ray_tmp

# Auto-detect two free GPUs with hold-and-wait
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
EXP_NAME="ExTra-NoRegen-JustRL-Qwen2.5-1.5B-v2"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.curiosity.enable=True \
  algorithm.curiosity.novelty_reward_scale=0.1 \
  algorithm.curiosity.max_rollouts_per_prompt=16 \
  algorithm.curiosity.max_prefixes_per_prompt=128 \
  algorithm.guided_resampling.enable=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=256 \
  data.max_prompt_length=1024 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=8 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward_model.reward_kwargs.max_resp_len=8192 \
  trainer.critic_warmup=0 \
  trainer.logger='["console", "wandb"]' \
  trainer.project_name='ExTra_Research' \
  trainer.experiment_name="$EXP_NAME" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.save_start_step=0 \
  trainer.test_freq=10 \
  trainer.total_training_steps=300 \
  trainer.default_local_dir="/external1/wenyang/checkpoints/ExTra_Research/$EXP_NAME" \
  trainer.val_before_train=True \
  "$@"
