#!/usr/bin/env bash
# Hopper server (maplecg-hopper): 3e-6 runs at step 250.
#
# Usage:
#   NUM_GPUS=4 bash eval_batch_hopper.sh           # auto-detect 4 free GPUs and wait
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_batch_hopper.sh   # use specific GPUs

set -e

EXTRA_REPO="${EXTRA_REPO:-$HOME/ExTra}"
EVAL_SCRIPT="${EXTRA_REPO}/evals/eval_extra_qwen.sh"

# Auto-detect GPUs if NUM_GPUS is set and CUDA_VISIBLE_DEVICES not provided
if [ -n "$NUM_GPUS" ] && [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    source "${EXTRA_REPO}/evals/wait_for_gpus.sh"
fi

export CKPT_BASE="${CKPT_BASE:-/external1/wenyang/checkpoints/ExTra_Qwen}"
export DATA_DIR="${DATA_DIR:-/home/wenyang/my_efs/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-./eval_outputs_v2}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-16}"

RUNS=(
    "03_ExTra_Full_Qwen3_0.01_hlr:250"
    "03_ExTra_Full_Qwen3_0.1_hlr:250"
    "10_ExTra_PolicySurprise_3e6_Qwen3_hopper:250"
)

for run_step in "${RUNS[@]}"; do
    RUN_NAME="${run_step%:*}"
    STEP="${run_step#*:}"
    CKPT_DIR="${CKPT_BASE}/${RUN_NAME}/global_step_${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[SKIP] $RUN_NAME @ step $STEP: checkpoint not found at $CKPT_DIR"
        continue
    fi

    METRICS_FILE="${OUTPUT_BASE}/${RUN_NAME}/step_${STEP}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "[SKIP] $RUN_NAME @ step $STEP: metrics already exist"
        continue
    fi

    echo ""
    echo "##########################################################"
    echo "# Hopper: $RUN_NAME @ step $STEP"
    echo "##########################################################"
    RUN_NAME="$RUN_NAME" STEP="$STEP" bash "$EVAL_SCRIPT" || echo "[FAIL] $RUN_NAME @ step $STEP"
done

echo ""
echo "Done. Results in: $OUTPUT_BASE/"
