#!/usr/bin/env bash
# AWS server (ip-172-31-95-50): 3e-6 runs at step 250.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval_extra_qwen_batch_aws.sh

set -e

EXTRA_REPO="${EXTRA_REPO:-$HOME/ExTra}"
EVAL_SCRIPT="${EXTRA_REPO}/evals/eval_extra_qwen.sh"

export CKPT_BASE="${CKPT_BASE:-/home/wenyang/my_efs/checkpoints/ExTra_Qwen}"
export DATA_DIR="${DATA_DIR:-/home/wenyang/my_efs/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-./eval_outputs}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-32}"

RUNS=(
    "01b_GRPO_Baseline_3e6_Qwen3:250"
    "02_ExTra_RegenOnly_Qwen3:250"
    "05_ExTra_Full_OptionB_Qwen3_3e6_0.5_aws:250"
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
    echo "# AWS: $RUN_NAME @ step $STEP"
    echo "##########################################################"
    RUN_NAME="$RUN_NAME" STEP="$STEP" bash "$EVAL_SCRIPT" || echo "[FAIL] $RUN_NAME @ step $STEP"
done

echo ""
echo "Done. Results in: $OUTPUT_BASE/"
