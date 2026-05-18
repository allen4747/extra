#!/usr/bin/env bash
# Batch evaluation: runs eval_extra_qwen.sh for a list of (RUN_NAME, STEP) pairs.
# Edit the RUNS array below to match the checkpoints available on this server.
#
# Usage:
#   CKPT_BASE=/home/wenyang/my_efs/checkpoints/ExTra_Qwen \
#   DATA_DIR=/home/wenyang/my_efs/datasets \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash eval_extra_qwen_batch.sh

set -e

EXTRA_REPO="${EXTRA_REPO:-$HOME/ExTra}"
EVAL_SCRIPT="${EXTRA_REPO}/evals/eval_extra_qwen.sh"

# --- Edit this list per server: only include checkpoints that exist locally ---
# Format: "RUN_NAME:STEP"
RUNS=(
    "01_GRPO_Baseline_Qwen3:200"
    "01_GRPO_Baseline_Qwen3:300"
    "01b_GRPO_Baseline_3e6_Qwen3:100"
    "01b_GRPO_Baseline_3e6_Qwen3:200"
    "02_ExTra_RegenOnly_Qwen3:200"
    "02_ExTra_RegenOnly_Qwen3:250"
    "03_ExTra_Full_Qwen3_0.01_hlr:100"
    "03_ExTra_Full_Qwen3_0.01_hlr:200"
    "03_ExTra_Full_Qwen3_0.1_hlr:150"
    "03_ExTra_Full_Qwen3_0.1_hlr:170"
    "06_ExTra_RegenOnly_1e6_Qwen3:200"
    "06_ExTra_RegenOnly_1e6_Qwen3:300"
    "07_ExTra_Full_OptionB_1e6_Qwen3:200"
    "07_ExTra_Full_OptionB_1e6_Qwen3:300"
    "08_ExTra_Full_OptionB_3e6_Qwen3:150"
    "08_ExTra_Full_OptionB_3e6_Qwen3:200"
)

CKPT_BASE="${CKPT_BASE:?ERROR: CKPT_BASE must be set}"

for run_step in "${RUNS[@]}"; do
    RUN_NAME="${run_step%:*}"
    STEP="${run_step#*:}"
    CKPT_DIR="${CKPT_BASE}/${RUN_NAME}/global_step_${STEP}"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[SKIP] $RUN_NAME @ step $STEP: checkpoint not found at $CKPT_DIR"
        continue
    fi

    METRICS_FILE="${OUTPUT_BASE:-./eval_outputs}/${RUN_NAME}/step_${STEP}/metrics.json"
    if [ -f "$METRICS_FILE" ]; then
        echo "[SKIP] $RUN_NAME @ step $STEP: metrics already exist at $METRICS_FILE"
        continue
    fi

    echo ""
    echo "##########################################################"
    echo "# Evaluating $RUN_NAME @ step $STEP"
    echo "##########################################################"

    RUN_NAME="$RUN_NAME" STEP="$STEP" \
        bash "$EVAL_SCRIPT" || echo "[FAIL] $RUN_NAME @ step $STEP — continuing"
done

echo ""
echo "All evaluations done. Sync metrics:"
echo "  rsync -av eval_outputs/ user@aggregator:~/extra_eval_results/"
