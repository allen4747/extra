#!/usr/bin/env bash
# EMNLP rebuttal — evaluate all rebuttal checkpoints on the six eval benchmarks.
#
# Mirrors evals/eval_batch_aws.sh style. Iterates over the four rebuttal runs
# and evaluates their final checkpoints (150 for 8B, 250 for Qwen3-seed2) with
# n=16 samples per problem via vLLM.

set -e

EXTRA_REPO="${EXTRA_REPO:-$HOME/ExTra}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${EXTRA_REPO}/evals/eval_extra_qwen.sh}"

export CKPT_BASE="${CKPT_BASE:-/home/wenyang/my_efs/checkpoints/ExTra_Rebuttal}"
export DATA_DIR="${DATA_DIR:-/home/wenyang/my_efs/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-./eval_outputs_rebuttal}"
export EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-16}"

# run_name : final_step
RUNS=(
    "01_GRPO_NanoNemotron_8B:150"
    "02_ExTra_Full_NanoNemotron_8B:150"
    "03_GRPO_Qwen3_seed2:250"
    "04_ExTra_Full_Qwen3_seed2:250"
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
        echo "[SKIP] $RUN_NAME @ step $STEP: metrics already exist at $METRICS_FILE"
        continue
    fi

    echo ""
    echo "##########################################################"
    echo "# Rebuttal eval: $RUN_NAME @ step $STEP"
    echo "##########################################################"
    RUN_NAME="$RUN_NAME" STEP="$STEP" bash "$EVAL_SCRIPT" \
        || echo "[FAIL] $RUN_NAME @ step $STEP"
done

echo ""
echo "Aggregating..."
python "${EXTRA_REPO}/evals/aggregate_eval_results.py" \
    --results_dir "$OUTPUT_BASE" \
    --output rebuttal_table

echo ""
echo "Done. Per-run metrics in: $OUTPUT_BASE/"
echo "Aggregated CSV: rebuttal_table_combined.csv"
