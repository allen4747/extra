#!/usr/bin/env bash
# Eval orchestrator: evaluate ExTra_Final checkpoints at global_step_150 and global_step_200.
# Modeled on 13_run_all_evals.sh but with:
#   - Multiple steps (150, 200)
#   - Env-overridable BASE_DIR, DATA_DIR, EVAL_WORKERS
#   - Idempotent: skips merge/gen/grade if grading_results.json and >=4 jsonls exist
#   - Appends per-task results to eval_outputs_final/summary_$(hostname -s).csv

set -euo pipefail

export PYTHONPATH="${PWD}/verl:${PYTHONPATH:-}"

BASE_DIR="${BASE_DIR:-/home/wenyang/my_efs/checkpoints/ExTra_Final}"
export DATA_DIR="${DATA_DIR:-$HOME/JustRL/data}"

# Set CUDA_VISIBLE_DEVICES based on EVAL_WORKERS (use GPU indices 0..N-1)
export CUDA_VISIBLE_DEVICES=4,5,6,7

STEPS=("global_step_150" "global_step_200")
SUMMARY_CSV="eval_outputs_final/summary_$(hostname -s).csv"
mkdir -p eval_outputs_final

# Write CSV header if file doesn't exist
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "experiment,step,task,mean_score,best_score,distinct_4gram,solve_none,solve_all,avg_output_length,format_error_rollouts" > "$SUMMARY_CSV"
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Checkpoint dir not found: $BASE_DIR"
    exit 1
fi

for STEP in "${STEPS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Scanning $BASE_DIR for $STEP checkpoints..."
    echo "============================================================"

    while IFS= read -r ckpt_dir; do
        if [ ! -d "$ckpt_dir/actor" ]; then
            continue
        fi
        EXP_NAME=$(basename "$(dirname "$ckpt_dir")")
        STEP_NUM="${STEP#global_step_}"
        OUTPUT_DIR="./eval_outputs_final/${EXP_NAME}_step_${STEP_NUM}"
        mkdir -p "$OUTPUT_DIR"

        echo ""
        echo "=========================================================="
        echo "Eval: $EXP_NAME @ $STEP"
        echo "Checkpoint: $ckpt_dir"
        echo "Output: $OUTPUT_DIR"
        echo "=========================================================="

        # --- Idempotency check ---
        JSONL_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.jsonl" 2>/dev/null | wc -l)
        if [ -f "$OUTPUT_DIR/grading_results.json" ] && [ "$JSONL_COUNT" -ge 4 ]; then
            echo "SKIP: grading_results.json and ${JSONL_COUNT} jsonls already exist."
        else
            # --- Merge FSDP -> HF ---
            HF_MODEL_DIR="$ckpt_dir/hf_model"
            if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
                echo "Merging FSDP shards into HuggingFace format..."
                python3 scripts/legacy_model_merger.py merge \
                    --backend fsdp \
                    --local_dir "$ckpt_dir/actor" \
                    --hf_model_path "$ckpt_dir/actor/huggingface" \
                    --target_dir "$HF_MODEL_DIR"
            else
                echo "Merged HF model already exists at $HF_MODEL_DIR"
            fi

            # --- Generate ---
            echo "Generating responses -> $OUTPUT_DIR"
            python3 /home/wenyang/ExTra/evals/gen_vllm.py \
                --model "$HF_MODEL_DIR" \
                --out_dir "$OUTPUT_DIR"

            # --- Grade ---
            echo "Grading responses..."
            python3 /home/wenyang/ExTra/evals/grade.py \
                --eval_dir "$OUTPUT_DIR"
        fi

        # --- Append to summary CSV ---
        if [ -f "$OUTPUT_DIR/grading_results.json" ]; then
            python3 -c "
import json, sys
with open('$OUTPUT_DIR/grading_results.json') as f:
    results = json.load(f)
for r in results:
    hp = r.get('hyperparameters', {})
    task = hp.get('task_name', 'unknown')
    print(','.join([
        '$EXP_NAME', '$STEP_NUM', task,
        str(r.get('mean_score', '')),
        str(r.get('best_score', '')),
        str(r.get('distinct_4gram', '')),
        str(r.get('solve_none', '')),
        str(r.get('solve_all', '')),
        str(r.get('avg_output_length', '')),
        str(r.get('format_error_rollouts', '')),
    ]))
" >> "$SUMMARY_CSV"
        fi

        echo "Done: $EXP_NAME @ $STEP"
        echo "----------------------------------------------------------"

    done < <(find "$BASE_DIR" -type d -name "$STEP" | sort)
done

echo ""
echo "============================================================"
echo "All evals complete. Summary at: $SUMMARY_CSV"
echo "============================================================"
