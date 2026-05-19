#!/usr/bin/env bash
# Evaluate a single ExTra_Qwen checkpoint across all benchmarks.
# Outputs:
#   - eval_outputs/{RUN_NAME}/step_{STEP}/{benchmark}_t1.0_p1.0_n32-MNT8192.jsonl  (raw responses)
#   - eval_outputs/{RUN_NAME}/step_{STEP}/grading_results.json                     (graded scores)
#   - eval_outputs/{RUN_NAME}/step_{STEP}/metrics.json                             (paper-ready summary)
#
# Usage:
#   RUN_NAME=06_ExTra_RegenOnly_1e6_Qwen3 \
#   STEP=200 \
#   CKPT_BASE=/home/wenyang/my_efs/checkpoints/ExTra_Qwen \
#   DATA_DIR=/home/wenyang/my_efs/datasets \
#   bash eval_extra_qwen.sh

set -e

# --- Required env vars ---
RUN_NAME="${RUN_NAME:?ERROR: RUN_NAME must be set (e.g. 06_ExTra_RegenOnly_1e6_Qwen3)}"
STEP="${STEP:?ERROR: STEP must be set (e.g. 200)}"
CKPT_BASE="${CKPT_BASE:?ERROR: CKPT_BASE must be set (e.g. /home/wenyang/my_efs/checkpoints/ExTra_Qwen)}"

# --- Optional env vars ---
DATA_DIR="${DATA_DIR:-/home/wenyang/my_efs/datasets}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-16}"
OUTPUT_BASE="${OUTPUT_BASE:-./eval_outputs_v2}"
EXTRA_REPO="${EXTRA_REPO:-$HOME/ExTra}"

export DATA_DIR EVAL_N_SAMPLES
export PYTHONPATH="${EXTRA_REPO}/verl:${PYTHONPATH:-}"

CKPT_DIR="${CKPT_BASE}/${RUN_NAME}/global_step_${STEP}"
HF_MODEL_DIR="${CKPT_DIR}/hf_model"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}/step_${STEP}"

mkdir -p "$OUTPUT_DIR"

# Resolve OUTPUT_DIR to absolute path so subsequent cd doesn't break it
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "=========================================================="
echo "  Evaluating: $RUN_NAME @ step $STEP"
echo "  Checkpoint: $CKPT_DIR"
echo "  Data dir:   $DATA_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  N samples:  $EVAL_N_SAMPLES"
echo "=========================================================="

# 1. Merge FSDP -> HF format if needed
if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
    echo "[1/3] Merging FSDP shards to HuggingFace format..."
    PYTHONPATH="${EXTRA_REPO}/verl:${PYTHONPATH:-}" \
        python3 "${EXTRA_REPO}/verl/scripts/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${CKPT_DIR}/actor" \
            --hf_model_path "${CKPT_DIR}/actor/huggingface" \
            --target_dir "$HF_MODEL_DIR"
else
    echo "[1/3] Merged HF model already exists at $HF_MODEL_DIR"
fi

# 2. Generate responses with vLLM
echo "[2/3] Generating responses (n=$EVAL_N_SAMPLES per problem)..."
python3 "${EXTRA_REPO}/evals/gen_vllm.py" \
    --model "$HF_MODEL_DIR" \
    --out_dir "$OUTPUT_DIR"

# 3. Grade responses
echo "[3/3] Grading responses..."
cd "${EXTRA_REPO}/evals"
python3 grade.py --eval_dir "$OUTPUT_DIR"

# 4. Build paper-ready metrics.json from grading_results.json
echo "[+] Building metrics.json summary..."
python3 - <<EOF
import json
from pathlib import Path

eval_dir = Path("$OUTPUT_DIR")
grading = json.loads((eval_dir / "grading_results.json").read_text())

summary = {
    "run_name": "$RUN_NAME",
    "step": int("$STEP"),
    "n_samples": int("$EVAL_N_SAMPLES"),
    "benchmarks": {},
}

for entry in grading:
    task = entry["hyperparameters"]["task_name"]
    summary["benchmarks"][task] = {
        "avg@n": round(entry["mean_score"], 4),
        "best@n": round(entry["best_score"], 4),
        "distinct_4gram": round(entry["distinct_4gram"], 4),
        "solve_none": entry["solve_none"],
        "solve_all": entry["solve_all"],
        "avg_output_length": round(entry["avg_output_length"], 1),
        "format_errors": entry["format_error_rollouts"],
    }

(eval_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
print(f"Wrote {eval_dir / 'metrics.json'}")
print(json.dumps(summary, indent=2))
EOF

echo "=========================================================="
echo "  Done: $RUN_NAME @ step $STEP"
echo "  Summary: $OUTPUT_DIR/metrics.json"
echo "=========================================================="
