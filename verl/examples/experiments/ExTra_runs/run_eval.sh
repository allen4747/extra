#!/usr/bin/env bash
# Run pass@k + diversity evaluation for GRPO baseline and ExTra-Full
# on GPU 2 sequentially (each uses ~6GB vLLM + model)

set -e
export CUDA_VISIBLE_DEVICES=2
export RAY_TMPDIR=/external1/wenyang/ray_tmp

EVAL_SCRIPT="$(dirname "$0")/eval_passk_diversity.py"
OUTPUT_DIR="/home/wenyang/ExTra/analysis/eval_results"
N_SAMPLES=16

echo "============================================"
echo "  ExTra Evaluation: pass@k + diversity"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  N_SAMPLES: $N_SAMPLES"
echo "============================================"

# 1. GRPO Baseline at step 200
echo ""
echo ">>> [1/2] GRPO-JustRL step 200"
python3 "$EVAL_SCRIPT" \
    --checkpoint /external1/wenyang/checkpoints/ExTra_Research/GRPO-JustRL-Qwen2.5-1.5B/global_step_200/actor \
    --name GRPO-JustRL-step200 \
    --n_samples $N_SAMPLES \
    --output_dir "$OUTPUT_DIR"

# 2. ExTra-Full at step 200
echo ""
echo ">>> [2/2] ExTra-Full-JustRL step 200"
python3 "$EVAL_SCRIPT" \
    --checkpoint /external1/wenyang/checkpoints/ExTra_Research/ExTra-JustRL-Qwen2.5-1.5B/global_step_200/actor \
    --name ExTra-Full-JustRL-step200 \
    --n_samples $N_SAMPLES \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "  Results in: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "Summary:"
for f in "$OUTPUT_DIR"/*.json; do
    if [[ "$f" != *"_responses"* ]]; then
        echo "--- $(basename "$f") ---"
        python3 -c "import json; d=json.load(open('$f')); [print(f'  {k}: {v}') for k,v in d.items() if 'response' not in str(k)]"
    fi
done
