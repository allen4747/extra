#!/usr/bin/env bash
# Eval orchestrator for ExTra_Final.
# Walks every checkpoint under ExTra_Final/ at global_step_300, merges FSDP -> HF,
# then generates and grades on MATH500 (and any other benchmark configured in evals/gen_vllm.py).

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PWD/verl:$PYTHONPATH

BASE_DIR="/external1/wenyang/checkpoints/ExTra_Final"

if [ ! -d "$BASE_DIR" ]; then
    echo "Checkpoint dir not found: $BASE_DIR. Have any of the 01-12 runs finished?"
    exit 1
fi

echo "Scanning $BASE_DIR for global_step_300 checkpoints..."

while IFS= read -r ckpt_dir; do
    if [ ! -d "$ckpt_dir/actor" ]; then
        continue
    fi
    EXP_NAME=$(basename $(dirname "$ckpt_dir"))
    echo "=========================================================="
    echo "Eval: $EXP_NAME"
    echo "Checkpoint: $ckpt_dir"
    echo "=========================================================="

    HF_MODEL_DIR="$ckpt_dir/hf_model"
    if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
        echo "Merging FSDP shards into HuggingFace format..."
        python3 verl/scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir "$ckpt_dir/actor" \
            --hf_model_path "$ckpt_dir/actor/huggingface" \
            --target_dir "$HF_MODEL_DIR"
    else
        echo "Merged HF model already exists at $HF_MODEL_DIR"
    fi

    OUTPUT_DIR="./eval_outputs_final/${EXP_NAME}_step_300"
    mkdir -p "$OUTPUT_DIR"
    echo "Generating responses -> $OUTPUT_DIR"

    python3 /home/wenyang/ExTra/evals/gen_vllm.py \
        --model "$HF_MODEL_DIR" \
        --out_dir "$OUTPUT_DIR"

    echo "Grading responses..."
    python3 /home/wenyang/ExTra/evals/grade.py \
        --eval_dir "$OUTPUT_DIR"

    echo "Done: $EXP_NAME"
    echo "----------------------------------------------------------"
    echo ""
done < <(find "$BASE_DIR" -type d -name "global_step_300" | sort)

echo "All ExTra_Final evals complete."
