#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PWD/verl:$PYTHONPATH

# This script automatically finds all saved checkpoints across your 1.5B and 7B 
# experiment directories, merges the FSDP weights to HF format, and runs the evaluation.

BASE_DIRS=(
    # "/external1/wenyang/checkpoints/ExTra_Research"
    "/home/wenyang/my_efs/checkpoints/ExTra_Research"
)

echo "Scanning for checkpoints to evaluate..."

for BASE_DIR in "${BASE_DIRS[@]}"; do
    if [ -d "$BASE_DIR" ]; then
        while IFS= read -r ckpt_dir; do
            if [ -d "$ckpt_dir/actor" ]; then
                echo "=========================================================="
                echo "Found checkpoint: $ckpt_dir"
                echo "=========================================================="
                
                # 1. Merge FSDP weights to Hugging Face format
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
                
                # 2. Evaluate the merged model
                # Extract the model name for a unique output directory (e.g. GRPO-JustRL-Nemotron-1.5B)
                MODEL_EXP_NAME=$(basename $(dirname "$ckpt_dir"))
                export OUTPUT_DIR="./eval_outputs/${MODEL_EXP_NAME}_step_300"
                
                echo "Starting evaluation for $MODEL_EXP_NAME (saving to $OUTPUT_DIR)"
                
                # Generate responses
                python3 /home/wenyang/ExTra/evals/gen_vllm.py \
                    --model "$HF_MODEL_DIR" \
                    --out_dir "$OUTPUT_DIR"
                
                # Grade the responses
                python3 /home/wenyang/ExTra/evals/grade.py \
                    --eval_dir "$OUTPUT_DIR"
                
                echo "Finished evaluation for $MODEL_EXP_NAME"
                echo "----------------------------------------------------------"
                echo ""
            fi
        done < <(find "$BASE_DIR" -type d -name "global_step_300" | sort)
    else
        echo "Directory not found or not yet created: $BASE_DIR (skipping)"
    fi
done

echo "All evaluations completed!"
