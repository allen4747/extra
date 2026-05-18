#!/usr/bin/env bash
# Helper: wait for N free GPUs (each <1GB used), then export CUDA_VISIBLE_DEVICES.
#
# Source this from a batch eval script:
#   NUM_GPUS=4 source "$(dirname "$0")/wait_for_gpus.sh"
#
# Env vars:
#   NUM_GPUS         - required, number of free GPUs to wait for
#   GPU_MEM_THRESHOLD - optional, MB threshold to consider free (default 1000)
#   POLL_INTERVAL    - optional, seconds between polls (default 30)

NUM_GPUS="${NUM_GPUS:?ERROR: NUM_GPUS must be set (e.g. NUM_GPUS=4)}"
GPU_MEM_THRESHOLD="${GPU_MEM_THRESHOLD:-1000}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

echo "Waiting for $NUM_GPUS GPUs (memory used < ${GPU_MEM_THRESHOLD} MB)..."

while true; do
    # Get list of free GPU indices
    free_gpus=()
    while IFS= read -r line; do
        idx=$(echo "$line" | awk -F',' '{print $1}' | tr -d ' ')
        used=$(echo "$line" | awk -F',' '{print $2}' | tr -d ' ')
        if [ -n "$idx" ] && [ -n "$used" ] && [ "$used" -lt "$GPU_MEM_THRESHOLD" ]; then
            free_gpus+=("$idx")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)

    n_free=${#free_gpus[@]}
    echo "  $(date +%H:%M:%S) — found $n_free free GPU(s): ${free_gpus[*]:-none}"

    if [ "$n_free" -ge "$NUM_GPUS" ]; then
        # Take first NUM_GPUS free GPUs
        selected=("${free_gpus[@]:0:$NUM_GPUS}")
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected[*]}")
        echo "  Selected GPUs: $CUDA_VISIBLE_DEVICES"
        break
    fi

    sleep "$POLL_INTERVAL"
done
