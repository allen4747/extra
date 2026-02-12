#!/bin/bash

# Function to check GPU availability
check_gpu() {
    for i in {0..7}
    do
        used_memory=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ $used_memory -lt 2000 ]; then
            echo $i
            return 0
        fi
    done
    return 1
}

# Wait for a free GPU
while true; do
    free_gpu=$(check_gpu)
    if [ $? -eq 0 ]; then
        echo "There is a free GPU $free_gpu"
        #### script ####
        export CUDA_VISIBLE_DEVICES=$free_gpu
        python adap_alloc.py
        break
    else
        echo "No free GPU available. Waiting for 90 seconds..."
        sleep 90
    fi
done