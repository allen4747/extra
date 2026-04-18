#!/usr/bin/env bash

# This script runs all ExTra research experiments in order.
# Each experiment is expected to take significant time.

export RAY_TMPDIR=/external1/wenyang/ray_tmp

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting ExTra Research Experiments..."




# Run two experiments in sequence.

# bash "${SCRIPT_DIR}/05_ablation_no_regeneration.sh"

# Ensure the second experiment starts after the first one finishes, to avoid GPU contention.
# Sleep for a short duration to allow resources to be released, if necessary.
# sleep 30


bash "${SCRIPT_DIR}/extra_curiosity_warmup0.sh"



echo "All experiments completed!"
