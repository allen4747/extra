#!/usr/bin/env bash
# Sequential orchestrator for all main ExTra paper experiments.
# Runs experiments one after another on the same set of GPUs.
#
# Usage:
#   bash run_all_experiments.sh
#
# To run only specific experiments, comment out the ones you don't need.
# For automated GPU-aware scheduling, use gpu_monitor.py instead.
#
# Optional env vars:
#   CUDA_VISIBLE_DEVICES  (default: 0,1,2,3)
#   LOG_DIR               (default: $HOME/ExTra_logs)

set -e

# CUDA_VISIBLE_DEVICES is inherited from gpu_monitor.py (or the caller).
# The two assigned GPU indices are passed in; individual scripts will inherit them.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$HOME/ExTra_logs}"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_exp() {
    local name="$1"
    local script="$2"
    echo ""
    echo "========================================"
    echo "$(timestamp)  START: $name"
    echo "========================================"
    bash "$script" 2>&1 | tee "$LOG_DIR/${name}.log"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "$(timestamp)  WARNING: $name exited with code $exit_code"
    else
        echo "$(timestamp)  DONE: $name"
    fi
    echo ""
    return $exit_code
}

echo "ExTra Research Experiments"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Logs: $LOG_DIR"
echo ""

# ── Main comparison experiments ──────────────────────────────────────────────
run_exp "01_grpo_baseline"         "$SCRIPT_DIR/01_grpo_baseline.sh"
run_exp "02_extra_full"            "$SCRIPT_DIR/02_extra_full.sh"
run_exp "03_grpo_entropy"          "$SCRIPT_DIR/03_grpo_entropy.sh"
run_exp "04_ablation_no_curiosity" "$SCRIPT_DIR/04_ablation_no_curiosity_warmup50.sh"
run_exp "05_ablation_no_regen"     "$SCRIPT_DIR/05_ablation_no_regeneration.sh"

# ── Warmup timing ablation (uses scripts in parent experiments/ dir) ──────────
EXPERIMENTS_DIR="$(dirname "$SCRIPT_DIR")"
run_exp "warmup_00" "$EXPERIMENTS_DIR/02_extra_warmup_0.sh"
run_exp "warmup_20" "$EXPERIMENTS_DIR/03_extra_warmup_20.sh"
run_exp "warmup_50" "$EXPERIMENTS_DIR/04_extra_warmup_50.sh"
run_exp "warmup_200" "$EXPERIMENTS_DIR/05_extra_warmup_200.sh"

echo "========================================"
echo "$(timestamp)  All experiments complete."
echo "Logs saved to: $LOG_DIR"
echo "========================================"
