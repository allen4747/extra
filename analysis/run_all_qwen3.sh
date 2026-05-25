#!/usr/bin/env bash
# Run all Qwen3-1.7B prefix-heuristic / Monte-Carlo / resampling analyses.
#
# Outputs (written under analysis/qwen3_outputs/, which is created if missing):
#   <various>.pkl                              intermediate caches (resumable)
#   prefix_correlations_raw.json               full correlation dict (one row per metric)
#   prefix_correlations_table.json             sorted by |within_problem_corr|
#   prefix_correlations_table.csv              paste-into-paper format
#   prefix_analysis_summary.json               final summary written by mc_orig.main
#   resampling_passk_qwen3.json                pass@k for {random, raw_entropy, smoothed_entropy}
#   resampling_passk_qwen3.csv                 same in CSV form
#   resampling_passk_bars.{pdf,png}            grouped bar plot (3 bars per pass@k)
#   resampling_passk_bars.png.data.json        sidecar: arms x k_values for re-styling
#   *.png + *.png.data.json                    each plot also dumps its data
#   run_<ts>.log + run_all_<ts>.log            full stdout/stderr
#
# Order (rough cost on 1 H200; vLLM TP scales with GPU count):
#   1. resampling_passk_qwen3.py    ~30-60 min on 8 GPUs (paper bar plot)
#   2. covo_experiment_multi_runs   ~1 hr on 1 GPU
#   3. monte_carlo_experiment       ~2-3 hr (produces prefix_metrics_data.pkl)
#   4. entropy_signal_analysis      ~15 min, CPU-only (consumes monte_carlo output)
#
# Usage:
#   conda activate verl
#   pip install matplotlib scipy tqdm sentence-transformers
#   cd analysis/
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_all_qwen3.sh
#
# All scripts are resumable: if a cache .pkl already exists, that phase
# is skipped.  We do NOT use `set -e`; if one script fails, the rest
# still attempt to run and partial results are persisted.

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p qwen3_outputs

MASTER_LOG="qwen3_outputs/run_all_$(date +%Y%m%d_%H%M%S).log"
echo "[run_all] writing master log to $MASTER_LOG"
echo "[run_all] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

run_step () {
    local label="$1"; shift
    echo ""                                                  | tee -a "$MASTER_LOG"
    echo "================================================="  | tee -a "$MASTER_LOG"
    echo "  $label"                                          | tee -a "$MASTER_LOG"
    echo "================================================="  | tee -a "$MASTER_LOG"
    if "$@" 2>&1 | tee -a "$MASTER_LOG" ; then
        echo "[run_all] $label OK"                           | tee -a "$MASTER_LOG"
    else
        echo "[run_all] $label FAILED (continuing)"          | tee -a "$MASTER_LOG"
    fi
}

# Paper-critical experiments first.
run_step "[1/4] resampling_passk_qwen3.py"             python resampling_passk_qwen3.py --dataset aime24
run_step "[2/4] covo_experiment_multi_runs_qwen3.py"   python covo_experiment_multi_runs_qwen3.py
run_step "[3/4] monte_carlo_experiment_qwen3.py"       python monte_carlo_experiment_qwen3.py --n_problems 60
run_step "[4/4] entropy_signal_analysis_qwen3.py"      python entropy_signal_analysis_qwen3.py

echo ""                                                       | tee -a "$MASTER_LOG"
echo "================================================="     | tee -a "$MASTER_LOG"
echo "  All Qwen3 analyses done."                            | tee -a "$MASTER_LOG"
echo "  Inspect: $HERE/qwen3_outputs/"                       | tee -a "$MASTER_LOG"
echo "  Master log: $MASTER_LOG"                             | tee -a "$MASTER_LOG"
echo "================================================="     | tee -a "$MASTER_LOG"
