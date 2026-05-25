#!/usr/bin/env bash
# Qwen3-1.7B paper analyses.  Two experiments only:
#
#   (1) Proxy correlation study  (monte_carlo + entropy_signal)
#       - Spearman correlations between many candidate prefix metrics
#         and the Monte-Carlo prefix pass-rate.
#       - Mixed pool: MATH-500 (level 5) + AMC23 + AIME24 + AIME25.
#       - Outputs: qwen3_outputs/prefix_correlations_table.{json,csv}
#                  qwen3_outputs/2b_prefix_metrics_data.pkl   (cache)
#                  qwen3_outputs/*.png                        (plots)
#
#   (2) Pass@k resampling experiment
#       - 3 arms (random, raw MTE, smoothed MTE) x k in {1, 8, 16}
#         on AIME24.
#       - Outputs: qwen3_outputs/resampling_passk_qwen3.{json,csv}
#                  qwen3_outputs/resampling_passk_bars.{pdf,png}
#
# Cost (1x 8-GPU H200 host, vLLM TP=8):
#   - resampling_passk    ~30-60 min
#   - monte_carlo         ~2-3 hr  (only step that uses the .pkl cache)
#   - entropy_signal      ~15 min, CPU-only (consumes monte_carlo cache)
#
# Usage:
#   conda activate ~/my_efs/envs/verl
#   pip install matplotlib scipy tqdm sentence-transformers
#   cd analysis/
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_all_qwen3.sh
#
# Resumable: monte_carlo skips if 2b_prefix_metrics_data.pkl exists.
# Delete that file to force a fresh run.  resampling_passk and
# entropy_signal always rerun (cheap or no cache).
# We do NOT use `set -e`; if one script fails, the rest still attempt
# to run and partial results are persisted.

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

# (A) Pass@k resampling bar plot (paper figure).
run_step "[1/3] resampling_passk_qwen3.py"        python resampling_passk_qwen3.py --dataset aime24

# (B) Proxy correlation study: heavy MC sampling first, then CPU-only analysis.
run_step "[2/3] monte_carlo_experiment_qwen3.py"  python monte_carlo_experiment_qwen3.py --n_problems 60
run_step "[3/3] entropy_signal_analysis_qwen3.py" python entropy_signal_analysis_qwen3.py

echo ""                                                       | tee -a "$MASTER_LOG"
echo "================================================="     | tee -a "$MASTER_LOG"
echo "  All Qwen3 analyses done."                            | tee -a "$MASTER_LOG"
echo "  Inspect: $HERE/qwen3_outputs/"                       | tee -a "$MASTER_LOG"
echo "  Master log: $MASTER_LOG"                             | tee -a "$MASTER_LOG"
echo "================================================="     | tee -a "$MASTER_LOG"
