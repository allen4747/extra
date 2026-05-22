#!/usr/bin/env bash
# Run all four prefix-heuristic / Monte-Carlo analyses on Qwen3-1.7B.
#
# Outputs (so they don't clobber the existing Qwen2.5-1.5B results):
#   analysis/qwen3_outputs/2b_prefix_collected_data.pkl
#   analysis/qwen3_outputs/2b_prefix_value_data.pkl
#   analysis/qwen3_outputs/2b_prefix_metrics_data.pkl
#   analysis/qwen3_outputs/prefix_metrics_data.pkl       (consumed by entropy_signal)
#   analysis/qwen3_outputs/prefix_pass_rate_distribution.png
#   analysis/qwen3_outputs/prefix_metric_vs_passrate.png
#   analysis/qwen3_outputs/prefix_value_trajectories.png
#   analysis/qwen3_outputs/prefix_analysis_summary.json
#   analysis/qwen3_outputs/covo_collected_data_qwen3.pkl
#
# Order is light-to-heavy:
#   1. resampling          ~30 min on 1 GPU (fast smoke test)
#   2. covo                ~1 hr on 1 GPU
#   3. monte_carlo         ~2-3 hr on 1 GPU  (produces prefix_metrics_data.pkl)
#   4. entropy_signal      ~15 min, CPU-only — consumes monte_carlo output
#
# Usage:
#   conda activate verl
#   cd analysis/
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_all_qwen3.sh
#
# After each step you may inspect intermediate outputs in qwen3_outputs/.
# All scripts are resumable: if a cache .pkl already exists, that phase is
# skipped on rerun.

set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

mkdir -p qwen3_outputs

echo "================================================="
echo "  [1/4] resampling_experiment_qwen3.py"
echo "================================================="
python resampling_experiment_qwen3.py

echo "================================================="
echo "  [2/4] covo_experiment_multi_runs_qwen3.py"
echo "================================================="
python covo_experiment_multi_runs_qwen3.py

echo "================================================="
echo "  [3/4] monte_carlo_experiment_qwen3.py"
echo "================================================="
python monte_carlo_experiment_qwen3.py

echo "================================================="
echo "  [4/4] entropy_signal_analysis_qwen3.py"
echo "================================================="
python entropy_signal_analysis_qwen3.py

echo "================================================="
echo "  All Qwen3 analyses done."
echo "  Inspect: $HERE/qwen3_outputs/"
echo "================================================="
