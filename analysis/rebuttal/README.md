# analysis/rebuttal/ — EMNLP rebuttal analyses

Post-training scripts that turn training/eval artifacts into the numbers the
rebuttal draft (`~/Downloads/ARR_May___ExTra/REBUTTAL_PLAN.md`) needs.

## Which script feeds which rebuttal paragraph

| Script | Output artifact | Rebuttal section | Reviewer concern |
|---|---|---|---|
| `bootstrap_ci.py` | `paper_table_ci.csv`, `paper_table_pvals.csv` | R1 / Tables R1–R2 | XB9Q W2/Q2, xvYm W1 (variance, CIs) |
| `mte_gap_summary.py` | `mte_gap_summary.md` | R2 online MTE gap + XB9Q Q4 stale-prefix | XB9Q W3/Q3, xvYm W2, XB9Q Q4 |
| `decontam_ngram.py` | `decontam_table.csv` | XB9Q Q6 answer paragraph | XB9Q Q6 (decontamination) |
| `reward_hacking_check.py` | `reward_hack.md` | XB9Q Q7 answer paragraph | XB9Q Q7 (reward hacking) |
| `wallclock_breakdown.py` | `wallclock.csv` | R§5.5 wall-clock paragraph | XB9Q W5/Q5 (efficiency) |

## How to run each (once training + eval are done)

```bash
# A1: bootstrap CIs + paired p-values
python analysis/rebuttal/bootstrap_ci.py \
  --eval_dir ./eval_outputs_rebuttal \
  --n_bootstrap 10000 \
  --out paper_table_ci.csv \
  --ref_run 01_GRPO_NanoNemotron_8B \
  --cmp_run 02_ExTra_Full_NanoNemotron_8B \
  --step 150 \
  --out_pvals paper_table_pvals_8b.csv

# A2: online MTE-gap summary
python analysis/rebuttal/mte_gap_summary.py \
  --log outputs/mte_gap_log.jsonl \
  --out mte_gap_summary_8b.md

# A3: wall-clock breakdown (prefers W&B, falls back to logs)
python analysis/rebuttal/wallclock_breakdown.py \
  --wandb_project ExTra_Rebuttal \
  --pairs 01_GRPO_NanoNemotron_8B,02_ExTra_Full_NanoNemotron_8B \
  --pairs 03_GRPO_Qwen3_seed2,04_ExTra_Full_Qwen3_seed2 \
  --out wallclock.csv

# A4: decontamination
python analysis/rebuttal/decontam_ngram.py \
  --train_file $HOME/data/math_dapo/train.parquet \
  --eval_dir  $HOME/my_efs/datasets \
  --n 13 \
  --out decontam_table.csv

# A5: reward-hacking sanity check
python analysis/rebuttal/reward_hacking_check.py \
  --eval_dir ./eval_outputs_rebuttal \
  --ref_run 01_GRPO_NanoNemotron_8B \
  --cmp_run 02_ExTra_Full_NanoNemotron_8B \
  --step 150 \
  --out reward_hack_8b.md
```

## Filling in the rebuttal placeholders

The rebuttal draft (Section 5 of `REBUTTAL_PLAN.md`) has a checklist of
bracketed placeholders like `[±X.X]`, `[p=…]`, `[A%]`, etc. Each maps to a
column in one of the CSV/MD outputs above. Concretely:

| Placeholder in rebuttal | Source file | Column / line |
|---|---|---|
| AIME24 pass@1 CI half-width | `paper_table_ci.csv` | `pass@1_ci_hi` − `pass@1_ci_lo`, row `benchmark=AIME24` |
| Paired-bootstrap p-value | `paper_table_pvals_8b.csv` | `pass@1_p_paired_boot`, `pass@k_p_paired_boot` |
| Online MTE gap A/B/C pp | `mte_gap_summary_8b.md` | `mean entropy gap` under `All steps` |
| Late-training gap D pp | `mte_gap_summary_8b.md` | `mean entropy gap` under `Late training` |
| Nemotron gap E pp | Same script run on Nemotron log (see runbook) | – |
| Wall-clock [G]/[E]/[P%] | `wallclock.csv` | `grpo_median_step_s`, `extra_median_step_s`, `overhead_frac` |
| Decontamination overlap | `decontam_table.csv` | `problems_with_any_overlap_frac` |
| Reward-hacking numbers | `reward_hack_8b.md` | Rendered table |
| Queue median age / depth | `mte_gap_summary_8b.md` | `queue_size_before: median …, 90th %ile …` and `prefix_memory_size` |
| Table R3 8B numbers | `rebuttal_table_combined.csv` (from `evals/aggregate_eval_results.py`) | `AIME24_pass@1`, `AIME24_pass@k`, … |
| Table R2 seed-2 numbers | Same aggregated CSV | Rows for `03_GRPO_Qwen3_seed2` and `04_ExTra_Full_Qwen3_seed2` |

## Reused utilities (imports)

- `verl.trainer.ppo.metric_utils.bootstrap_metric` — available in the repo;
  `bootstrap_ci.py` uses a simpler problem-level bootstrap to preserve the
  paired structure the reviewers care about.
- `evals/grade.py` — same boxed-answer extractor used by
  `reward_hacking_check.py` (via a local `BOXED_RE`).
- `evals/gen_vllm.py:load_samples` — same parquet schema used by
  `decontam_ngram.py:load_problem_texts`.

Do **not** re-run the offline MTE Monte Carlo pipeline in `analysis/monte_carlo_experiment*.py` — the offline evidence is already in the paper; the rebuttal cites those results and adds the *online* gap only.
