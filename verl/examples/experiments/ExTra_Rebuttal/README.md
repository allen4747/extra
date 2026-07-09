# ExTra_Rebuttal — EMNLP rebuttal experiments

Scripts to run the two experiments the reviewers asked for, plus the eval wrapper.

## What each script does

| Script | Purpose | Reviewer concern | Wall-clock (8xH100) |
|---|---|---|---|
| `01_grpo_nano8b.sh` | GRPO baseline on `nvidia/Llama-3.1-Nemotron-Nano-8B-v1`, 150 steps | XB9Q W6, xvYm W3 (scale) | ~12–18h |
| `02_extra_full_nano8b.sh` | Full ExTra on same 8B model, 150 steps, MTE-gap logging on | Same + XB9Q W3/Q3, xvYm W2 (MTE) | ~14–20h |
| `03_grpo_qwen3_seed2.sh` | GRPO on Qwen3-1.7B, second seed | XB9Q W2/Q2, xvYm W1 (variance) | ~8–12h |
| `04_extra_full_qwen3_seed2.sh` | Full ExTra on Qwen3-1.7B, second seed, MTE-gap logging on | Same | ~8–12h |
| `05_eval_all_rebuttal.sh` | Evaluate the four checkpoints on all six benchmarks (pass@1, pass@16) | Feeds R1/R3 tables | ~4–6h |

Wall-clock ranges assume no-offload FSDP-8 sharding for the 8B runs. If you
apply RUNBOOK Fallback tier 3 (optimizer offload) the 8B time grows to
~20–24h; tier 4 (full offload) grows to ~28–36h. Plan accordingly.

## Environment

```bash
conda activate ~/my_efs/envs/verl
```

## Launch (two nodes)

Two runs in parallel — one per node, `nnodes=1, n_gpus_per_node=8` each:

```bash
# Node 1
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh |& tee node1_grpo8b.log

# Node 2 (concurrently)
bash verl/examples/experiments/ExTra_Rebuttal/02_extra_full_nano8b.sh |& tee node2_extra8b.log
```

Then after both finish (~day 3):

```bash
# Node 1
bash verl/examples/experiments/ExTra_Rebuttal/03_grpo_qwen3_seed2.sh |& tee node1_grpo_seed2.log

# Node 2
bash verl/examples/experiments/ExTra_Rebuttal/04_extra_full_qwen3_seed2.sh |& tee node2_extra_seed2.log
```

## Evaluation

Once all runs finish:

```bash
bash verl/examples/experiments/ExTra_Rebuttal/05_eval_all_rebuttal.sh
```

## Env-var overrides

Every training script honors:

- `MODEL_PATH` — override the HF hub id / local path
- `TRAIN_FILE`, `VAL_FILE` — data locations
- `EXP_NAME` — experiment name (also determines checkpoint dir)
- `CKPT_ROOT` — checkpoint root, defaults to `~/my_efs/checkpoints/ExTra_Rebuttal`
- `TOTAL_STEPS` — training steps (defaults 150 for 8B, 250 for Qwen3)
- `SEED` (Qwen3 seed-2 scripts only) — random seed, default 2024

Eval script honors: `EXTRA_REPO`, `CKPT_BASE`, `DATA_DIR`, `OUTPUT_BASE`, `EVAL_N_SAMPLES`.

## Outputs

- **Checkpoints**: `$CKPT_ROOT/<EXP_NAME>/global_step_<N>/`
- **W&B**: project `ExTra_Rebuttal`
- **Eval metrics**: `./eval_outputs_rebuttal/<RUN_NAME>/step_<N>/metrics.json`
- **MTE-gap online log** (only ExTra runs with `log_mte_gap=True`): `outputs/mte_gap_log.jsonl` under the working dir when the trainer launched.

## After training

See `RUNBOOK.md` (same folder) for day-by-day commands and analysis-script invocations.
