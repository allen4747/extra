# EMNLP Rebuttal RUNBOOK

Copy-pasteable commands to run all rebuttal experiments and analyses in a
5-day window using 2 nodes × 8 H100 = 16 H100 total.

Deliverable at the end: a filled-in rebuttal draft in
`~/Downloads/ARR_May___ExTra/REBUTTAL_PLAN.md`, with every bracketed
placeholder replaced by a real number from the CSV/MD outputs below.

## Assumptions

- Two nodes, each with 8 × H100.
- Repo path on remote: `~/ExTra` (adjust `EXTRA_REPO` if different).
- Conda env: `~/my_efs/envs/verl` (matches `CLAUDE.md`).
- Weights & Biases logged in (used by wallclock analysis; not required for training).

## Day 0 — setup (both nodes)

```bash
cd ~/ExTra
git fetch origin && git checkout main && git pull
conda activate ~/my_efs/envs/verl

# Cache the 8B model on both nodes (skips download at first training step)
huggingface-cli download nvidia/Llama-3.1-Nemotron-Nano-8B-v1
# Qwen3-1.7B should already be cached; if not:
huggingface-cli download Qwen/Qwen3-1.7B

# Sanity-check the env can import the patched trainer
python -c "import verl.verl.trainer.ppo.ray_trainer as _; print('OK')"
```

**Pre-flight memory check** — the 8B scripts assume ~65 GB peak/GPU on 80 GB
H100 (no offload, FSDP-8 sharding, TP=2 for vLLM, GMU=0.80). If your H100s
are the 40 GB SXM variant or you have other processes resident, jump
straight to "Fallback tier 3" in the fallback section before launching.

**On the dev laptop (analyses that need no GPU):** you can already run
`decontam_ngram.py` if you have the parquets locally — see Day 3.

## Day 1 — launch 8B runs (two parallel, one per node)

```bash
# NODE 1
cd ~/ExTra
conda activate ~/my_efs/envs/verl
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  |& tee -a node1_grpo8b.log

# NODE 2 (in parallel)
cd ~/ExTra
conda activate ~/my_efs/envs/verl
bash verl/examples/experiments/ExTra_Rebuttal/02_extra_full_nano8b.sh \
  |& tee -a node2_extra8b.log
```

**Check-in after ~1h:**
- `wandb` dashboard should show both runs advancing past step 5.
- Node 2 (ExTra): expect `outputs/mte_gap_log.jsonl` to appear once
  training passes `warmup_steps=30`. Confirm:
  ```bash
  ls -la outputs/mte_gap_log.jsonl && wc -l outputs/mte_gap_log.jsonl
  ```
  (Empty until step 31, then grows steadily.)

**If Day-1 fails with OOM on 8B** — see "Fallbacks" below.

## Day 2 — 8B runs continue

Nothing to launch. Monitor:

```bash
# From either node, tail the training log
tail -f node1_grpo8b.log
tail -f node2_extra8b.log

# Quick pass rate snapshot mid-training
grep -E "val/.*/pass|val_step_reward" node2_extra8b.log | tail -20
```

## Day 3 — eval the 8B runs; launch Qwen3 seed-2 pair

The 8B runs should finish around Day 2 → Day 3 boundary.

```bash
# On either node
cd ~/ExTra
conda activate ~/my_efs/envs/verl
bash verl/examples/experiments/ExTra_Rebuttal/05_eval_all_rebuttal.sh
```

This walks over each `EXP_NAME/global_step_150` and generates `n=16` samples
across all six benchmarks, then grades and aggregates. Wall-clock ~4–6h on
a single 8 × H100 node (uses vLLM's tensor parallelism).

**Meanwhile, launch the seed-2 pair on the other node:**

```bash
# NODE 1 (whichever is free after eval)
bash verl/examples/experiments/ExTra_Rebuttal/03_grpo_qwen3_seed2.sh \
  |& tee -a node1_grpo_seed2.log

# NODE 2 (in parallel, once its 8B eval is done)
bash verl/examples/experiments/ExTra_Rebuttal/04_extra_full_qwen3_seed2.sh \
  |& tee -a node2_extra_seed2.log
```

**Run in parallel on your laptop / dev machine (no GPU needed):**

```bash
# Decontamination check — needs only parquet files locally
python analysis/rebuttal/decontam_ngram.py \
  --train_file $HOME/data/math_dapo/train.parquet \
  --eval_dir  $HOME/my_efs/datasets \
  --n 13 \
  --out decontam_table.csv
```

## Day 4 — run all analyses; assemble tables

After the 8B eval finishes and the Qwen3 seed-2 runs finish:

```bash
cd ~/ExTra
conda activate ~/my_efs/envs/verl

# 1. Aggregate eval CSVs (already done by 05_eval_all_rebuttal.sh, but if
#    seed-2 runs were evaluated separately, re-aggregate):
python evals/aggregate_eval_results.py \
  --results_dir eval_outputs_rebuttal \
  --output rebuttal_table

# 2. Bootstrap CIs for 8B pair
python analysis/rebuttal/bootstrap_ci.py \
  --eval_dir eval_outputs_rebuttal \
  --n_bootstrap 10000 \
  --out paper_table_ci_8b.csv \
  --ref_run 01_GRPO_NanoNemotron_8B \
  --cmp_run 02_ExTra_Full_NanoNemotron_8B \
  --step 150 \
  --out_pvals paper_table_pvals_8b.csv

# 3. Bootstrap CIs for Qwen3 seed-2 pair
python analysis/rebuttal/bootstrap_ci.py \
  --eval_dir eval_outputs_rebuttal \
  --n_bootstrap 10000 \
  --out paper_table_ci_qwen3seed2.csv \
  --ref_run 03_GRPO_Qwen3_seed2 \
  --cmp_run 04_ExTra_Full_Qwen3_seed2 \
  --step 250 \
  --out_pvals paper_table_pvals_qwen3seed2.csv

# 4. Online MTE-gap for 8B ExTra run
python analysis/rebuttal/mte_gap_summary.py \
  --log outputs/mte_gap_log.jsonl \
  --out mte_gap_summary_8b.md
# (Move / rename the log if the Qwen3-seed2 ExTra run wrote to the same
#  path; the simplest workflow is to rename mte_gap_log.jsonl to
#  mte_gap_log_8b.jsonl right after the 8B ExTra run finishes.)

# 5. Wall-clock overhead (prefer W&B)
python analysis/rebuttal/wallclock_breakdown.py \
  --wandb_project ExTra_Rebuttal \
  --pairs 01_GRPO_NanoNemotron_8B,02_ExTra_Full_NanoNemotron_8B \
  --pairs 03_GRPO_Qwen3_seed2,04_ExTra_Full_Qwen3_seed2 \
  --out wallclock.csv
# If wandb is not available on this host:
python analysis/rebuttal/wallclock_breakdown.py \
  --log_glob 'outputs/**/main_ppo.log' \
  --out wallclock.csv

# 6. Reward-hacking sanity check
python analysis/rebuttal/reward_hacking_check.py \
  --eval_dir eval_outputs_rebuttal \
  --ref_run 01_GRPO_NanoNemotron_8B \
  --cmp_run 02_ExTra_Full_NanoNemotron_8B \
  --step 150 \
  --out reward_hack_8b.md
```

Now open `~/Downloads/ARR_May___ExTra/REBUTTAL_PLAN.md` and fill in the
Section 5 checklist. `analysis/rebuttal/README.md` has the exact placeholder
→ column map.

## Day 5 — polish and submit

- Sanity-check numbers land in the ranges reported by the paper (Qwen3 seed-2
  should be within ~1–2 pp of seed-1 aggregate).
- Draft the two per-reviewer responses from the combined text in
  `REBUTTAL_PLAN.md` §4 if the venue splits by reviewer.
- Submit.

## Fallbacks

### 8B OOMs at step 1 — tiered fallback

The default 8B config assumes full FSDP-8 sharding (no offload),
`gpu_memory_utilization=0.80`, `tensor_model_parallel_size=2`, and dynamic
batching capped at `ppo_max_token_len_per_gpu=32768`. Peak per-GPU memory
should stay under ~65 GB. If you still hit OOM, apply the tiers in order —
each tier trades ~10-30% throughput for headroom, so use the smallest tier
that keeps training stable.

**Tier 1 — squeeze KV cache and dynamic-bsz cap** (cheap, ~5% slowdown):
```bash
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.rollout.max_num_batched_tokens=24576
```

**Tier 2 — halve micro-batch, drop dynamic-bsz cap further** (~15% slowdown):
```bash
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

**Tier 3 — enable optimizer offload only** (~25% slowdown but frees ~10-15 GB/GPU):
```bash
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.60
```

**Tier 4 — full param + optimizer offload, TP=4** (last resort, ~40-50% slowdown):
```bash
bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
```

Apply the same overrides to `02_extra_full_nano8b.sh`. **Do NOT mix tiers
between GRPO and ExTra** — the paired comparison depends on both runs using
the same memory config so throughput/wall-clock is comparable in the R§5.5
paragraph.

### 8B step-time > 15 min (won't finish in the window)

Cut steps and eval budget:

```bash
TOTAL_STEPS=100 bash verl/examples/experiments/ExTra_Rebuttal/01_grpo_nano8b.sh \
  actor_rollout_ref.rollout.val_kwargs.n=8
```

Then note in the rebuttal caption: "n=8 samples per problem (pass@8) due to
rebuttal compute budget."

### Hydra rejects `+trainer.seed`

Only a concern for scripts 03/04. One-line patch to the config schema:

```bash
# Add `seed: int = 42` to the AlgoConfig-adjacent trainer config
python - <<'PY'
from pathlib import Path
cfg = Path("verl/verl/trainer/config/ppo_trainer.yaml")
text = cfg.read_text()
if "\nseed:" not in text.split("trainer:")[-1]:
    text = text.replace("trainer:", "trainer:\n  seed: 42", 1)
    cfg.write_text(text)
    print("Added trainer.seed=42 default")
PY
```

Then re-launch. This does **not** guarantee bit-exact reproducibility (Ray +
vLLM introduce nondeterminism), but it changes the sampling seed and gives
the reviewers a different draw.

### `outputs/mte_gap_log.jsonl` is empty after 30+ steps

Check:
1. `+algorithm.guided_resampling.log_mte_gap=True` is on the ExTra script's CLI.
2. Training is past `warmup_steps=30`.
3. At least one prompt in each batch is "hard" (all-incorrect group). Early
   on, hard prompts can be rare on Qwen3; the log fills up more once training
   pushes accuracy higher and there are stable hard problems. If genuinely
   nothing appears, drop warmup to 10:
   `algorithm.guided_resampling.warmup_steps=10`.

### W&B is unreachable from the AWS node

The wallclock script has a `--log_glob` fallback that parses timestamps out
of `outputs/**/main_ppo.log`. The units are per-step wall-clock in seconds;
the median is the reliable summary.

## Verification snippets

Before pushing this branch, on the dev laptop:

```bash
# Shell syntax
bash -n verl/examples/experiments/ExTra_Rebuttal/*.sh

# Python parse
python -c "import ast, glob; [ast.parse(open(p).read()) for p in glob.glob('analysis/rebuttal/*.py')]"

# Trainer imports (needs the conda env; otherwise skip)
python -c "import verl.verl.trainer.ppo.ray_trainer as _; print('OK')"
```

After the first training step lands on remote, verify the MTE log format:

```bash
head -1 outputs/mte_gap_log.jsonl | python -m json.tool
# Should show: step, ts, prompt_hash, prefix_memory_size, queue_size_before,
# mte_selected {…}, random_alt {…}
```
