# ExTra_Final — Paper Submission Experiment Plan

All scripts in this folder use a **single canonical config** (length budget tuned for
R1-Distill / Nemotron long-CoT, greedy `mean@1` validation, project `ExTra_Final` on wandb).
The only per-experiment differences are: base model, `algorithm.curiosity.*`,
`algorithm.guided_resampling.*`, and `CUDA_VISIBLE_DEVICES`.

Hardware assumption: **8× H100, allocated 2 GPUs per run, 4 runs in parallel**.

---

## Experiment matrix (12 runs)

| # | Script | Base | Method | Slot | Wave |
|---|--------|------|--------|------|------|
| 01 | `01_grpo_r1distill.sh` | R1-Distill-Qwen-1.5B | GRPO baseline | A (0,1) | 1 |
| 02 | `02_extra_full_r1distill.sh` | R1-Distill-Qwen-1.5B | **ExTra-Full** | B (2,3) | 1 |
| 05 | `05_grpo_nemotron.sh` | Nemotron-1.5B | GRPO baseline | C (4,5) | 1 |
| 06 | `06_extra_full_nemotron.sh` | Nemotron-1.5B | **ExTra-Full** | D (6,7) | 1 |
| 03 | `03_extra_regenonly_r1distill.sh` | R1-Distill | regen only (α=0) | A | 2 |
| 04 | `04_extra_curionly_r1distill.sh` | R1-Distill | curiosity only | B | 2 |
| 07 | `07_extra_regenonly_nemotron.sh` | Nemotron | regen only | C | 2 |
| 08 | `08_extra_curionly_nemotron.sh` | Nemotron | curiosity only | D | 2 |
| 09 | `09_extra_tau1_r1distill.sh` | R1-Distill | ExTra-Full, τ=1.0 | A | 3 |
| 10 | `10_extra_alpha005_r1distill.sh` | R1-Distill | ExTra-Full, α=0.05 | B | 3 |
| 11 | `11_extra_alpha02_r1distill.sh` | R1-Distill | ExTra-Full, α=0.2 | C | 3 |
| 12 | `12_extra_warmup200_r1distill.sh` | R1-Distill | ExTra-Full, warmup=200 | D | 3 |
| 13 | `13_run_all_evals.sh` | — | eval orchestrator | all 8 | 4 |

**Headline table** (Wave 1): GRPO vs ExTra-Full on **two reasoning bases** — the main result.
**2×2 ablation** (Wave 2): {curiosity, regen} × {on, off} on each base — isolates contribution of each component.
**Sensitivity** (Wave 3): τ, α, warmup on the primary base only — robustness to hyperparameters.

---

## Running order (4-day plan)

### Day 0 (now): Smoke-test 01 for ~30 steps
Before kicking off the full 4-way parallel set, run 01 alone for ~30 steps and verify in wandb:

- `rollout/response_length/mean` is rising into 4–8k range, not pinned at 16384
- `critic/score/mean` is non-zero (typically 0.3–0.6 by step 20)
- No OOM, no truncation explosion

If anything looks broken, fix before launching the full set. Cost: ~1 hour.

### Day 1: Wave 1 — main results
```bash
bash 01_grpo_r1distill.sh        # GPUs 0,1
bash 02_extra_full_r1distill.sh  # GPUs 2,3
bash 05_grpo_nemotron.sh         # GPUs 4,5
bash 06_extra_full_nemotron.sh   # GPUs 6,7
```
Run all 4 in parallel (e.g., 4 separate tmux/screen sessions). Expected wall-clock per run: **~10–14 hours** at 16k context on 2× H100. End of Day 1: 4 runs done.

### Day 2: Wave 2 — ablations
```bash
bash 03_extra_regenonly_r1distill.sh  # GPUs 0,1
bash 04_extra_curionly_r1distill.sh   # GPUs 2,3
bash 07_extra_regenonly_nemotron.sh   # GPUs 4,5
bash 08_extra_curionly_nemotron.sh    # GPUs 6,7
```
End of Day 2: 8 runs total.

### Day 3: Wave 3 — sensitivity
```bash
bash 09_extra_tau1_r1distill.sh         # GPUs 0,1
bash 10_extra_alpha005_r1distill.sh     # GPUs 2,3
bash 11_extra_alpha02_r1distill.sh      # GPUs 4,5
bash 12_extra_warmup200_r1distill.sh    # GPUs 6,7
```
End of Day 3: all 12 training runs done.

### Day 4: eval + writeup
```bash
bash 13_run_all_evals.sh
```
Walks every `global_step_300` checkpoint, merges FSDP → HF, runs `evals/gen_vllm.py` then `evals/grade.py`. Expected wall-clock: ~2–4 hours total for 12 checkpoints (1.5B vLLM is fast).

After eval finishes, pull MATH500 / AIME24 / AMC23 numbers into the paper table.

---

## Compute budget summary

- 12 training runs × ~12 hours / 4-way parallel = **~36 hours wall-clock = ~1.5 days of nights+day**
- + Day 0 smoke (~1 hour) + Day 4 eval (~3 hours) ≈ **2 days minimum, 4 days comfortable**
- Total GPU-hours: ~12 × 12 × 2 = **~290 H100-hours**

---

## Drop-the-scope plan if time slips

If only 2 days remain when starting, drop in this order:
1. Drop Wave 3 (sensitivity sweeps 09–12) — keep ablation grid + main result
2. Drop the Nemotron ablation pair (07, 08) — keep R1-Distill ablations + 2-base headline
3. Drop the Nemotron pair entirely (05, 06) — single-base paper, weak but submittable

Minimum viable: **01, 02, 03, 04** (4 runs, 1 day) → defensible R1-Distill-only paper with 2×2 ablation.

---

## Eval benchmarks

`evals/gen_vllm.py` controls which benchmarks each checkpoint is graded on. Verify it covers:
- **MATH500** (in-distribution-ish, primary)
- **AIME24** or **AMC23** (held-out, harder, shows real generalization)
- **GSM8K** (sanity check, optional)

If only MATH500 is wired up there, expand it before kicking off Wave 4 — single-benchmark eval is the second-most-common reviewer complaint after single-base.

---

## Wandb hygiene

All runs log to project `ExTra_Final`. Filter by `experiment_name` prefix `01_`, `02_`, etc.
Drop runs from the old `ExTra_Research` project from any paper figures — those have stale configs (max_response_length=8192, sampled mean@8 validation) and aren't comparable to ExTra_Final curves.
