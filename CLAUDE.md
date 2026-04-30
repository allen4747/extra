# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ExTra** (Exploratory Trajectory Optimization of LLM Reinforcement Learning) is a research project built on the **verl** library. It enhances GRPO by introducing entropy-guided resampling and curiosity-driven novelty rewards to improve data efficiency in RL training of LLMs.

## Key Architecture

- **`verl/verl/trainer/ppo/ray_trainer.py`**: Main trainer (`RayPPOTrainer`) and the ExTra-specific `CuriosityMemory` class for novelty tracking and prefix-guided resampling.
- **`verl/verl/trainer/main_ppo.py`**: Hydra-decorated entry point for training.
- **`verl/verl/workers/`**: Ray-based distributed workers (actor, rollout, reward_manager).
- **`verl/verl/trainer/config/`**: Hydra YAML configs for algorithms and training.
- **`verl/examples/experiments/ExTra_runs/`**: Experiment shell scripts (baselines, ablations, full ExTra).
- **`analysis/`**: Python scripts for analyzing ablation results (entropy signals, pass@k, trajectory comparisons).
- **`auto_extra.py`**: GPU monitor and parallel experiment scheduler for the shared node.

### Training Flow

Single-controller pattern: `RayPPOTrainer` orchestrates Ray worker groups (actor, rollout via vLLM/SGLang, reward).

### ExTra-Specific Config Parameters

```
algorithm.curiosity.enable=True
algorithm.curiosity.novelty_reward_scale=0.1
algorithm.guided_resampling.enable=True
algorithm.guided_resampling.tau=0.1
algorithm.guided_resampling.regen_batch_size=32
```

## Env

```bash
conda activate ~/my_efs/envs/verl
```

## Running Experiments

```bash
# Under verl/ folder
# Run a single experiment (typically needs 2 GPUs)
bash examples/experiments/ExTra_runs/02_extra_full.sh
```


## Development Conventions

- Prefer commenting out previous code and adding new logic over directly overwriting existing code.
- Main concern is experiment design — modify key algorithm files with caution.
- Current problem is the poor performance of ExTra.
- Experiment scripts configure `CUDA_VISIBLE_DEVICES` for 2-GPU or 4-GPU runs.
- Logging to Weights & Biases under project `ExTra_Research`.
