# ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning

This repository is the official implementation of **ExTra**, a recipe built on
[verl](https://github.com/volcengine/verl) that augments GRPO with two mechanisms targeting the *exploration bottleneck* in RL fine-tuning of LLMs.


## Project layout

```
ExTra/
└── verl/
    ├── verl/               # upstream verl source
    ├── recipe/
    │   ├── extra/          # ← ExTra lives here
    │   │   ├── main_extra.py
    │   │   ├── extra_ray_trainer.py
    │   │   ├── curiosity_memory.py
    │   │   ├── extra_core_algos.py
    │   │   ├── config/extra_trainer.yaml
    │   │   └── scripts/    # experiment scripts
    │   ├── dapo/     
    │   ├── entropy/
    │   └── ...
    ├── examples/           # standard verl examples
    └── ...
```

## Quickstart

### Install

Please refer to the official [verl](https://github.com/volcengine/verl) for dependence installation.

### Run

```bash
cd verl

# Vanilla GRPO baseline.
bash recipe/extra/scripts/grpo_baseline.sh

# Full ExTra: curiosity + guided resampling.
bash recipe/extra/scripts/extra_full.sh
```


## Acknowledgements

ExTra is built on top of the [verl](https://github.com/volcengine/verl) project.

## Citation

```bibtex
@misc{extra2026,
  title         = {ExTra: Exploratory Trajectory Optimisation for RL Fine-tuning of Language Models},
  author        = {<authors>},
  year          = {2026},
  eprint        = {<arxiv id>},
  archivePrefix = {arXiv}
}
```
