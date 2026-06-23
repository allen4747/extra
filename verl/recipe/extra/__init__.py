# Copyright 2025 ExTra Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ExTra: Exploratory Trajectory Optimization recipe for verl.

Two on-top mechanisms over GRPO:

  - **Curiosity reward**: a per-prompt novelty bonus, gated by sequence correctness,
    added to GRPO advantages. Three novelty metrics are supported: ``embedding``
    (Sentence-Transformers cosine distance vs. recent rollouts), ``ngram_count``
    (inverse-sqrt n-gram visitation), and ``policy_surprise`` (z-scored negative
    log-prob among correct solutions in a group).

  - **Entropy-guided resampling**: prefixes of low-entropy rollouts from hard
    prompts (0-pass-rate groups) are stored with their embeddings; at later steps
    a softmax-smoothed lowest-entropy prefix is selected and injected as a guided
    prompt for subsequent rollouts, focusing exploration on promising trajectories.
"""

from .curiosity_memory import CuriosityMemory
from .extra_core_algos import compute_grpo_outcome_advantage_with_positive_novelty
from .extra_ray_trainer import RayEXTRATrainer

__all__ = [
    "CuriosityMemory",
    "compute_grpo_outcome_advantage_with_positive_novelty",
    "RayEXTRATrainer",
]
