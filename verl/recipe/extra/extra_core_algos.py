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
"""ExTra-specific advantage estimator: GRPO with a positively-gated novelty bonus."""

from collections import defaultdict

import numpy as np
import torch


def compute_grpo_outcome_advantage_with_positive_novelty(
    token_level_rewards: torch.Tensor,
    intrinsic_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    novelty_alpha: float,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    novelty_after_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO-style outcome advantage with a novelty bonus gated by positive sequence reward.

    Option A (``novelty_after_norm=False``, default):
        ``score = sequence_reward + novelty_alpha * intrinsic_reward * 1[sequence_reward > 0]``
        Then apply standard GRPO group-wise centering/normalization.

    Option B (``novelty_after_norm=True``):
        Compute standard GRPO advantage first, then add the novelty bonus to correct rollouts.
        This preserves GRPO's variance reduction and never penalizes correct solutions.

    Returns ``(advantages, returns)`` — both broadcast to ``[batch, response_len]``
    by multiplying with ``response_mask``.
    """
    sequence_scores = token_level_rewards.sum(dim=-1)
    positive_mask = (sequence_scores > 0).to(sequence_scores.dtype)

    if not novelty_after_norm:
        # Option A: novelty enters before group normalization
        scores = sequence_scores + novelty_alpha * intrinsic_rewards * positive_mask
    else:
        # Option B: use raw task reward for GRPO normalization
        scores = sequence_scores.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        if novelty_after_norm:
            # Option B: add novelty bonus after GRPO normalization
            scores = scores + novelty_alpha * intrinsic_rewards * positive_mask

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
