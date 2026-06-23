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
"""Per-prompt memory for rollout novelty and entropy-guided prefix resampling.

The class is intentionally framework-light: it accepts already-computed embeddings
(or raw token-id lists, for n-gram and policy-surprise metrics) and returns
intrinsic rewards as plain tensors. All embedding inference is the caller's job.
"""

from collections import Counter, defaultdict, deque
from typing import Any, Optional

import torch
import torch.nn.functional as F


class CuriosityMemory:
    """Per-prompt memory for rollout novelty and prefix-guided resampling."""

    def __init__(
        self,
        pad_token_id: int,
        max_rollouts_per_prompt: int = 16,
        max_prefixes_per_prompt: int = 64,
        min_history_before_reward: int = 8,
    ):
        self.pad_token_id = pad_token_id
        self.max_rollouts_per_prompt = max_rollouts_per_prompt
        self.max_prefixes_per_prompt = max_prefixes_per_prompt
        self.min_history_before_reward = min_history_before_reward
        self.rollout_memory: dict[tuple[int, ...], deque[torch.Tensor]] = {}
        self.prefix_memory: dict[tuple[int, ...], deque[dict[str, Any]]] = {}
        self.reward_variance_ema: dict[tuple[int, ...], float] = {}
        self.ema_alpha = 0.1
        # Lazy: created on first use of n-gram novelty.
        self.ngram_counts: dict[tuple[int, ...], Counter] = {}

    def key(self, token_ids) -> tuple[int, ...]:
        """Normalize a prompt to a hashable key (pad tokens stripped)."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().tolist()
        else:
            ids = list(token_ids)
        return tuple(int(t) for t in ids if int(t) != self.pad_token_id)

    # ------------------------------------------------------------------ #
    # Novelty metric A: embedding-based cosine distance                  #
    # ------------------------------------------------------------------ #
    def add_rollout_embeddings(
        self,
        prompt_keys: list[tuple[int, ...]],
        embeddings: torch.Tensor,
        normalize_per_group: bool = True,
    ) -> torch.Tensor:
        """Return novelty reward = nearest-neighbor cosine distance to history + current batch siblings.

        Warm-up: returns zero reward until the per-prompt history has accumulated
        at least ``min_history_before_reward`` entries (so the first batch behaves
        identically to the baseline).

        Args:
            normalize_per_group: If True (Option A), normalize rewards within each group by max.
                If False (Option B), return raw cosine distances.
        """
        rewards = torch.zeros(embeddings.shape[0], device=embeddings.device, dtype=embeddings.dtype)

        # 1. Group indices and embeddings by prompt key
        grouped_embs = defaultdict(list)
        for i, (k, emb) in enumerate(zip(prompt_keys, embeddings, strict=True)):
            grouped_embs[k].append((i, emb))

        # 2. Process each prompt key
        for k, items in grouped_embs.items():
            if k not in self.rollout_memory:
                self.rollout_memory[k] = deque(maxlen=self.max_rollouts_per_prompt)

            hist = self.rollout_memory[k]

            indices = [x[0] for x in items]
            new_embs = torch.stack([x[1] for x in items], dim=0)  # [M, D]

            # Warm-up: skip novelty reward until enough history exists
            if len(hist) < self.min_history_before_reward:
                pass
            else:
                # Compare against history + current batch siblings
                hist_tensor = torch.stack(list(hist), dim=0).to(embeddings.device, dtype=embeddings.dtype)
                pool = torch.cat([hist_tensor, new_embs], dim=0)

                M = new_embs.shape[0]
                H = pool.shape[0] - M

                new_embs_norm = F.normalize(new_embs, p=2, dim=-1)
                pool_norm = F.normalize(pool, p=2, dim=-1)
                sim = torch.mm(new_embs_norm, pool_norm.transpose(0, 1))  # [M, H+M]

                # Mask out self-comparisons
                mask = torch.ones_like(sim, dtype=torch.bool)
                mask[torch.arange(M), torch.arange(H, H + M)] = False
                sim = sim.masked_fill(~mask, float("-inf"))

                max_sim, _ = sim.max(dim=1)
                nearest_dist = 1.0 - max_sim
                rewards[indices] = torch.clamp(nearest_dist, min=0.0)

            # 3. Update the historical memory
            new_embs_detached = new_embs.detach().float().cpu()
            for emb_cpu in new_embs_detached:
                self.rollout_memory[k].append(emb_cpu)

        # 4. Optionally normalize rewards within each prompt group
        if normalize_per_group:
            for k, items in grouped_embs.items():
                indices = [x[0] for x in items]
                group_rewards = rewards[indices]
                max_reward = group_rewards.max()
                if max_reward > 1e-8:
                    rewards[indices] = group_rewards / max_reward

        return rewards

    # ------------------------------------------------------------------ #
    # Novelty metric B: count-based n-gram visitation                    #
    # ------------------------------------------------------------------ #
    def compute_ngram_count_novelty(
        self,
        prompt_keys: list[tuple[int, ...]],
        token_id_lists: list[list[int]],
        ngram_n: int = 2,
    ) -> torch.Tensor:
        """Count-based exploration: reward responses containing rare n-grams.

        Maintains per-prompt n-gram visitation counts. Novelty for a response is
        the mean inverse-square-root count over its n-grams:
            novelty(y|x) = mean_{g in ngrams(y)} 1/sqrt(N_x(g) + 1)

        Only updates counts for responses whose rewards are positive (call
        ``update_ngram_counts`` separately after gating).
        """
        rewards = torch.zeros(len(token_id_lists), dtype=torch.float32)

        grouped = defaultdict(list)
        for i, (k, toks) in enumerate(zip(prompt_keys, token_id_lists, strict=True)):
            grouped[k].append((i, toks))

        for k, items in grouped.items():
            if k not in self.ngram_counts:
                self.ngram_counts[k] = Counter()

            counts = self.ngram_counts[k]

            for idx, toks in items:
                if len(toks) < ngram_n:
                    continue
                ngrams = [tuple(toks[j:j + ngram_n]) for j in range(len(toks) - ngram_n + 1)]
                if not ngrams:
                    continue
                novelty = sum(1.0 / (counts[g] + 1) ** 0.5 for g in ngrams) / len(ngrams)
                rewards[idx] = novelty

        return rewards

    def update_ngram_counts(
        self,
        prompt_keys: list[tuple[int, ...]],
        token_id_lists: list[list[int]],
        positive_mask: torch.Tensor,
        ngram_n: int = 2,
    ) -> None:
        """Update n-gram counts only for correct (positive reward) responses."""
        for i, (k, toks) in enumerate(zip(prompt_keys, token_id_lists, strict=True)):
            if positive_mask[i].item() <= 0:
                continue
            if k not in self.ngram_counts:
                self.ngram_counts[k] = Counter()
            if len(toks) < ngram_n:
                continue
            ngrams = [tuple(toks[j:j + ngram_n]) for j in range(len(toks) - ngram_n + 1)]
            self.ngram_counts[k].update(ngrams)

    # ------------------------------------------------------------------ #
    # Novelty metric C: policy-surprise (correct + low-prob)             #
    # ------------------------------------------------------------------ #
    def compute_policy_surprise_novelty(
        self,
        prompt_keys: list[tuple[int, ...]],
        seq_log_probs: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Policy-surprise novelty: reward correct solutions the policy finds unlikely.

        Within each prompt group's correct solutions, compute z-scored negative
        log-probability. Solutions with lower log-prob (more surprising to the
        current policy) get higher novelty bonus. Returns 0 for groups with fewer
        than 2 correct solutions.
        """
        rewards = torch.zeros_like(seq_log_probs)

        grouped = defaultdict(list)
        for i, k in enumerate(prompt_keys):
            grouped[k].append(i)

        for k, indices in grouped.items():
            correct_indices = [i for i in indices if positive_mask[i].item() > 0]
            if len(correct_indices) < 2:
                continue

            correct_logprobs = seq_log_probs[correct_indices]
            mean_lp = correct_logprobs.mean()
            std_lp = correct_logprobs.std()
            if std_lp < 1e-8:
                continue

            # Negative z-score: lower log-prob (more surprising) → higher novelty.
            # Clamp to [0, inf) — only reward surprising solutions, don't penalize typical ones.
            surprise = -(correct_logprobs - mean_lp) / (std_lp + 1e-8)
            surprise = torch.clamp(surprise, min=0.0)

            for j, idx in enumerate(correct_indices):
                rewards[idx] = surprise[j]

        return rewards

    # ------------------------------------------------------------------ #
    # Prefix memory for entropy-guided resampling                        #
    # ------------------------------------------------------------------ #
    def add_prefix_entry(
        self,
        prompt_key: tuple[int, ...],
        prefix_token_ids: list[int],
        prefix_entropy_mean: float,
        embedding: torch.Tensor,
    ) -> None:
        if prompt_key not in self.prefix_memory:
            self.prefix_memory[prompt_key] = deque(maxlen=self.max_prefixes_per_prompt)
        self.prefix_memory[prompt_key].append(
            {
                "prefix_token_ids": tuple(prefix_token_ids),
                "prefix_entropy_mean": float(prefix_entropy_mean),
                "embedding": embedding.detach().float().cpu(),
            }
        )

    def select_best_prefix(self, prompt_key: tuple[int, ...], tau: float = 0.1) -> Optional[dict[str, Any]]:
        """Pick the prefix whose neighbourhood (softmax-weighted by embedding similarity)
        has the lowest mean entropy. Higher confidence with diverse support."""
        entries = self.prefix_memory.get(prompt_key)
        if not entries:
            return None

        embs = torch.stack([x["embedding"] for x in entries], dim=0).float()
        embs = F.normalize(embs, p=2, dim=1)
        raw_scores = torch.tensor([x["prefix_entropy_mean"] for x in entries], dtype=torch.float32)
        sim_matrix = torch.mm(embs, embs.t())
        weights = torch.softmax(sim_matrix / max(float(tau), 1e-6), dim=1)
        smoothed = torch.mv(weights, raw_scores)
        best_idx = int(torch.argmin(smoothed).item())

        best = dict(entries[best_idx])
        best["smoothed_prefix_entropy_mean"] = float(smoothed[best_idx].item())
        return best
