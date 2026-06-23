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
"""ExTra trainer: GRPO + curiosity reward + entropy-guided prefix resampling.

The class is a subclass of ``RayPPOTrainer``. Everything related to worker
initialisation, checkpointing, validation, and dataloading is inherited.
``fit()`` is fully overridden so the ExTra-specific hooks can be inserted in
the exact places they need to live.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking

from .curiosity_memory import CuriosityMemory
from .extra_core_algos import compute_grpo_outcome_advantage_with_positive_novelty


class RayEXTRATrainer(RayPPOTrainer):
    """ExTra trainer — GRPO + curiosity reward + entropy-guided resampling.

    The two ExTra mechanisms are independent and individually opt-in via
    ``algorithm.curiosity.enable`` and ``algorithm.guided_resampling.enable``.
    With both disabled this trainer reduces to vanilla GRPO/PPO.
    """

    # ------------------------------------------------------------------ #
    # Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def init_workers(self):
        super().init_workers()
        self._init_extra_state()

    def _init_extra_state(self):
        """Allocate CuriosityMemory, queues, caches, embedding model.

        Called once after the base workers are up. Honours the master
        ``curiosity.enable`` / ``guided_resampling.enable`` flags — if both
        are off we keep the dispatch hooks but skip the embedding-model load
        so a vanilla GRPO run pays no GPU memory tax.
        """
        self.curiosity_enabled = bool(self._cfg_get("algorithm.curiosity.enable", False))
        self.guided_resampling_enabled = bool(self._cfg_get("algorithm.guided_resampling.enable", False))

        rollout_n = int(self._cfg_get("actor_rollout_ref.rollout.n", 8))
        self.curiosity_memory = CuriosityMemory(
            pad_token_id=self.tokenizer.pad_token_id,
            max_rollouts_per_prompt=int(self._cfg_get("algorithm.curiosity.max_rollouts_per_prompt", 16)),
            max_prefixes_per_prompt=int(self._cfg_get("algorithm.curiosity.max_prefixes_per_prompt", 128)),
            min_history_before_reward=rollout_n,
        )
        self._guided_prompt_queue: list[list[int]] = []

        self.embedding_model = None
        self._emb_on_gpu = False
        if self.curiosity_enabled or self.guided_resampling_enabled:
            # Sentence-Transformers is only used when at least one ExTra mechanism is on.
            # The driver process may not have CUDA visible (multi-node Ray) — fall back to CPU.
            from sentence_transformers import SentenceTransformer

            model_name = str(self._cfg_get(
                "algorithm.guided_resampling.embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ))
            try:
                torch.cuda.init()
                _emb_device = "cuda:0"
            except Exception:
                _emb_device = "cpu"
            self.embedding_model = SentenceTransformer(model_name, device=_emb_device)
            self._emb_on_gpu = (_emb_device == "cuda:0")

        self._prefix_emb_cache: dict[int, torch.Tensor] = {}
        self._prefix_emb_cache_max = 100_000
        self._newline_token_ids: Optional[set[int]] = None

    # ------------------------------------------------------------------ #
    # Small utilities                                                     #
    # ------------------------------------------------------------------ #
    def _cfg_get(self, path: str, default: Any) -> Any:
        value = OmegaConf.select(self.config, path)
        return default if value is None else value

    def _prompt_key(self, token_ids: torch.Tensor) -> tuple[int, ...]:
        return self.curiosity_memory.key(token_ids)

    def _build_newline_token_ids(self) -> set[int]:
        if self._newline_token_ids is not None:
            return self._newline_token_ids
        nl_ids: set[int] = set()
        for text in ["\n", "\n\n"]:
            nl_ids.update(self.tokenizer.encode(text, add_special_tokens=False))
        # Some tokenizers represent \n as a stand-alone token in the first ~1k ids.
        for tid in range(min(1000, self.tokenizer.vocab_size)):
            try:
                if self.tokenizer.decode([tid]) == "\n":
                    nl_ids.add(tid)
            except Exception:
                pass
        self._newline_token_ids = nl_ids
        return nl_ids

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings with the Sentence-Transformer, returning CPU tensors.

        Briefly moves the model to GPU (when available) for batched inference, then
        offloads back to CPU to keep vLLM's GPU memory free.
        """
        assert self.embedding_model is not None, "Embedding model not initialised."
        with torch.no_grad():
            if self._emb_on_gpu:
                self.embedding_model.to("cuda:0")
            embs = self.embedding_model.encode(texts, convert_to_tensor=True, batch_size=512).cpu()
            if self._emb_on_gpu:
                self.embedding_model.to("cpu")
                torch.cuda.empty_cache()
        return embs

    # ------------------------------------------------------------------ #
    # Guided resampling                                                   #
    # ------------------------------------------------------------------ #
    def _hard_prompt_keys(
        self, prompt_keys: list[tuple[int, ...]], reward_tensor: torch.Tensor
    ) -> list[tuple[int, ...]]:
        """Return the prompt-keys with pass-rate at or below 1/n (== "hard")."""
        n_rollout = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        threshold = 1.0 / n_rollout

        seq_scores = reward_tensor.sum(dim=-1)
        grouped_pass: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for k, score in zip(prompt_keys, seq_scores.cpu().tolist()):
            grouped_pass[k].append(1.0 if score > 0.0 else 0.0)

        hard = []
        for k, vals in grouped_pass.items():
            pass_rate = float(np.mean(vals)) if vals else 0.0
            if pass_rate <= threshold:
                hard.append(k)
        return hard

    def _update_prefix_memory(
        self,
        batch: DataProto,
        entropy_mat: torch.Tensor,
        hard_prompt_keys: list[tuple[int, ...]],
        prompt_keys: list[tuple[int, ...]],
    ) -> None:
        """For each hard prompt, pick the single lowest-entropy rollout and store
        a curated set of prefix snapshots (split on newline tokens)."""
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if getattr(self, "global_steps", 0) <= warmup_steps:
            return
        if entropy_mat is None or not hard_prompt_keys:
            return

        hard_set = set(hard_prompt_keys)
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]

        # Pick the most confident rollout per hard prompt.
        prompt_best: dict[tuple[int, ...], tuple[int, float]] = {}
        for i, pk in enumerate(prompt_keys):
            if pk not in hard_set:
                continue
            valid_len = int(response_mask[i].sum().item())
            if valid_len <= 0:
                continue
            mean_ent = float(entropy_mat[i, :valid_len].mean().item())
            if pk not in prompt_best or mean_ent < prompt_best[pk][1]:
                prompt_best[pk] = (i, mean_ent)
        if not prompt_best:
            return

        nl_ids = self._build_newline_token_ids()
        all_items: list[tuple[tuple[int, ...], list[int], float]] = []
        for pk, (idx, _) in prompt_best.items():
            valid_len = int(response_mask[idx].sum().item())
            id_list = responses[idx][:valid_len].cpu().tolist()

            split_positions = [pos + 1 for pos, tid in enumerate(id_list) if tid in nl_ids]
            if not split_positions:
                continue

            # Merge splits that are too close together (< 32 tokens).
            merged = [split_positions[0]]
            for pos in split_positions[1:]:
                if pos - merged[-1] >= 32:
                    merged.append(pos)
            split_positions = merged

            # Drop the trailing boundary (we want prefixes, not the full response).
            prefix_boundaries = split_positions[:-1] if len(split_positions) > 1 else split_positions

            # Cap at 15 prefixes per prompt, evenly spaced.
            if len(prefix_boundaries) > 15:
                step = len(prefix_boundaries) / 15
                prefix_boundaries = [prefix_boundaries[int(i * step)] for i in range(15)]

            for boundary in prefix_boundaries:
                prefix_ids = id_list[:boundary]
                if not prefix_ids:
                    continue
                prefix_len = min(boundary, valid_len)
                prefix_entropy_mean = float(entropy_mat[idx, :prefix_len].mean().item())
                all_items.append((pk, prefix_ids, prefix_entropy_mean))

        if not all_items:
            return

        # Deduplicate + batched encode.
        texts_to_encode: list[str] = []
        text_to_idx: dict[int, int] = {}
        item_hashes: list[int] = []
        for _, prefix_ids, _ in all_items:
            h = hash(tuple(prefix_ids))
            item_hashes.append(h)
            if h not in self._prefix_emb_cache and h not in text_to_idx:
                text_to_idx[h] = len(texts_to_encode)
                texts_to_encode.append(self.tokenizer.decode(prefix_ids, skip_special_tokens=True))

        if texts_to_encode:
            new_embeddings = self._embed_texts(texts_to_encode)
            for text_hash, idx in text_to_idx.items():
                self._prefix_emb_cache[text_hash] = new_embeddings[idx]

        for (pk, prefix_ids, prefix_entropy_mean), h in zip(all_items, item_hashes):
            self.curiosity_memory.add_prefix_entry(pk, prefix_ids, prefix_entropy_mean, self._prefix_emb_cache[h])

        # LRU-evict oldest entries.
        if len(self._prefix_emb_cache) > self._prefix_emb_cache_max:
            excess = len(self._prefix_emb_cache) - self._prefix_emb_cache_max
            for k in list(self._prefix_emb_cache.keys())[:excess]:
                del self._prefix_emb_cache[k]

    def _enqueue_guided_prompts(self, hard_prompt_keys: list[tuple[int, ...]]) -> None:
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if getattr(self, "global_steps", 0) <= warmup_steps:
            return

        tau = float(self._cfg_get("algorithm.guided_resampling.tau", 0.1))
        max_prompt_len = int(self._cfg_get("data.max_prompt_length", 2048))
        for k in hard_prompt_keys:
            best = self.curiosity_memory.select_best_prefix(k, tau=tau)
            if best is None:
                continue
            guided_ids = (list(k) + list(best["prefix_token_ids"]))[:max_prompt_len]
            if guided_ids:
                self._guided_prompt_queue.append(guided_ids)

        max_queue = int(self._cfg_get("algorithm.guided_resampling.max_queue_size", 16))
        if len(self._guided_prompt_queue) > max_queue:
            self._guided_prompt_queue = self._guided_prompt_queue[-max_queue:]

    def _dequeue_guided_prompts_for_regen(self) -> list[list[int]]:
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if not self.guided_resampling_enabled or getattr(self, "global_steps", 0) <= warmup_steps:
            return []
        if not self._guided_prompt_queue:
            return []

        regen_batch_size = int(
            self._cfg_get(
                "algorithm.guided_resampling.regen_batch_size",
                self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            )
        )
        n = min(len(self._guided_prompt_queue), regen_batch_size)
        prompts = self._guided_prompt_queue[:n]
        self._guided_prompt_queue = self._guided_prompt_queue[n:]
        return prompts

    def _inject_guided_prompts_into_batch(self, batch: DataProto) -> int:
        """Replace the tail of the dataloader batch with queued guided prompts so a
        single ``generate_sequences`` call covers both regular and guided traffic."""
        guided_prompt_lists = self._dequeue_guided_prompts_for_regen()
        if not guided_prompt_lists:
            return 0

        batch_size = batch.batch["input_ids"].shape[0]
        n_replace = min(len(guided_prompt_lists), batch_size)
        if n_replace < len(guided_prompt_lists):
            # Stash the overflow for the next step.
            self._guided_prompt_queue = guided_prompt_lists[n_replace:] + self._guided_prompt_queue
        guided_prompt_lists = guided_prompt_lists[:n_replace]

        pad_id = self.tokenizer.pad_token_id
        max_len = batch.batch["input_ids"].shape[1]

        for i, ids in enumerate(guided_prompt_lists):
            idx = batch_size - n_replace + i  # replace from the tail
            seq_len = min(len(ids), max_len)
            if seq_len == 0:
                continue
            batch.batch["input_ids"][idx].fill_(pad_id)
            batch.batch["attention_mask"][idx].zero_()
            batch.batch["position_ids"][idx].zero_()
            # Left-pad: tokens at the right end.
            batch.batch["input_ids"][idx, max_len - seq_len:] = torch.tensor(ids[:seq_len], dtype=torch.long)
            batch.batch["attention_mask"][idx, max_len - seq_len:] = 1
            batch.batch["position_ids"][idx, max_len - seq_len:] = torch.arange(seq_len, dtype=torch.long)
            if "raw_prompt_ids" in batch.non_tensor_batch:
                batch.non_tensor_batch["raw_prompt_ids"][idx] = np.array(ids[:seq_len], dtype=np.int64)
            if "raw_prompt" in batch.non_tensor_batch:
                batch.non_tensor_batch["raw_prompt"][idx] = None

        return n_replace

    # ------------------------------------------------------------------ #
    # Novelty reward                                                      #
    # ------------------------------------------------------------------ #
    def _compute_intrinsic_rewards(
        self,
        batch: DataProto,
        prompt_keys: list[tuple[int, ...]],
        novelty_after_norm: bool,
        novelty_lambda: float,
    ) -> Optional[torch.Tensor]:
        """Compute the novelty bonus tensor (``intrinsic_rewards``) and store it on the batch.

        Returns the tensor for logging convenience; ``None`` if no metric ran.
        """
        novelty_metric = str(self._cfg_get("algorithm.curiosity.novelty_metric", "embedding"))

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        valid_lengths = response_mask.sum(dim=-1).int()

        # Tail (last 512 tokens) of each response — used as the unit of comparison.
        tail_id_lists: list[list[int]] = []
        for i in range(responses.shape[0]):
            vl = int(valid_lengths[i].item())
            tail_id_lists.append(responses[i][max(0, vl - 512):vl].tolist())

        device = batch.batch["token_level_scores"].device

        if novelty_metric == "ngram_count":
            ngram_n = int(self._cfg_get("algorithm.curiosity.novelty_ngram", 2))
            novelty_rewards = self.curiosity_memory.compute_ngram_count_novelty(
                prompt_keys, tail_id_lists, ngram_n=ngram_n,
            )
            seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
            pos_mask = (seq_rewards > 0).float()
            self.curiosity_memory.update_ngram_counts(prompt_keys, tail_id_lists, pos_mask, ngram_n=ngram_n)

        elif novelty_metric == "policy_surprise":
            old_log_probs = batch.batch["old_log_probs"]  # [batch, seq_len]
            seq_log_probs = (old_log_probs * response_mask).sum(dim=-1)
            pos_mask = (batch.batch["token_level_scores"].sum(dim=-1) > 0).float()
            novelty_rewards = self.curiosity_memory.compute_policy_surprise_novelty(
                prompt_keys, seq_log_probs, pos_mask,
            )

        else:
            # Default: embedding cosine distance to recent rollouts (+ batch siblings).
            rollout_texts = self.tokenizer.batch_decode(tail_id_lists, skip_special_tokens=True)
            rollout_embeddings = self._embed_texts(rollout_texts)
            novelty_rewards = self.curiosity_memory.add_rollout_embeddings(
                prompt_keys, rollout_embeddings,
                normalize_per_group=(not novelty_after_norm),
            )

            if not novelty_after_norm:
                # Option A: per-prompt EMA-variance gating.
                dynamic_scales = []
                for k in prompt_keys:
                    var_r = self.curiosity_memory.reward_variance_ema.get(k, 0.0)
                    dynamic_scales.append(float(np.exp(-novelty_lambda * var_r)))
                scales = torch.tensor(dynamic_scales, device=novelty_rewards.device, dtype=novelty_rewards.dtype)
                novelty_rewards = novelty_rewards * scales

        batch.batch["intrinsic_rewards"] = novelty_rewards.to(device)
        return batch.batch["intrinsic_rewards"]

    def _update_reward_variance_ema(self, prompt_keys: list[tuple[int, ...]], seq_rewards: torch.Tensor) -> None:
        prompt_to_rewards: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for k, r in zip(prompt_keys, seq_rewards.cpu().tolist()):
            prompt_to_rewards[k].append(r)
        alpha = self.curiosity_memory.ema_alpha
        ema = self.curiosity_memory.reward_variance_ema
        for k, r_list in prompt_to_rewards.items():
            var = float(np.var(r_list)) if len(r_list) > 1 else 0.0
            ema[k] = (1 - alpha) * ema[k] + alpha * var if k in ema else var

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #
    def fit(self):
        """Training loop with ExTra hooks woven in.

        The structure mirrors the upstream ``RayPPOTrainer.fit()`` so future
        verl updates can be re-applied with a small diff.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.total_samples_seen = 0

        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, float] = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # === ExTra hook 1: inject queued guided prompts ===
                n_guided_injected = self._inject_guided_prompts_into_batch(batch)
                if n_guided_injected > 0:
                    metrics["exploration/guided_injected"] = n_guided_injected

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                for opt_key in ("multi_modal_data", "raw_prompt", "tools_kwargs",
                                "interaction_kwargs", "index", "agent_name"):
                    if opt_key in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(opt_key)

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    self.total_samples_seen += len(batch.batch)
                    metrics["trainer/total_samples_seen"] = self.total_samples_seen

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # === ExTra hook 2: capture per-token entropies ===
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        policy_entropys = entropys  # snapshot for exploration block
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch:
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        novelty_lambda = float(self._cfg_get("algorithm.curiosity.novelty_lambda", 1.0))
                        novelty_scale = float(self._cfg_get("algorithm.curiosity.novelty_reward_scale", 0.0))
                        novelty_after_norm = bool(self._cfg_get("algorithm.curiosity.novelty_after_norm", False))

                        # === ExTra hook 3: novelty + guided-resampling bookkeeping ===
                        prompt_keys: list[tuple[int, ...]] = []
                        if self.curiosity_enabled or self.guided_resampling_enabled:
                            with marked_timer("exploration", timing_raw, color="purple"):
                                prompt_keys = [self._prompt_key(p) for p in batch.batch["prompts"]]
                                seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                                self._update_reward_variance_ema(prompt_keys, seq_rewards)

                                if self.curiosity_enabled:
                                    self._compute_intrinsic_rewards(
                                        batch, prompt_keys,
                                        novelty_after_norm=novelty_after_norm,
                                        novelty_lambda=novelty_lambda,
                                    )
                                    metrics["exploration/novelty_reward_mean"] = float(
                                        batch.batch["intrinsic_rewards"].mean().item()
                                    )

                                if self.guided_resampling_enabled:
                                    hard_keys = self._hard_prompt_keys(prompt_keys, reward_tensor)
                                    self._update_prefix_memory(batch, policy_entropys, hard_keys, prompt_keys)
                                    self._enqueue_guided_prompts(hard_keys)
                                    metrics["exploration/hard_prompt_count"] = len(hard_keys)
                                    metrics["exploration/guided_queue_size"] = len(self._guided_prompt_queue)
                                    if hard_keys:
                                        metrics["exploration/hard_passrate_threshold"] = 1.0 / max(
                                            int(self.config.actor_rollout_ref.rollout.n), 1
                                        )

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        use_positive_novelty_adv = (
                            self.curiosity_enabled
                            and novelty_scale != 0.0
                            and "intrinsic_rewards" in batch.batch
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO
                        )

                        if use_positive_novelty_adv:
                            advantages, returns = compute_grpo_outcome_advantage_with_positive_novelty(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                intrinsic_rewards=batch.batch["intrinsic_rewards"],
                                response_mask=batch.batch["response_mask"],
                                index=batch.non_tensor_batch["uid"],
                                novelty_alpha=novelty_scale,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                novelty_after_norm=novelty_after_norm,
                            )
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id", batch.non_tensor_batch["request_id"].tolist()
                                )
                            self._dump_generations(
                                inputs=inputs, outputs=outputs, gts=sample_gts, scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict, dump_path=rollout_data_dir,
                            )

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    save_start_step = self.config.trainer.get("save_start_step", 0)
                    if self.config.trainer.save_freq > 0 and self.global_steps >= save_start_step and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                    weights = getattr(self.train_dataloader.sampler, "weights", None)
                    if weights is not None:
                        n_easy = int((weights < 0.5).sum().item())
                        metrics["curriculum/n_easy_downweighted"] = n_easy
                        metrics["curriculum/pct_active"] = 1.0 - n_easy / len(weights)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
