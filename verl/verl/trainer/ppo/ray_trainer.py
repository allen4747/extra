# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
import warnings
from collections import Counter, deque
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    process_thoughts,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from sentence_transformers import SentenceTransformer

WorkerType = type[Worker]


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

    def key(self, token_ids: torch.Tensor | list[int] | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().tolist()
        else:
            ids = list(token_ids)
        return tuple(int(t) for t in ids if int(t) != self.pad_token_id)

    # def add_rollout_embeddings(self, prompt_keys: list[tuple[int, ...]], embeddings: torch.Tensor) -> torch.Tensor:
    #     """Return novelty reward = nearest-neighbor cosine distance to history."""
    #     rewards = torch.zeros(embeddings.shape[0], device=embeddings.device, dtype=embeddings.dtype)

    #     for i, (k, emb) in enumerate(zip(prompt_keys, embeddings, strict=True)):
    #         if k not in self.rollout_memory:
    #             self.rollout_memory[k] = deque(maxlen=self.max_rollouts_per_prompt)

    #         hist = self.rollout_memory[k]
    #         emb_detached = emb.detach().float().cpu()
    #         if len(hist) == 0:
    #             rewards[i] = 1.0
    #         else:
    #             hist_tensor = torch.stack(list(hist), dim=0).to(embeddings.device, dtype=embeddings.dtype)
    #             sim = F.cosine_similarity(emb.unsqueeze(0), hist_tensor, dim=-1)
    #             nearest_dist = 1.0 - sim.max()
    #             rewards[i] = torch.clamp(nearest_dist, min=0.0)

    #         hist.append(emb_detached)

    #     return rewards

    def add_rollout_embeddings(self, prompt_keys: list[tuple[int, ...]], embeddings: torch.Tensor, normalize_per_group: bool = True) -> torch.Tensor:
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
                # No novelty reward yet; rewards stay at 0.0
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
                sim = sim.masked_fill(~mask, float('-inf'))

                max_sim, _ = sim.max(dim=1)
                nearest_dist = 1.0 - max_sim
                rewards[indices] = torch.clamp(nearest_dist, min=0.0)

            # 3. Update the historical memory
            new_embs_detached = new_embs.detach().float().cpu()
            for emb_cpu in new_embs_detached:
                self.rollout_memory[k].append(emb_cpu)

        # 4. Optionally normalize rewards within each data_key group
        if normalize_per_group:
            for k, items in grouped_embs.items():
                indices = [x[0] for x in items]
                group_rewards = rewards[indices]
                max_reward = group_rewards.max()
                if max_reward > 1e-8:
                    group_rewards = group_rewards / max_reward
                    rewards[indices] = group_rewards

        return rewards

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

        Only updates counts for responses whose rewards are positive (called
        externally after filtering). All responses get a novelty score computed
        here; the positive_mask gating happens in the advantage function.
        """
        rewards = torch.zeros(len(token_id_lists), dtype=torch.float32)

        if not hasattr(self, "ngram_counts"):
            self.ngram_counts: dict[tuple[int, ...], Counter] = {}

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
        if not hasattr(self, "ngram_counts"):
            self.ngram_counts: dict[tuple[int, ...], Counter] = {}

        for i, (k, toks) in enumerate(zip(prompt_keys, token_id_lists, strict=True)):
            if positive_mask[i].item() <= 0:
                continue
            if k not in self.ngram_counts:
                self.ngram_counts[k] = Counter()
            if len(toks) < ngram_n:
                continue
            ngrams = [tuple(toks[j:j + ngram_n]) for j in range(len(toks) - ngram_n + 1)]
            self.ngram_counts[k].update(ngrams)

    def compute_policy_surprise_novelty(
        self,
        prompt_keys: list[tuple[int, ...]],
        seq_log_probs: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Policy-surprise novelty: reward correct solutions the policy finds unlikely.

        Within each prompt group's correct solutions, compute z-scored negative
        log-probability. Solutions with lower log-prob (more surprising to the
        current policy) get higher novelty bonus.

        Returns 0 for groups with fewer than 2 correct solutions (no comparison).
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

            # Negative z-score: lower log-prob (more surprising) → higher novelty
            surprise = -(correct_logprobs - mean_lp) / (std_lp + 1e-8)
            # Clamp to [0, inf) — only reward surprising solutions, don't penalize typical ones
            surprise = torch.clamp(surprise, min=0.0)

            for j, idx in enumerate(correct_indices):
                rewards[idx] = surprise[j]

        return rewards

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


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.curiosity_enabled = bool(self._cfg_get("algorithm.curiosity.enable", True))
        self.guided_resampling_enabled = bool(self._cfg_get("algorithm.guided_resampling.enable", True))
        rollout_n = int(self._cfg_get("actor_rollout_ref.rollout.n", 8))
        self.curiosity_memory = CuriosityMemory(
            pad_token_id=self.tokenizer.pad_token_id,
            max_rollouts_per_prompt=int(self._cfg_get("algorithm.curiosity.max_rollouts_per_prompt", 16)),
            max_prefixes_per_prompt=int(self._cfg_get("algorithm.curiosity.max_prefixes_per_prompt", 512)),
            min_history_before_reward=rollout_n,
        )
        self._guided_prompt_queue: list[list[int]] = []

        # Load embedding model. Try GPU for speed; fall back to CPU if driver
        # process has no CUDA access (common in Ray multi-node setups).
        if self.curiosity_enabled or self.guided_resampling_enabled:
            try:
                torch.cuda.init()
                _emb_device = 'cuda:0'
            except Exception:
                _emb_device = 'cpu'
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=_emb_device)
            self._emb_on_gpu = (_emb_device == 'cuda:0')
        else:
            self.embedding_model = None
            self._emb_on_gpu = False
        self._prefix_emb_cache: dict[int, torch.Tensor] = {}
        self._prefix_emb_cache_max = 100_000

    def _cfg_get(self, path: str, default: Any) -> Any:
        value = OmegaConf.select(self.config, path)
        return default if value is None else value

    def _prompt_key(self, token_ids: torch.Tensor) -> tuple[int, ...]:
        return self.curiosity_memory.key(token_ids)

    # def _compute_prefix_embeddings(
    #     self,
    #     prompt_key: tuple[int, ...],
    #     prefix_token_lists: list[list[int]],
    # ) -> torch.Tensor:
    #     if len(prefix_token_lists) == 0:
    #         return torch.empty(0)

    #     pad_id = self.tokenizer.pad_token_id
    #     n = len(prefix_token_lists)
    #     max_resp = max(len(x) for x in prefix_token_lists)
    #     max_prompt = len(prompt_key)
    #     max_total = max_prompt + max_resp

    #     input_ids = torch.full((n, max_total), pad_id, dtype=torch.long)
    #     attention_mask = torch.zeros((n, max_total), dtype=torch.long)
    #     position_ids = torch.zeros((n, max_total), dtype=torch.long)
    #     responses = torch.full((n, max_resp), pad_id, dtype=torch.long)
    #     answer_length = torch.full((n, max_resp), pad_id, dtype=torch.long)

    #     prompt_ids = list(prompt_key)
    #     for i, prefix_ids in enumerate(prefix_token_lists):
    #         total = prompt_ids + prefix_ids
    #         total_len = len(total)
    #         resp_len = len(prefix_ids)

    #         input_ids[i, :total_len] = torch.tensor(total, dtype=torch.long)
    #         attention_mask[i, :total_len] = 1
    #         position_ids[i, :total_len] = torch.arange(total_len, dtype=torch.long)
    #         responses[i, :resp_len] = torch.tensor(prefix_ids, dtype=torch.long)
    #         answer_length[i, :resp_len] = responses[i, :resp_len]

    #     prefix_batch = DataProto.from_dict(
    #         tensors={
    #             "input_ids": input_ids,
    #             "attention_mask": attention_mask,
    #             "position_ids": position_ids,
    #             "responses": responses,
    #             "answer_length": answer_length,
    #         }
    #     )
    #     # Pad to handle arbitrary prefix counts not divisible by DP chunk size
    #     size_divisor = self.actor_rollout_wg.world_size
    #     prefix_batch_padded, pad_size = pad_dataproto_to_divisor(prefix_batch, size_divisor)
        
    #     emb_outputs_padded = self.actor_rollout_wg.compute_embedding(prefix_batch_padded)
    #     emb_outputs = unpad_dataproto(emb_outputs_padded, pad_size=pad_size)

    #     return emb_outputs.batch["embeddings"].detach().cpu()

    def _update_prefix_memory(
        self,
        batch: DataProto,
        entropy_mat: torch.Tensor,
        hard_prompt_keys: list[tuple[int, ...]],
        prompt_keys: list[tuple[int, ...]] = None,
    ) -> None:
        """Store prefixes only for hard prompts (0% pass rate), picking the
        single lowest-entropy rollout per prompt to keep the workload small.

        Uses token-level newline detection to find prefix boundaries directly
        in token-ID space, avoiding expensive tokenizer.encode() calls.
        """
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if getattr(self, "global_steps", 0) <= warmup_steps:
            return

        if entropy_mat is None or len(hard_prompt_keys) == 0:
            return

        hard_set = set(hard_prompt_keys)

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]

        # --- Phase 0: for each hard prompt, pick the 1 rollout with lowest
        #     mean entropy (most confident trajectory) ---
        prompt_best: dict[tuple[int, ...], tuple[int, float]] = {}
        if prompt_keys is not None:
            for i, pk in enumerate(prompt_keys):
                if pk not in hard_set:
                    continue
                valid_len = int(response_mask[i].sum().item())
                if valid_len <= 0:
                    continue
                mean_ent = float(entropy_mat[i, :valid_len].mean().item())
                if pk not in prompt_best or mean_ent < prompt_best[pk][1]:
                    prompt_best[pk] = (i, mean_ent)
        else:
            prompts = batch.batch["prompts"]
            for i in range(prompts.shape[0]):
                pk = self._prompt_key(prompts[i])
                if pk not in hard_set:
                    continue
                valid_len = int(response_mask[i].sum().item())
                if valid_len <= 0:
                    continue
                mean_ent = float(entropy_mat[i, :valid_len].mean().item())
                if pk not in prompt_best or mean_ent < prompt_best[pk][1]:
                    prompt_best[pk] = (i, mean_ent)

        if len(prompt_best) == 0:
            return

        # --- Build newline token ID set (cached) ---
        if not hasattr(self, '_newline_token_ids'):
            nl_ids = set()
            for text in ['\n', '\n\n']:
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                nl_ids.update(encoded)
            # Also check single-char decode for common newline tokens
            for tid in range(self.tokenizer.vocab_size):
                if tid >= 1000:
                    break
                try:
                    decoded = self.tokenizer.decode([tid])
                    if decoded == '\n':
                        nl_ids.add(tid)
                except Exception:
                    pass
            self._newline_token_ids = nl_ids

        # --- Phase 1: extract prefixes using token-level split ---
        all_items: list[tuple[tuple[int, ...], list[int], float]] = []

        for pk, (idx, _) in prompt_best.items():
            valid_len = int(response_mask[idx].sum().item())
            response_ids = responses[idx][:valid_len]
            id_list = response_ids.cpu().tolist()

            # Find newline positions as split boundaries
            split_positions = []
            for pos, tid in enumerate(id_list):
                if tid in self._newline_token_ids:
                    split_positions.append(pos + 1)

            if len(split_positions) == 0:
                continue

            # Merge splits that are too close (< 32 tokens apart)
            merged = [split_positions[0]]
            for pos in split_positions[1:]:
                if pos - merged[-1] >= 32:
                    merged.append(pos)
            split_positions = merged

            # Take all but the last split as prefix boundaries (skip trailing)
            prefix_boundaries = split_positions[:-1] if len(split_positions) > 1 else split_positions

            # Limit to at most 15 prefixes per prompt (evenly spaced if more)
            if len(prefix_boundaries) > 15:
                step = len(prefix_boundaries) / 15
                prefix_boundaries = [prefix_boundaries[int(i * step)] for i in range(15)]

            for boundary in prefix_boundaries:
                prefix_ids = id_list[:boundary]
                if len(prefix_ids) == 0:
                    continue
                prefix_len = min(boundary, valid_len)
                prefix_entropy_mean = float(entropy_mat[idx, :prefix_len].mean().item())
                all_items.append((pk, prefix_ids, prefix_entropy_mean))

        if len(all_items) == 0:
            return

        # --- Phase 2: deduplicate and batch-encode for embeddings ---
        texts_to_encode: list[str] = []
        text_to_idx: dict[int, int] = {}
        item_hashes: list[int] = []

        for pk, prefix_ids, _ in all_items:
            h = hash(tuple(prefix_ids))
            item_hashes.append(h)
            if h not in self._prefix_emb_cache and h not in text_to_idx:
                text_to_idx[h] = len(texts_to_encode)
                texts_to_encode.append(self.tokenizer.decode(prefix_ids, skip_special_tokens=True))

        # --- Phase 3: single batched encode for novel prefixes ---
        if len(texts_to_encode) > 0:
            with torch.no_grad():
                if self._emb_on_gpu:
                    self.embedding_model.to('cuda:0')
                new_embeddings = self.embedding_model.encode(
                    texts_to_encode, convert_to_tensor=True, batch_size=512,
                ).cpu()
                if self._emb_on_gpu:
                    self.embedding_model.to('cpu')
                    torch.cuda.empty_cache()
            for text_hash, idx in text_to_idx.items():
                self._prefix_emb_cache[text_hash] = new_embeddings[idx]

        # --- Phase 4: scatter cached embeddings back and update memory ---
        for (pk, prefix_ids, prefix_entropy_mean), h in zip(all_items, item_hashes):
            emb = self._prefix_emb_cache[h]
            self.curiosity_memory.add_prefix_entry(pk, prefix_ids, prefix_entropy_mean, emb)

        # Evict oldest entries if cache is too large
        if len(self._prefix_emb_cache) > self._prefix_emb_cache_max:
            excess = len(self._prefix_emb_cache) - self._prefix_emb_cache_max
            keys_to_drop = list(self._prefix_emb_cache.keys())[:excess]
            for k in keys_to_drop:
                del self._prefix_emb_cache[k]

    def _hard_prompt_keys(self, prompt_keys: list[tuple[int, ...]], reward_tensor: torch.Tensor) -> list[tuple[int, ...]]:
        n_rollout = max(int(self.config.actor_rollout_ref.rollout.n), 1)
        threshold = 1.0 / n_rollout

        seq_scores = reward_tensor.sum(dim=-1)
        grouped_pass: dict[tuple[int, ...], list[float]] = defaultdict(list)

        for k, score in zip(prompt_keys, seq_scores.cpu().tolist()):
            grouped_pass[k].append(1.0 if score > 0.0 else 0.0)

        hard = []
        for k, vals in grouped_pass.items():
            pass_rate = float(np.mean(vals)) if len(vals) > 0 else 0.0
            if pass_rate <= threshold:
                hard.append(k)
        return hard

    def _enqueue_guided_prompts(self, hard_prompt_keys: list[tuple[int, ...]]) -> None:
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if getattr(self, "global_steps", 0) <= warmup_steps:
            return

        tau = float(self._cfg_get("algorithm.guided_resampling.tau", 0.1))
        for k in hard_prompt_keys:
            best = self.curiosity_memory.select_best_prefix(k, tau=tau)
            if best is None:
                continue
            guided_ids = list(k) + list(best["prefix_token_ids"])
            max_prompt_len = int(self._cfg_get("data.max_prompt_length", 2048))
            guided_ids = guided_ids[:max_prompt_len]
            if len(guided_ids) > 0:
                self._guided_prompt_queue.append(guided_ids)

        max_queue = int(self._cfg_get("algorithm.guided_resampling.max_queue_size", 4096))
        if len(self._guided_prompt_queue) > max_queue:
            self._guided_prompt_queue = self._guided_prompt_queue[-max_queue:]

    def _dequeue_guided_prompts_for_regen(self) -> list[list[int]]:
        warmup_steps = int(self._cfg_get("algorithm.guided_resampling.warmup_steps", 100))
        if not self.guided_resampling_enabled or getattr(self, "global_steps", 0) <= warmup_steps:
            return []

        if len(self._guided_prompt_queue) == 0:
            return []

        regen_batch_size = int(
            self._cfg_get(
                "algorithm.guided_resampling.regen_batch_size",
                self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            )
        )
        # Dequeue up to regen_batch_size, but don't block if fewer are available
        n = min(len(self._guided_prompt_queue), regen_batch_size)
        prompts = self._guided_prompt_queue[:n]
        self._guided_prompt_queue = self._guided_prompt_queue[n:]
        return prompts

    def _inject_guided_prompts_into_batch(self, batch: DataProto) -> int:
        """Replace tail of the dataloader batch with queued guided prompts.

        This avoids a second generation call by piggy-backing guided prompts
        onto the normal batch.  Returns the number of prompts replaced.
        """
        guided_prompt_lists = self._dequeue_guided_prompts_for_regen()
        if len(guided_prompt_lists) == 0:
            return 0

        batch_size = batch.batch["input_ids"].shape[0]
        n_replace = min(len(guided_prompt_lists), batch_size)
        # If we dequeued more than fits, re-queue the excess at the front
        if n_replace < len(guided_prompt_lists):
            self._guided_prompt_queue = guided_prompt_lists[n_replace:] + self._guided_prompt_queue
        guided_prompt_lists = guided_prompt_lists[:n_replace]

        # Build guided tensors with the same max_prompt_length as the batch
        pad_id = self.tokenizer.pad_token_id
        max_len = batch.batch["input_ids"].shape[1]

        for i, ids in enumerate(guided_prompt_lists):
            idx = batch_size - n_replace + i  # replace from the tail
            seq_len = min(len(ids), max_len)
            if seq_len == 0:
                continue
            # Zero out old content
            batch.batch["input_ids"][idx].fill_(pad_id)
            batch.batch["attention_mask"][idx].zero_()
            batch.batch["position_ids"][idx].zero_()
            # Left-pad: place tokens at the right end
            batch.batch["input_ids"][idx, max_len - seq_len:] = torch.tensor(
                ids[:seq_len], dtype=torch.long
            )
            batch.batch["attention_mask"][idx, max_len - seq_len:] = 1
            batch.batch["position_ids"][idx, max_len - seq_len:] = torch.arange(
                seq_len, dtype=torch.long
            )
            # Update raw_prompt_ids
            if "raw_prompt_ids" in batch.non_tensor_batch:
                batch.non_tensor_batch["raw_prompt_ids"][idx] = np.array(
                    ids[:seq_len], dtype=np.int64
                )
            # Clear text-form fields that don't apply to guided prompts
            if "raw_prompt" in batch.non_tensor_batch:
                batch.non_tensor_batch["raw_prompt"][idx] = None

        return n_replace

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    # TODO: support semantic entropy guided sampling in dataloader
    def _create_curiosity_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.total_samples_seen = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
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

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
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
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Inject queued guided-resampling prompts by replacing tail
                # of the dataloader batch (single generation call, no extra pass).
                n_guided_injected = self._inject_guided_prompts_into_batch(batch)
                if n_guided_injected > 0:
                    metrics["exploration/guided_injected"] = n_guided_injected

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
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

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        policy_entropys = entropys
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Compute and update per-prompt reward variance EMA
                        seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                        prompt_keys = [self._prompt_key(x) for x in batch.batch["prompts"]]
                        
                        prompt_to_rewards = defaultdict(list)
                        for k, r in zip(prompt_keys, seq_rewards.cpu().tolist()):
                            prompt_to_rewards[k].append(r)
                            
                        for k, r_list in prompt_to_rewards.items():
                            var = np.var(r_list) if len(r_list) > 1 else 0.0
                            if k in self.curiosity_memory.reward_variance_ema:
                                self.curiosity_memory.reward_variance_ema[k] = (
                                    (1 - self.curiosity_memory.ema_alpha) * self.curiosity_memory.reward_variance_ema[k] 
                                    + self.curiosity_memory.ema_alpha * var
                                )
                            else:
                                self.curiosity_memory.reward_variance_ema[k] = var

                        novelty_lambda = float(self._cfg_get("algorithm.curiosity.novelty_lambda", 1.0))
                        novelty_scale = float(self._cfg_get("algorithm.curiosity.novelty_reward_scale", 0.0))
                        novelty_after_norm = bool(self._cfg_get("algorithm.curiosity.novelty_after_norm", False))

                        if self.curiosity_enabled or self.guided_resampling_enabled:
                            with marked_timer("exploration", timing_raw, color="purple"):
                                entropy_mat = policy_entropys

                                if self.curiosity_enabled:
                                    novelty_metric = str(self._cfg_get("algorithm.curiosity.novelty_metric", "embedding"))

                                    responses = batch.batch["responses"]
                                    response_mask = batch.batch["response_mask"]
                                    valid_lengths = response_mask.sum(dim=-1).int()

                                    # Extract last-512 token ids for each response
                                    tail_id_lists = []
                                    for i in range(responses.shape[0]):
                                        vl = valid_lengths[i].item()
                                        tail_ids = responses[i][max(0, vl - 512):vl].tolist()
                                        tail_id_lists.append(tail_ids)

                                    if novelty_metric == "ngram_count":
                                        # Count-based exploration: n-gram visitation novelty
                                        ngram_n = int(self._cfg_get("algorithm.curiosity.novelty_ngram", 2))
                                        novelty_rewards = self.curiosity_memory.compute_ngram_count_novelty(
                                            prompt_keys, tail_id_lists, ngram_n=ngram_n,
                                        )
                                        # Update counts only for correct solutions (after computing rewards)
                                        seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                                        pos_mask = (seq_rewards > 0).float()
                                        self.curiosity_memory.update_ngram_counts(
                                            prompt_keys, tail_id_lists, pos_mask, ngram_n=ngram_n,
                                        )
                                        batch.batch["intrinsic_rewards"] = novelty_rewards.to(batch.batch["token_level_scores"].device)

                                    elif novelty_metric == "policy_surprise":
                                        # Policy-surprise: reward correct solutions the model finds unlikely
                                        old_log_probs = batch.batch["old_log_probs"]  # [batch, seq_len]
                                        seq_log_probs = (old_log_probs * response_mask).sum(dim=-1)  # [batch]
                                        seq_rewards = batch.batch["token_level_scores"].sum(dim=-1)
                                        pos_mask = (seq_rewards > 0).float()
                                        novelty_rewards = self.curiosity_memory.compute_policy_surprise_novelty(
                                            prompt_keys, seq_log_probs, pos_mask,
                                        )
                                        batch.batch["intrinsic_rewards"] = novelty_rewards.to(batch.batch["token_level_scores"].device)

                                    else:
                                        # Default: embedding-based cosine distance (original approach)
                                        rollout_texts = self.tokenizer.batch_decode(
                                            tail_id_lists, skip_special_tokens=True
                                        )

                                        with torch.no_grad():
                                            if self._emb_on_gpu:
                                                self.embedding_model.to('cuda:0')
                                            rollout_embeddings = self.embedding_model.encode(
                                                rollout_texts, convert_to_tensor=True, batch_size=512,
                                            ).cpu()
                                            if self._emb_on_gpu:
                                                self.embedding_model.to('cpu')
                                                torch.cuda.empty_cache()

                                        novelty_rewards = self.curiosity_memory.add_rollout_embeddings(
                                            prompt_keys, rollout_embeddings,
                                            normalize_per_group=(not novelty_after_norm),
                                        )

                                        if novelty_after_norm:
                                            batch.batch["intrinsic_rewards"] = novelty_rewards.to(batch.batch["token_level_scores"].device)
                                        else:
                                            # Option A (legacy): apply dynamic scaling
                                            dynamic_scales = []
                                            for k in prompt_keys:
                                                var_r = self.curiosity_memory.reward_variance_ema.get(k, 0.0)
                                                gamma_x = np.exp(-novelty_lambda * var_r)
                                                dynamic_scales.append(gamma_x)

                                            dynamic_scales_tensor = torch.tensor(dynamic_scales, device=novelty_rewards.device, dtype=novelty_rewards.dtype)
                                            batch.batch["intrinsic_rewards"] = (novelty_rewards * dynamic_scales_tensor).to(batch.batch["token_level_scores"].device)

                                    metrics["exploration/novelty_reward_mean"] = float(
                                        batch.batch["intrinsic_rewards"].mean().item()
                                    )

                                if self.guided_resampling_enabled:
                                    hard_keys = self._hard_prompt_keys(prompt_keys, reward_tensor)
                                    self._update_prefix_memory(batch, entropy_mat, hard_keys, prompt_keys)
                                    self._enqueue_guided_prompts(hard_keys)
                                    metrics["exploration/hard_prompt_count"] = len(hard_keys)
                                    metrics["exploration/guided_queue_size"] = len(self._guided_prompt_queue)
                                    if len(hard_keys) > 0:
                                        metrics["exploration/hard_passrate_threshold"] = 1.0 / max(
                                            int(self.config.actor_rollout_ref.rollout.n), 1
                                        )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        use_positive_novelty_adv = (
                            self.curiosity_enabled
                            and novelty_scale != 0.0
                            and "intrinsic_rewards" in batch.batch
                            and self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO
                        )

                        if use_positive_novelty_adv:
                            advantages, returns = core_algos.compute_grpo_outcome_advantage_with_positive_novelty(
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

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
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
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
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

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
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

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                    if hasattr(self.train_dataloader.sampler, 'weights'):
                        n_easy = (self.train_dataloader.sampler.weights < 0.5).sum().item()
                        metrics["curriculum/n_easy_downweighted"] = n_easy
                        metrics["curriculum/pct_active"] = 1.0 - n_easy / len(self.train_dataloader.sampler.weights)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
