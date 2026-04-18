# Copyright 2025 Amazon.com Inc and/or its affiliates
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
from abc import abstractmethod
from collections.abc import Sized

from omegaconf import DictConfig
from torch.utils.data import Sampler, WeightedRandomSampler
import torch
from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


class ProbabilisticCurriculumSampler(AbstractCurriculumSampler):
    """
    A sampler that samples data based on maintainable weights (probabilities).
    Weights are updated adaptively via the `update` method.
    """

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.num_samples = len(data_source)
        # Initialize weights uniformly (or load from checkpoint)
        self.weights = 5 * torch.ones(self.num_samples, dtype=torch.float32)
        # self.weights = torch.ones(self.num_samples, dtype=torch.float32) # this is bad
        
        self.generator = torch.Generator()
        self.generator.manual_seed(data_config.get("seed", 1))
        self.replacement = True
        # Hyperparameters for update logic
        self.temperature = data_config.get("sampler_temperature", 1.0)

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples

    def update(self, batch: DataProto) -> None:
        """
        Update sampling weights based on batch metrics (e.g., semantic entropy).
        Requires 'index' in batch.non_tensor_batch to map back to source.
        """
        # 1. Get original dataset indices
        if "index" not in batch.non_tensor_batch:
            # Warning: Cannot update without indices
            return

        indices = batch.non_tensor_batch["index"]
        
        metrics = batch.batch.get("intrinsic_var", None)
        
        if metrics is None:
            return
        metrics = metrics.to("cpu").float()
        # Some indices are dupilicated, so we take the variance of the metrics for the same index
        index_metric_dict = {}
        for idx, metric in zip(indices, metrics):
            if idx not in index_metric_dict:
                index_metric_dict[idx] = []
            index_metric_dict[idx].append(metric.item())
        # Get the variance for each index and update the weights
        for idx, metric_list in index_metric_dict.items():
            if len(metric_list) > 1:
                metric_tensor = torch.tensor(metric_list)
                variance = torch.var(metric_tensor)
            else:
                variance = torch.tensor(0.0)
            # Update weight using exponential of variance
            self.weights[idx] += variance

        # for idx, metric in zip(indices, metrics):
        #     self.weights[idx] = torch.exp(metric / self.temperature)

    def state_dict(self):
        return {
            "weights": self.weights,
            "rng_state": self.generator.get_state()
        }

    def load_state_dict(self, state_dict):
        self.weights = state_dict["weights"]
        self.generator.set_state(state_dict["rng_state"])