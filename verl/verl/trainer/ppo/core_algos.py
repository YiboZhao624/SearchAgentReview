# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"
    OPTIMAL_TOKEN_BASELINE = "optimal_token_baseline"
    TIR_OPTIMAL_TOKEN_BASELINE = "tir_optimal_token_baseline"
    GRPO_STEP_PAST_AVG = "grpo_step_past_avg"
    GRPO_STEP_RTG = "grpo_step_rtg"
    GRPO_WHITENED = "grpo_whitened"
    GRPO_LEX_RANK = "grpo_lex_rank"
    GRPO_STEP_RESPONSIBILITY = "grpo_step_responsibility"
    GRPO_GIGPO = "grpo_gigpo"
    GRPO_IGPO = "grpo_igpo"
    TREE_GRPO = "tree_grpo"
    GRPO_CALIBADV = "grpo_calibadv"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


@register_adv_est(AdvantageEstimator.OPTIMAL_TOKEN_BASELINE)
def compute_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = False,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        batch_size, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # Compute returns (reward-to-go) for each timestep
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(batch_size):
            prompt_groups[index[i]].append(i)

        # Initialize baselines tensor [batch_size, seq_len]
        baselines = torch.zeros_like(returns)

        # Compute per-step baseline for each prompt group
        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            if N == 1:
                # Single trajectory - no baseline (advantage = return)
                continue

            traj_idx = torch.tensor(trajectory_indices, device=device)

            # Extract group data [N, seq_len]
            returns_group = returns[traj_idx]
            w_cumulative_group = w_cumulative[traj_idx]
            mask_group = response_mask[traj_idx]

            # Compute per-timestep baseline: B_t = Σ[G_t × W_t] / Σ[W_t]
            # where W_t = Σ_{j=1}^t ||s_j||² (cumulative path variance)
            # Shape: [seq_len]
            numerator = (returns_group * w_cumulative_group * mask_group).sum(dim=0)  # Sum over trajectories
            denominator = (w_cumulative_group * mask_group).sum(dim=0) + epsilon

            baseline_per_step = numerator / denominator  # [seq_len]

            # Assign to all trajectories in this group
            baselines[traj_idx] = baseline_per_step.unsqueeze(0).expand(N, -1)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths = mask_group.sum(dim=-1)
                sorted_lengths, _ = torch.sort(response_lengths)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages: A_t = G_t - B_t
        advantages = (returns - baselines) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE)
def compute_multi_turn_optimal_token_baseline_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    old_log_probs: torch.Tensor,
    sum_pi_squared: torch.Tensor,
    rollout_is_weights: torch.Tensor = None,
    handle_zero_tail: bool = True,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantages using Optimal Token Baseline (OTB).

    Unlike the group mean based baseline which uses a single baseline per trajectory,
    this computes a unique baseline for each timestep using cumulative path variance.

    Theory:
        For each timestep t in each prompt group:
            B_t* = E[G_t × W_t] / E[W_t]
        where W_t = Σ_{j=1}^t ||s_j||² (cumulative path-variance proxy)
        and ||s_j||² = 1 - 2π_j + Σπ²

    The cumulative sum W_t captures the "realized energy" of trajectory has been up to timestep t,
    giving higher weight to predicting rewards on high-variance paths.

    Args:
        token_level_rewards: Rewards at each token position [shape: (bs, response_length)]
        response_mask: Binary mask for valid tokens (1) vs padding (0) [shape: (bs, response_length)]
        index: Prompt indices for grouping trajectories from same prompt [shape: (bs,)]
        old_log_probs: Log probabilities from training policy during generation [shape: (bs, response_length)]
        sum_pi_squared: Sum of squared probabilities over vocabulary Σπ² [shape: (bs, response_length)]
        rollout_is_weights: Pre-computed IS weights for W correction [shape: (bs, response_length)],
            None if not using IS
        handle_zero_tail: If True, zero baselines will be set in the portion of the longest trajectory
            that extends beyond the second-longest trajectory in the prompt group.
            Default: False
        epsilon: Small constant for numerical stability (default: 1e-8)

    Returns:
        advantages: OTB advantage estimates [shape: (bs, response_length)]
        returns: Cumulative rewards (returns) from each position [shape: (bs, response_length)]

    Note on Rollout Importance Sampling:
        When rollout_is_weights is provided, W_t is scaled by ρ̄²(t) to minimize MSE under truncated IS:
            B_t* = Σ[G_t × ρ̄²(t) × W_t] / Σ[ρ̄²(t) × W_t]
    """
    with torch.no_grad():
        # Compute returns (reward-to-go) for each timestep
        token_returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        # Step 1: Compute w_per_timestep = 1 - 2π_t + Σπ²)
        pi_t = torch.exp(old_log_probs)
        w_per_timestep = 1 - 2 * pi_t + sum_pi_squared

        # Step 2: Apply rollout importance sampling correction (if enabled)
        if rollout_is_weights is not None:
            # Scale W by ρ̄² to minimize MSE under truncated IS
            w_per_timestep = w_per_timestep * (rollout_is_weights**2)

        # Step 3: Compute cumulative path-variance proxy: W_t = Σ_{j=1}^t w_j
        # This measures accumulated variance from the start of the trajectory up to timestep t
        w_cumulative = (w_per_timestep * response_mask).cumsum(dim=-1)

        # Step 4: Concatenate returns and w_cumulative for each trajectory
        # This allows us to compute baseline per timestep for each trajectory
        response_lengths = response_mask.sum(dim=-1).to(dtype=torch.long)  # [shape: (bs * n, )]
        max_response_length = int(response_lengths.max().item()) if response_lengths.numel() > 0 else 0
        all_w_values = w_cumulative.new_zeros(
            (len(response_lengths), max_response_length)
        )  # [shape: (bs * n, max_response_length)]
        all_returns = torch.zeros_like(all_w_values)
        for i in range(len(response_lengths)):
            length = int(response_lengths[i].item())
            if length == 0:
                continue
            mask = response_mask[i].bool()
            all_w_values[i, :length] = w_cumulative[i, mask]
            all_returns[i, :length] = token_returns[i, mask]

        # Group trajectories by prompt
        prompt_groups = defaultdict(list)
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            prompt_groups[index[i]].append(i)

        # Compute optimal baseline for each prompt group
        baselines = torch.zeros_like(all_returns)

        for _, trajectory_indices in prompt_groups.items():
            N = len(trajectory_indices)
            traj_idx = torch.tensor(trajectory_indices, device=all_returns.device)

            if N == 1:
                # Single trajectory - no baseline (keep original reward as advantage)
                baselines[traj_idx[0]] = 0.0
                continue

            # Extract group data
            w_group = all_w_values[traj_idx]  # [shape: (N, max_response_length)]
            R_group = all_returns[traj_idx]  # [shape: (N, max_response_length)]
            # Direct optimal baseline - single value for all in group
            b_star = (R_group * w_group).sum(dim=0) / (w_group.sum(dim=0) + epsilon)
            # Convert to match baselines dtype (epsilon can cause float64 promotion)
            baselines[traj_idx] = b_star.to(baselines.dtype)

            if handle_zero_tail:
                # Optionally zero out the portion of the longest trajectory that extends
                # beyond the second-longest trajectory in the prompt group.
                response_lengths_group = response_lengths[traj_idx]
                sorted_lengths, _ = torch.sort(response_lengths_group)
                max_length = int(sorted_lengths[-1].item())
                second_max_length = int(sorted_lengths[-2].item())
                max_length_idx = (response_lengths_group == max_length).nonzero(as_tuple=True)[0]
                if max_length_idx.numel() == 1 and max_length > second_max_length:
                    max_length_traj_idx = trajectory_indices[int(max_length_idx[0])]
                    baselines[max_length_traj_idx, second_max_length:] = 0.0

        # Compute advantages
        all_advantages = all_returns - baselines  # [shape: (bs * n, max_response_length)]

        advantages = torch.zeros_like(token_returns)  # [shape: (bs * n, turn * response_length)]
        for i in range(len(response_lengths)):
            if response_lengths[i] == 0:
                continue
            advantages[i, response_mask[i].bool()] = all_advantages[i, : response_lengths[i]]

        advantages = advantages * response_mask  # [shape: (bs * n * turn, response_length)]

    return advantages, token_returns


def _get_step_boundaries(response_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Extract per-sample assistant-turn boundaries from response_mask.

    Args:
        response_mask: (bs, response_length), 1 for LLM-generated tokens, 0 for tool/padding.

    Returns:
        List of per-sample lists of (start, end_inclusive) index pairs, one pair per turn.
    """
    mask_np = response_mask.cpu().numpy().astype(np.int32)
    result = []
    for row in mask_np:
        bounds = []
        in_block = False
        start = 0
        for t, m in enumerate(row):
            if m == 1 and not in_block:
                start = t
                in_block = True
            elif m == 0 and in_block:
                bounds.append((start, t - 1))
                in_block = False
        if in_block:
            bounds.append((start, len(row) - 1))
        result.append(bounds)
    return result


@register_adv_est(AdvantageEstimator.GRPO_STEP_PAST_AVG)
def compute_grpo_step_past_avg_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_scores: list[list[float]],
    outcome_scores: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step-level advantage with within-trajectory past-average baseline.

    For step t in trajectory i:
      - t == 0: baseline = group mean of first-step scores
      - t > 0:  baseline = mean of r_s^{i,0..t-1}  (within-trajectory history)
    The outcome reward is group-normalised and added to the last step.

    Args:
        token_level_rewards: (bs, response_length) — used only for device/dtype.
        response_mask:       (bs, response_length) — 1 for assistant tokens.
        index:               (bs,) group ids for GRPO normalisation.
        step_scores:         list[list[float]] — per-sample ordered step scores.
        outcome_scores:      (bs,) outcome scores (e.g. EM/F1).
        epsilon:             numerical stability.
        norm_adv_by_std_in_grpo: scale by group std.

    Returns:
        advantages, returns — both (bs, response_length).
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        advantages = torch.zeros(bs, resp_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        step_boundaries = _get_step_boundaries(response_mask)

        # Group mean of first-step scores (baseline for t=0).
        id2first = defaultdict(list)
        for i in range(bs):
            scores_i = step_scores[i] if step_scores[i] else []
            if scores_i:
                id2first[index[i]].append(scores_i[0])
        id2first_mean = {idx: float(np.mean(v)) for idx, v in id2first.items()}

        # Group mean/std of outcome scores (for last-step outcome bonus).
        id2outcome = defaultdict(list)
        for i in range(bs):
            id2outcome[index[i]].append(float(outcome_scores[i]))
        id2outcome_mean = {idx: float(np.mean(v)) for idx, v in id2outcome.items()}
        id2outcome_std = {idx: float(np.std(v)) if len(v) > 1 else 1.0 for idx, v in id2outcome.items()}

        for i in range(bs):
            bounds = step_boundaries[i]
            scores_i = step_scores[i] if step_scores[i] else []
            n_steps = len(bounds)
            for t, (start, end) in enumerate(bounds):
                # Step-level advantage relative to baseline.
                if t < len(scores_i):
                    r_t = scores_i[t]
                    if t == 0:
                        b_t = id2first_mean.get(index[i], 0.0)
                    else:
                        b_t = float(np.mean(scores_i[:t]))
                    a_t = r_t - b_t
                else:
                    a_t = 0.0

                # Add group-normalised outcome reward to the last step.
                if t == n_steps - 1:
                    g = index[i]
                    o_mean = id2outcome_mean.get(g, 0.0)
                    o_std = id2outcome_std.get(g, 1.0)
                    if norm_adv_by_std_in_grpo:
                        outcome_adv = (float(outcome_scores[i]) - o_mean) / (o_std + epsilon)
                    else:
                        outcome_adv = float(outcome_scores[i]) - o_mean
                    a_t = a_t + outcome_adv

                advantages[i, start : end + 1] = a_t

        advantages = advantages * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_STEP_RTG)
def compute_grpo_step_rtg_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_scores: list[list[float]],
    outcome_scores: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Step-level advantage using reward-to-go, group-normalised per step.

    For step t in trajectory i:
      G_{i,t} = sum_{t' >= t} r_s^{i,t'} + r_o^i

    G_{i,t} is group-normalised across trajectories that have at least t+1 steps.
    Trajectories with fewer steps use their available RTG (already includes outcome).

    Args:
        token_level_rewards: (bs, response_length) — used only for device/dtype.
        response_mask:       (bs, response_length) — 1 for assistant tokens.
        index:               (bs,) group ids for GRPO normalisation.
        step_scores:         list[list[float]] — per-sample ordered step scores.
        outcome_scores:      (bs,) outcome scores (e.g. EM/F1).
        epsilon:             numerical stability.
        norm_adv_by_std_in_grpo: scale by group std.

    Returns:
        advantages, returns — both (bs, response_length).
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        advantages = torch.zeros(bs, resp_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        step_boundaries = _get_step_boundaries(response_mask)

        # Pre-compute reward-to-go per sample per step.
        rtg: list[list[float]] = []
        for i in range(bs):
            scores_i = step_scores[i] if step_scores[i] else []
            r_o = float(outcome_scores[i])
            rtg_i = []
            running = r_o
            for t in reversed(range(len(scores_i))):
                running = scores_i[t] + running
                rtg_i.append(running)
            rtg_i.reverse()  # now rtg_i[t] = sum_{t'>=t} r_s[t'] + r_o
            if not rtg_i:
                rtg_i = [r_o]  # no steps: just outcome reward
            rtg.append(rtg_i)

        # Find max number of steps across batch.
        max_steps = max(len(bounds) for bounds in step_boundaries) if step_boundaries else 1

        # Group-normalise RTG at each step level.
        for step_t in range(max_steps):
            id2rtg_t: dict[str, list[float]] = defaultdict(list)
            for i in range(bs):
                bounds = step_boundaries[i]
                rtg_i = rtg[i]
                # Use step_t RTG if available, else use the last available RTG.
                t_idx = min(step_t, len(rtg_i) - 1)
                id2rtg_t[index[i]].append(rtg_i[t_idx])

            id2mean_t = {idx: float(np.mean(v)) for idx, v in id2rtg_t.items()}
            id2std_t = {idx: float(np.std(v)) if len(v) > 1 else 1.0 for idx, v in id2rtg_t.items()}

            for i in range(bs):
                bounds = step_boundaries[i]
                if step_t >= len(bounds):
                    continue
                start, end = bounds[step_t]
                rtg_i = rtg[i]
                t_idx = min(step_t, len(rtg_i) - 1)
                g = index[i]
                if norm_adv_by_std_in_grpo:
                    a_t = (rtg_i[t_idx] - id2mean_t[g]) / (id2std_t[g] + epsilon)
                else:
                    a_t = rtg_i[t_idx] - id2mean_t[g]
                advantages[i, start : end + 1] = a_t

        advantages = advantages * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_WHITENED)
def compute_grpo_whitened_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    outcome_scores: np.ndarray,
    step_means: np.ndarray,
    outcome_weight: float = 0.7,
    step_weight: float = 0.3,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO with in-group whitened outcome and step advantages.

    For each group:
      outcome_adv_i = (outcome_i - mean(outcome_group)) / (std(outcome_group) + eps)
      step_adv_i    = (step_i - mean(step_group)) / (std(step_group) + eps)
      advantage_i   = w_out * outcome_adv_i + w_step * step_adv_i

    Then broadcast to token level: advantages[i] = advantage_i * response_mask[i]
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        advantages = torch.zeros(bs, resp_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)

        # Normalise weights
        w_total = outcome_weight + step_weight
        if w_total <= 0:
            w_out, w_step = 0.7, 0.3
        else:
            w_out, w_step = outcome_weight / w_total, step_weight / w_total

        # Group stats
        id2outcome: dict[Any, list[float]] = defaultdict(list)
        id2step: dict[Any, list[float]] = defaultdict(list)
        id2indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(bs):
            g = index[i]
            id2outcome[g].append(float(outcome_scores[i]))
            id2step[g].append(float(step_means[i]))
            id2indices[g].append(i)

        for g, indices in id2indices.items():
            o_vals = id2outcome[g]
            s_vals = id2step[g]

            o_mean = float(np.mean(o_vals))
            o_std = float(np.std(o_vals)) if len(o_vals) > 1 else 1.0
            s_mean = float(np.mean(s_vals))
            s_std = float(np.std(s_vals)) if len(s_vals) > 1 else 1.0

            for j, i in enumerate(indices):
                if norm_adv_by_std_in_grpo:
                    o_adv = (o_vals[j] - o_mean) / (o_std + epsilon)
                    s_adv = (s_vals[j] - s_mean) / (s_std + epsilon)
                else:
                    o_adv = o_vals[j] - o_mean
                    s_adv = s_vals[j] - s_mean

                adv_i = w_out * o_adv + w_step * s_adv
                advantages[i] = adv_i * response_mask[i]

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_LEX_RANK)
def compute_grpo_lex_rank_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    outcome_scores: np.ndarray,
    step_means: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO with lexicographic rank advantage: sort by (outcome, step_mean), use rank as advantage.

    For each group:
      1. Sort trajectories by (outcome_score, step_mean) ascending.
      2. Assign ranks 1..N (ties get average rank).
      3. Normalize: adv_i = (rank_i - mean_rank) / (std_rank + eps)
      4. Broadcast to token level.

    This eliminates sign-flip risk and only uses the ordering (not magnitude) of LLM judge scores.
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        advantages = torch.zeros(bs, resp_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)

        # Group trajectories by prompt index
        id2indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(bs):
            id2indices[index[i]].append(i)

        for g, indices in id2indices.items():
            n = len(indices)
            if n <= 1:
                continue

            # Partition into success / failure subgroups by outcome
            # Use group mean as threshold (matches standard GRPO sign semantics)
            outcomes = np.array([float(outcome_scores[i]) for i in indices])
            steps = np.array([float(step_means[i]) for i in indices])
            outcome_mean = outcomes.mean()

            success_idx = [j for j in range(n) if outcomes[j] > outcome_mean]
            failure_idx = [j for j in range(n) if outcomes[j] < outcome_mean]
            tie_idx = [j for j in range(n) if outcomes[j] == outcome_mean]

            # All outcomes identical → rank purely by step_mean, centered at 0
            if not success_idx and not failure_idx:
                # All outcomes identical → rank purely by step_mean, centered at 0
                order = np.argsort(steps)
                ranks = np.empty(n, dtype=np.float64)
                ranks[order] = np.linspace(-1.0, 1.0, n)
                for j, i in enumerate(indices):
                    adv_i = ranks[j]
                    if norm_adv_by_std_in_grpo:
                        r_std = ranks.std()
                        adv_i = adv_i / (r_std + epsilon)
                    advantages[i] = float(adv_i) * response_mask[i]
                continue

            # Ties at mean: assign to whichever side is smaller, break further ties by step_mean
            tie_sorted = sorted(tie_idx, key=lambda j: steps[j])
            for j in tie_sorted:
                if len(failure_idx) <= len(success_idx):
                    failure_idx.append(j)
                else:
                    success_idx.append(j)

            # Within each subgroup: rank by step_mean, map to designated range
            # Failure → [-1, 0), Success → (0, +1]
            def _subgroup_advantages(sub_idx: list[int], low: float, high: float) -> dict[int, float]:
                """Rank within subgroup by step_mean, map to [low, high] linearly."""
                if not sub_idx:
                    return {}
                if len(sub_idx) == 1:
                    return {sub_idx[0]: (low + high) / 2.0}
                sub_steps = np.array([steps[j] for j in sub_idx])
                order = np.argsort(sub_steps)
                advs = {}
                for rank_pos, j_in_order in enumerate(order):
                    # rank_pos: 0 .. len-1 → map to [low, high]
                    t = rank_pos / (len(sub_idx) - 1)  # 0 to 1
                    advs[sub_idx[j_in_order]] = low + t * (high - low)
                return advs

            adv_map = {}
            adv_map.update(_subgroup_advantages(failure_idx, -1.0, -epsilon))
            adv_map.update(_subgroup_advantages(success_idx, epsilon, 1.0))

            # Optional: normalize by std for consistency with other GRPO estimators
            all_advs = np.array([adv_map[j] for j in range(n)])
            if norm_adv_by_std_in_grpo:
                a_std = all_advs.std()
                if a_std > epsilon:
                    all_advs = all_advs / a_std

            for j, i in enumerate(indices):
                advantages[i] = float(all_advs[j]) * response_mask[i]

    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_STEP_RESPONSIBILITY)
def compute_grpo_step_responsibility_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_scores: list[list[float]],
    outcome_scores: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    responsibility_min: float = 0.2,
    responsibility_max: float = 1.0,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO with per-step responsibility modulation (Scheme F).

    Outcome determines advantage **direction**, per-step quality determines **magnitude**.

    Algorithm:
      1. Compute standard GRPO trajectory-level advantage from outcome_scores.
      2. For each step, normalize its score to q ∈ [0, 1] (group-level min-max).
      3. Compute responsibility:
         - adv >= 0 (success): responsibility = q   (good step → high credit)
         - adv <  0 (failure): responsibility = 1-q  (good step → low blame)
      4. Clamp responsibility to [responsibility_min, responsibility_max].
      5. token_advantage = trajectory_adv × responsibility_t

    Properties:
      - Sign flip impossible: direction fully determined by outcome.
      - Within-trajectory credit assignment: different steps get different magnitudes.
      - Robust to LLM judge noise: clamp bounds worst-case effect.
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        advantages = torch.zeros(bs, resp_len, dtype=token_level_rewards.dtype, device=token_level_rewards.device)
        step_boundaries = _get_step_boundaries(response_mask)

        # --- Step 1: Standard GRPO trajectory-level advantage from outcome ---
        id2outcome: dict[Any, list[float]] = defaultdict(list)
        id2indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(bs):
            g = index[i]
            id2outcome[g].append(float(outcome_scores[i]))
            id2indices[g].append(i)

        traj_adv = np.zeros(bs, dtype=np.float64)
        for g, indices in id2indices.items():
            o_vals = np.array(id2outcome[g])
            o_mean = o_vals.mean()
            o_std = o_vals.std() if len(o_vals) > 1 else 1.0
            for j, i in enumerate(indices):
                if norm_adv_by_std_in_grpo:
                    traj_adv[i] = (o_vals[j] - o_mean) / (o_std + epsilon)
                else:
                    traj_adv[i] = o_vals[j] - o_mean

        # --- Step 2: Collect all step scores in each group for min-max normalization ---
        id2all_step_scores: dict[Any, list[float]] = defaultdict(list)
        for i in range(bs):
            g = index[i]
            for s in (step_scores[i] if step_scores[i] else []):
                id2all_step_scores[g].append(float(s))

        id2step_min: dict[Any, float] = {}
        id2step_max: dict[Any, float] = {}
        for g, all_s in id2all_step_scores.items():
            id2step_min[g] = min(all_s) if all_s else 0.0
            id2step_max[g] = max(all_s) if all_s else 1.0

        # --- Step 3-5: Per-step responsibility modulation ---
        for i in range(bs):
            g = index[i]
            bounds = step_boundaries[i]
            scores_i = step_scores[i] if step_scores[i] else []
            s_min = id2step_min.get(g, 0.0)
            s_max = id2step_max.get(g, 1.0)
            s_range = s_max - s_min

            a_i = traj_adv[i]

            for t, (start, end) in enumerate(bounds):
                if t < len(scores_i):
                    # Normalize step score to [0, 1] via group min-max
                    if s_range > epsilon:
                        q = (float(scores_i[t]) - s_min) / s_range
                    else:
                        q = 0.5  # All scores identical → neutral

                    # Responsibility: credit (success) vs blame (failure)
                    if a_i >= 0:
                        resp = q
                    else:
                        resp = 1.0 - q

                    # Clamp to bound worst-case effect
                    resp = max(responsibility_min, min(responsibility_max, resp))
                else:
                    # No score for this step (e.g., final answer turn) → neutral
                    resp = (responsibility_min + responsibility_max) / 2.0

                advantages[i, start : end + 1] = float(a_i * resp)

        advantages = advantages * response_mask
    return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_GIGPO)
def compute_grpo_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    anchor_obs_per_traj=None,
    step_scores_per_traj=None,
    gamma: float = 0.9,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    similarity_method: str = "sequence_matcher",
    embedding_service_url: str = "http://localhost:8000",
    compute_mean_std_cross_steps: bool = True,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GiGPO advantage estimator (https://arxiv.org/abs/2505.10978), adapted for
    per-trajectory batch format. See verl/trainer/ppo/core_gigpo.py for details."""
    from verl.trainer.ppo.core_gigpo import compute_grpo_gigpo_advantage as _impl
    return _impl(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        anchor_obs_per_traj=anchor_obs_per_traj,
        step_scores_per_traj=step_scores_per_traj,
        gamma=gamma,
        step_advantage_w=step_advantage_w,
        mode=mode,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        similarity_method=similarity_method,
        embedding_service_url=embedding_service_url,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("sapo")
def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the smoothed policy objective and related metrics for SAPO.

    See https://arxiv.org/pdf/2511.20347 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For SAPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)

    # temperature for positive and negative token updates
    tau_pos = torch.as_tensor(config.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        """The gating function used in SAPO"""
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    # compute IS at token level:
    # r_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t})]
    # In log space: log(r_{i,t}(θ)) = log_prob - ol_log_prob
    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    # finally exp() to remove log and get r_{i,t}(θ)
    ratio = torch.exp(negative_approx_kl)

    # tau_{i,t} is tau_pos if adv > 0 else tau_neg
    taus = torch.where(
        condition=advantages > 0,
        input=tau_pos,  # if A_{i,t} > 0 we set to tau_pos
        other=tau_neg,  # if A_{i,t} <= 0 we set to tau_neg
    )

    # compute the gates f_{i,t}(r_{i,t}(θ)) at token level
    gates = gate_function(ratio, taus)

    # compute policy gradient loss
    pg_losses = -gates * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for SAPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for both pg_clipfrac and pg_clipfrac_lower (not used in SAPO)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    # compute KL for metrics tracking
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # return metrics dict
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("cispo")
def compute_policy_loss_cispo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    # Compute importance sampling ratio: π_θ / π_θ_old
    negative_approx_kl = log_prob - old_log_prob
    # Clamp for numerical stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # CISPO: Clip the importance sampling weights
    # KEY: Apply stop gradient to the clipped ratio
    # This prevents gradients from flowing through the ratio computation and clipping
    # Gradients only flow through log_prob in the final loss term
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()

    # CISPO objective function (to maximize): J = sg(clip(ratio)) * A * log π_θ
    # Loss function (to minimize): L = -J = -sg(clip(ratio)) * A * log_prob
    pg_losses = -clipped_ratio_sg * advantages * log_prob

    # Track clipping statistics
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)

    # Apply rollout importance sampling weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in CISPO)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_reinforce(
    rollout_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute REINFORCE-style policy gradient loss with optional IS correction.

    This function implements policy gradient (REINFORCE) with optional importance
    sampling correction for rollout-training policy mismatch.

    Mathematical formulation:
        Without IS (rollout_is_weights=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is_weights provided):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length). Used for KL computation.
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        response_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length). Should already include rejection sampling.
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        config: Actor config (required for global_batch_info).
        rollout_is_weights: Pre-computed IS weights (π_current / π_rollout).
            Shape: (batch_size, seq_length). None to disable IS correction.

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy gradient loss
            metrics: Dictionary with "actor/ppo_kl"

    Note:
        Unlike PPO (compute_policy_loss_vanilla), this function:
        - Does NOT use PPO clipping
        - Uses log π(a|s) directly (not ratio)
        - IS weights are applied as multiplicative factor
    """
    assert config is not None, "ActorConfig must be provided for REINFORCE loss"

    # Compute pure policy gradient loss with optional IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_metrics = {
        "actor/ppo_kl": kl_divergence.detach().item(),
    }

    return pg_loss, pg_metrics


@register_policy_loss("bypass_mode")
def compute_policy_loss_bypass_mode(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Bypass mode policy loss supporting both REINFORCE and PPO-clip.

    This function is the entry point for bypass mode, where old_log_prob = rollout_log_prob.
    It computes IS weights and rejection masks, then dispatches to either REINFORCE or
    PPO-clip loss based on the loss_type configuration.

    IMPORTANT - Bypass mode semantics:
        In bypass mode, the trainer sets old_log_prob = rollout_log_prob.
        This means:
        - For REINFORCE: We use IS weights w = π_current / π_rollout explicitly
        - For PPO-clip: The PPO ratio π_current / π_old = π_current / π_rollout
          already incorporates the IS correction through clipping, so we do NOT
          apply additional IS weights (would be double-counting)

    Loss types:
        - "ppo_clip" (default): PPO clipped objective (compute_policy_loss_vanilla)
            L = -E[min(r*A, clip(r)*A)] where r = π_current / π_rollout
            Note: IS weights are NOT applied (clipping handles the ratio)
        - "reinforce": REINFORCE-style policy gradient with IS correction
            L = -E[w * log π(a|s) * A] where w = π_current / π_rollout

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob.
            Shape: (batch_size, seq_length)
        log_prob: Current policy log probabilities.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates.
            Shape: (batch_size, seq_length)
        response_mask: Valid token mask (1=valid, 0=padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation mode (passed to underlying loss function).
        config: Actor config containing rollout_correction settings in policy_loss.
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally).

    Config options (in config.policy_loss.rollout_correction):
        loss_type: "ppo_clip" (default) or "reinforce"
        rollout_is: IS aggregation level ("token", "sequence", or None)
        rollout_is_threshold: Upper threshold for truncating IS weights (default: 2.0)
        rollout_rs: Rejection sampling level (see rollout_corr_helper for supported modes)
        rollout_rs_threshold: Threshold specification for rejection sampling
        rollout_is_batch_normalize: Whether to normalize IS weights to mean=1.0

    Returns:
        Tuple of (loss, metrics):
            loss: Scalar policy loss
            metrics: Dictionary with rollout correction metrics and actor/ppo_kl
    """
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "config is required for bypass_mode loss"

    # Extract rollout_correction config from policy_loss
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='bypass_mode', ensure rollout_correction config is passed."
        )

    # Extract parameters
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip")
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    # In bypass mode: old_log_prob IS rollout_log_prob
    rollout_log_prob = old_log_prob

    # Compute IS weights and rejection mask
    # Note: For PPO-clip, we still compute IS weights for metrics, but don't apply them
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy (for IS ratio: π_current / π_rollout)
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=response_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
            )
        )

    # Extract IS weights tensor (or None if disabled)
    computed_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (RS + veto)
    effective_mask = modified_response_mask

    # Dispatch to appropriate loss function based on loss_type
    if loss_type == "reinforce":
        # REINFORCE: Apply IS weights explicitly
        pg_loss, pg_metrics = compute_policy_loss_reinforce(
            rollout_log_prob=rollout_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=computed_is_weights,
        )

    elif loss_type == "ppo_clip":
        # PPO-clip: The ratio π_current/π_old = π_current/π_rollout already handles IS
        # DO NOT apply IS weights - would be double-counting!
        # The clipping mechanism constrains the effective IS ratio
        pg_loss, pg_metrics = compute_policy_loss_vanilla(  # type: ignore[call-arg]
            old_log_prob=rollout_log_prob,  # = old_log_prob in bypass mode
            log_prob=log_prob,
            advantages=advantages,
            response_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=None,  # Explicitly None - no IS weights for PPO-clip
        )

    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'reinforce' or 'ppo_clip'.")

    # Merge rollout correction metrics
    pg_metrics.update(rollout_metrics)

    return pg_loss, pg_metrics


# ---------------------------------------------------------------------------
# IGPO: Turn-level discounted advantage
# ---------------------------------------------------------------------------

def _compute_turn_level_advantage(
    normalized_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    bsz: int,
    seq_len: int,
    device: torch.device,
    turn_boundary_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Turn-level discounted accumulation + broadcast.

    Each turn is defined by reward position (non-zero reward marks end of turn).

    Steps:
        1. Identify turn boundaries per sample.
        2. Backward accumulation: ``A_i = r_i + gamma * A_{i+1}``.
        3. Broadcast ``A_i`` to all tokens in turn *i*.
    """
    discounted_returns = torch.zeros(bsz, seq_len, device=device, dtype=normalized_rewards.dtype)

    for sample_idx in range(bsz):
        sample_rewards = normalized_rewards[sample_idx]
        sample_mask = response_mask[sample_idx]

        if turn_boundary_mask is not None:
            reward_positions = turn_boundary_mask[sample_idx].nonzero(as_tuple=True)[0].tolist()
        else:
            reward_positions = (sample_rewards != 0).nonzero(as_tuple=True)[0].tolist()

        if len(reward_positions) == 0:
            continue

        # Backward accumulation
        turn_data = []
        next_turn_adv = 0.0
        for pos in reversed(reward_positions):
            turn_reward = sample_rewards[pos].item()
            turn_adv = turn_reward + gamma * next_turn_adv
            turn_data.append((pos, turn_adv))
            next_turn_adv = turn_adv
        turn_data.reverse()

        # Broadcast to all tokens per turn
        prev_end = 0
        for reward_pos, adv in turn_data:
            for t in range(prev_end, reward_pos + 1):
                if sample_mask[t] == 1:
                    discounted_returns[sample_idx, t] = adv
            prev_end = reward_pos + 1

    return discounted_returns


@register_adv_est(AdvantageEstimator.GRPO_IGPO)
def compute_grpo_igpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    gamma: float = 1.0,
    info_gain_norm_mode: str = "joint",
    curriculum_f1_weight: float = 1.0,
    curriculum_ig_weight: float = 1.0,
    **kwargs,
):
    """IGPO advantage: group-normalized rewards → turn-level discounted accumulation."""
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    # -- Build masks --
    with torch.no_grad():
        last_valid_pos = (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1)
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        f1_mask = (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1)
        ig_mask = (response_mask == 1) & (~f1_mask) & (token_level_rewards != 0)

    # -- Curriculum weights --
    if curriculum_f1_weight != 1.0 or curriculum_ig_weight != 1.0:
        weighted_rewards = token_level_rewards.clone()
        weighted_rewards = torch.where(f1_mask, token_level_rewards * curriculum_f1_weight, weighted_rewards)
        weighted_rewards = torch.where(ig_mask, token_level_rewards * curriculum_ig_weight, weighted_rewards)
        token_level_rewards = weighted_rewards

    # -- Group mapping --
    unique_indices, inverse_indices = np.unique(index, return_inverse=True)
    group_ids = torch.tensor(inverse_indices, device=device, dtype=torch.long)
    num_groups = len(unique_indices)
    group_ids_expanded = group_ids.unsqueeze(1).expand(-1, seq_len)

    def _compute_group_stats(mask):
        flat_mask = mask.view(-1)
        flat_rewards = token_level_rewards.view(-1)
        flat_group_ids = group_ids_expanded.reshape(-1)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return torch.zeros(num_groups, device=device), torch.ones(num_groups, device=device)
        valid_rewards = flat_rewards[valid_idx]
        valid_groups = flat_group_ids[valid_idx]
        group_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, valid_rewards)
        group_count = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, torch.ones_like(valid_rewards))
        group_mean = group_sum / group_count.clamp(min=1.0)
        expanded_mean = group_mean[valid_groups]
        sq_diff = (valid_rewards - expanded_mean) ** 2
        group_sq_sum = torch.zeros(num_groups, device=device).scatter_add_(0, valid_groups, sq_diff)
        group_var = group_sq_sum / group_count.clamp(min=1.0)
        group_std = torch.sqrt(group_var + 1e-8)
        group_std = torch.where(group_count <= 1, torch.ones_like(group_std), group_std)
        return group_mean, group_std

    # -- Normalization --
    normalized_rewards = torch.zeros_like(token_level_rewards)

    if info_gain_norm_mode == "separate":
        f1_mean, f1_std = _compute_group_stats(f1_mask)
        f1_mean_map = f1_mean[group_ids_expanded]
        f1_std_map = f1_std[group_ids_expanded]
        norm_f1 = (token_level_rewards - f1_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_f1 = norm_f1 / (f1_std_map + epsilon)
        normalized_rewards = torch.where(f1_mask, norm_f1, normalized_rewards)

        ig_mean, ig_std = _compute_group_stats(ig_mask)
        ig_mean_map = ig_mean[group_ids_expanded]
        ig_std_map = ig_std[group_ids_expanded]
        norm_ig = (token_level_rewards - ig_mean_map)
        if norm_adv_by_std_in_grpo:
            norm_ig = norm_ig / (ig_std_map + epsilon)
        normalized_rewards = torch.where(ig_mask, norm_ig, normalized_rewards)
    else:  # joint
        joint_mask = f1_mask | ig_mask
        g_mean, g_std = _compute_group_stats(joint_mask)
        mean_map = g_mean[group_ids_expanded]
        std_map = g_std[group_ids_expanded]
        norm_val = (token_level_rewards - mean_map)
        if norm_adv_by_std_in_grpo:
            norm_val = norm_val / (std_map + epsilon)
        normalized_rewards = torch.where(joint_mask, norm_val, normalized_rewards)

    # -- Turn-level discounted accumulation + broadcast --
    discounted_returns = _compute_turn_level_advantage(
        normalized_rewards=normalized_rewards,
        response_mask=response_mask,
        gamma=gamma,
        bsz=bsz,
        seq_len=seq_len,
        device=device,
        turn_boundary_mask=f1_mask | ig_mask,
    )

    return discounted_returns, discounted_returns


@register_adv_est(AdvantageEstimator.TREE_GRPO)
def compute_tree_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    tree_index: np.ndarray | None = None,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tree-GRPO advantage: inter-tree + intra-tree (additive).

    Following the Tree-GRPO paper:
    - Inter-tree: normalize scores across ALL leaves of the same prompt (by uid)
    - Intra-tree: normalize scores within leaves of the same tree (by tree_uid)
    - Final advantage = inter + intra

    Args:
        token_level_rewards: (bs, response_length) — reward on last response token.
        response_mask: (bs, response_length).
        index: (bs,) — prompt group ID (uid).
        tree_index: (bs,) — tree ID (tree_uid). If None, falls back to standard GRPO.
        epsilon: Small value for numerical stability.
        config: Algorithm configuration.

    Returns:
        advantages, returns: Both (bs, response_length).
    """
    if tree_index is None:
        return compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=epsilon,
            config=config,
        )

    norm_adv_by_std = True
    if config is not None:
        norm_adv_by_std = config.get("norm_adv_by_std_in_grpo", True)

    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    bsz = scores.shape[0]

    with torch.no_grad():
        # Inter-tree: normalize by prompt group (uid)
        uid2indices = defaultdict(list)
        for i in range(bsz):
            uid2indices[index[i]].append(i)

        inter_scores = torch.zeros_like(scores)
        for uid, indices in uid2indices.items():
            group_scores = scores[indices]
            mean = group_scores.mean()
            std = group_scores.std() if len(indices) > 1 else torch.tensor(1.0)
            for idx in indices:
                if norm_adv_by_std:
                    inter_scores[idx] = (scores[idx] - mean) / (std + epsilon)
                else:
                    inter_scores[idx] = scores[idx] - mean

        # Intra-tree: normalize by tree (tree_uid) — computed independently from raw scores
        tree2indices = defaultdict(list)
        for i in range(bsz):
            tree2indices[tree_index[i]].append(i)

        intra_scores = torch.zeros_like(scores)
        for tree_uid, indices in tree2indices.items():
            group_scores = scores[indices]
            mean = group_scores.mean()
            std = group_scores.std() if len(indices) > 1 else torch.tensor(1.0)
            for idx in indices:
                if norm_adv_by_std:
                    intra_scores[idx] = (scores[idx] - mean) / (std + epsilon)
                else:
                    intra_scores[idx] = scores[idx] - mean

        # Final: additive combination (matching Tree-GRPO paper)
        advantages = (inter_scores + intra_scores).unsqueeze(-1) * response_mask

    return advantages, advantages


def _build_think_mask(response_ids: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    """Build a boolean mask for tokens inside think blocks (between <think> and </think>).

    Args:
        response_ids: (bs, response_length) token IDs.
        tokenizer: tokenizer with think_start_id and think_end_id attributes.

    Returns:
        Boolean mask (bs, response_length): True for tokens inside think blocks.
    """
    think_start_id = getattr(tokenizer, "think_start_id", None)
    think_end_id = getattr(tokenizer, "think_end_id", None)

    if think_start_id is None or think_end_id is None:
        # Try to get from vocab (Qwen models use special tokens for think blocks)
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        think_start_id = vocab.get("</think>", None)
        think_end_id = vocab.get("</think>", None)

    if think_start_id is None or think_end_id is None:
        # No think tokens found — all tokens are outside think blocks
        return torch.zeros_like(response_ids, dtype=torch.bool)

    bs, resp_len = response_ids.shape
    mask = torch.zeros(bs, resp_len, dtype=torch.bool, device=response_ids.device)

    for i in range(bs):
        in_think = False
        for t in range(resp_len):
            token_id = response_ids[i, t].item()
            if token_id == think_start_id:
                in_think = True
                mask[i, t] = True
            elif token_id == think_end_id:
                mask[i, t] = True  # include the end token itself
                in_think = False
            elif in_think:
                mask[i, t] = True

    return mask


@register_adv_est(AdvantageEstimator.GRPO_CALIBADV)
def compute_grpo_calibadv_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    tokenizer: Any = None,
    outcome_scores: np.ndarray | None = None,
    step_scores: list[list[float]] | None = None,
    doc_ids_per_step: list[list[list[str]]] | None = None,
    lambda_advreb: float = 1.0,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CalibAdv advantage estimator: PreThink + SoftPen + AdvReb.

    Three modules:
    1. PreThink: zero-out advantages for tokens inside think blocks.
    2. SoftPen: reduce negative advantage on intermediate steps that retrieve
       useful docs (overlap with silver doc set from correct rollouts).
    3. AdvReb: scale up positive advantage on the final answer step to rebalance
       the advantage ratio between positive and negative outcomes.

    Args:
        token_level_rewards: (bs, response_length) — used for device/dtype.
        response_mask: (bs, response_length) — 1 for assistant tokens.
        index: (bs,) group ids for GRPO normalization.
        tokenizer: for think-block detection.
        outcome_scores: (bs,) per-sample outcome rewards (F1 × format).
        step_scores: per-sample per-step doc overlap ratios.
        doc_ids_per_step: per-sample per-step doc_id lists.
        lambda_advreb: scaling factor for AdvReb (default 1.0).
        epsilon: numerical stability.
        norm_adv_by_std_in_grpo: scale by group std.
        config: algorithm configuration.

    Returns:
        advantages, returns — both (bs, response_length).
    """
    with torch.no_grad():
        bs, resp_len = token_level_rewards.shape
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype

        # --- Step 1: Outcome scores ---
        if outcome_scores is None:
            outcome_scores_np = np.array(
                [float(token_level_rewards[i].sum().item()) for i in range(bs)],
                dtype=np.float32,
            )
        else:
            outcome_scores_np = np.array(outcome_scores, dtype=np.float32)

        outcome_tensor = torch.tensor(outcome_scores_np, dtype=dtype, device=device)

        # --- Step 2: Group normalization on outcome scores ---
        id2outcome: dict[Any, list[float]] = defaultdict(list)
        id2indices: dict[Any, list[int]] = defaultdict(list)
        for i in range(bs):
            id2outcome[index[i]].append(outcome_scores_np[i])
            id2indices[index[i]].append(i)

        id2mean: dict[Any, float] = {}
        id2std: dict[Any, float] = {}
        for idx in id2outcome:
            scores_tensor = torch.tensor(id2outcome[idx], dtype=dtype, device=device)
            if len(id2outcome[idx]) == 1:
                id2mean[idx] = 0.0
                id2std[idx] = 1.0
            else:
                id2mean[idx] = float(scores_tensor.mean())
                id2std[idx] = float(scores_tensor.std())

        outcome_adv = torch.zeros(bs, dtype=dtype, device=device)
        for i in range(bs):
            if norm_adv_by_std_in_grpo:
                outcome_adv[i] = (outcome_tensor[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                outcome_adv[i] = outcome_tensor[i] - id2mean[index[i]]

        # --- Step 3: Step boundaries ---
        step_boundaries = _get_step_boundaries(response_mask)

        # --- Step 4: Think mask ---
        # Get response_ids from the batch data (passed via config or extra kwargs)
        response_ids = config.get("response_ids", None) if config is not None else None
        if response_ids is None:
            # Fallback: no think masking if response_ids not available
            think_mask = torch.zeros(bs, resp_len, dtype=torch.bool, device=device)
        else:
            think_mask = _build_think_mask(response_ids, tokenizer)

        # --- Step 5: Build silver doc set per group ---
        # Silver docs = union of doc_ids from samples with outcome_score > 0
        id2silver_docs: dict[Any, set[str]] = defaultdict(set)
        if doc_ids_per_step is not None:
            for i in range(bs):
                if outcome_scores_np[i] > 0:
                    for step_doc_ids in doc_ids_per_step[i]:
                        id2silver_docs[index[i]].update(step_doc_ids)

        # --- Step 6: Per-step advantage computation ---
        advantages = torch.zeros(bs, resp_len, dtype=dtype, device=device)

        # Compute AdvReb ratio per group
        id2advreb_ratio: dict[Any, float] = {}
        for idx in id2outcome:
            pos_abs = [abs(s) for s in id2outcome[idx] if s > 0]
            neg_abs = [abs(s) for s in id2outcome[idx] if s <= 0]
            mean_pos = sum(pos_abs) / len(pos_abs) if pos_abs else 1.0
            mean_neg = sum(neg_abs) / len(neg_abs) if neg_abs else 1.0
            id2advreb_ratio[idx] = mean_neg / (mean_pos + epsilon)

        for i in range(bs):
            boundaries = step_boundaries[i]
            n_steps = len(boundaries)
            silver_docs = id2silver_docs.get(index[i], set())
            r_g = id2advreb_ratio.get(index[i], 1.0)

            for step_idx, (start, end) in enumerate(boundaries):
                is_final_step = (step_idx == n_steps - 1)

                if is_final_step:
                    # AdvReb: scale up positive advantages on final answer step
                    if outcome_adv[i] > 0:
                        step_adv = outcome_adv[i] * (1 + lambda_advreb * r_g)
                    else:
                        step_adv = outcome_adv[i]
                else:
                    # Intermediate step: SoftPen
                    if outcome_adv[i] < 0:
                        # Compute overlap with silver docs
                        step_doc_ids_set: set[str] = set()
                        if doc_ids_per_step is not None and step_idx < len(doc_ids_per_step[i]):
                            step_doc_ids_set = set(doc_ids_per_step[i][step_idx])

                        if step_doc_ids_set and silver_docs:
                            overlap = len(step_doc_ids_set & silver_docs)
                            union = len(step_doc_ids_set | silver_docs)
                            overlap_ratio = overlap / (union + epsilon)
                        else:
                            overlap_ratio = 0.0

                        softpen_factor = 1 - overlap_ratio
                        step_adv = outcome_adv[i] * softpen_factor
                    else:
                        step_adv = outcome_adv[i]

                # Broadcast step advantage to all tokens in this step
                for t in range(start, end + 1):
                    if think_mask[i, t]:
                        advantages[i, t] = 0.0  # PreThink: zero think tokens
                    else:
                        advantages[i, t] = step_adv

        # Mask out non-response tokens
        advantages = advantages * response_mask

    return advantages, advantages
