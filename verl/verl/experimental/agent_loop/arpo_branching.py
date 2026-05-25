# Copyright 2025 Search-Agent Authors
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
"""ARPO entropy-based adaptive branching coordinator.

Implements the branching decision algorithm from ARPO (Agentic Reinforced
Policy Optimization): at tool-call boundaries, trajectories with high
entropy (uncertainty) are branched to concentrate sampling budget on
high-uncertainty decision points.
"""

import asyncio
import copy
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class BranchRequest:
    """A pending branch: deep-copied agent_data ready to spawn a new trajectory."""

    sample_idx: int
    agent_data: Any  # AgentData instance (deep-copied)
    parent_trajectory_id: str


class ArpoBranchCoordinator:
    """ARPO entropy-based adaptive branching coordinator.

    Manages branching decisions across all trajectories for a batch.
    At each tool-call boundary, decides whether to branch based on
    entropy delta from the trajectory's initial entropy.

    Args:
        n_rollouts: Target number of rollouts per prompt.
        initial_rollouts: Number of rollouts to start with (< n_rollouts).
        beam_size: Max branches per active trajectory per tool boundary.
        branch_probability: Base threshold for branching decision.
        entropy_weight: Sensitivity to entropy delta (higher = more
            responsive to uncertainty).
    """

    def __init__(
        self,
        n_rollouts: int,
        initial_rollouts: int,
        beam_size: int = 2,
        branch_probability: float = 0.5,
        entropy_weight: float = 0.5,
    ):
        self.n_rollouts = n_rollouts
        self.initial_rollouts = initial_rollouts
        self.beam_size = beam_size
        self.branch_probability = branch_probability
        self.entropy_weight = entropy_weight

        # Tracks how many active + completed rollouts exist per sample_idx
        self._rollout_counts: dict[int, int] = {}
        # Initial entropy per trajectory (trajectory_id -> float)
        self._initial_entropy: dict[str, float] = {}
        # Pending branches to be spawned
        self._branch_queue: list[BranchRequest] = []
        self._lock = asyncio.Lock()

    def register_trajectory(self, sample_idx: int, trajectory_id: str) -> None:
        """Register a new trajectory (initial or branched)."""
        self._rollout_counts[sample_idx] = self._rollout_counts.get(sample_idx, 0) + 1

    def get_rollout_count(self, sample_idx: int) -> int:
        """Return current rollout count for a sample."""
        return self._rollout_counts.get(sample_idx, 0)

    async def on_tool_boundary(
        self,
        sample_idx: int,
        trajectory_id: str,
        agent_data: Any,
        entropy: float,
    ) -> None:
        """Called by agent loop after tool processing. May enqueue a branch.

        Implements ARPO's branching decision:
        1. Track initial entropy per trajectory.
        2. Compute entropy_delta = entropy_now - entropy_initial.
        3. prob = random() - entropy_weight * entropy_delta (clamped to [0,1]).
        4. Branch if prob <= branch_probability.
        5. Branch = deep-copy agent_data, put on branch_queue.
        6. Stop branching when sample has n_rollouts.

        Args:
            sample_idx: Index of the prompt in the batch.
            trajectory_id: Unique ID for this trajectory.
            agent_data: Current AgentData state to potentially branch from.
            entropy: Entropy computed from recent generation logprobs.
        """
        async with self._lock:
            # Record initial entropy on first tool boundary
            if trajectory_id not in self._initial_entropy:
                self._initial_entropy[trajectory_id] = entropy

            # Already at capacity for this sample
            if self._rollout_counts.get(sample_idx, 0) >= self.n_rollouts:
                return

            # Compute branching decision
            entropy_initial = self._initial_entropy[trajectory_id]
            entropy_delta = entropy - entropy_initial

            # Higher entropy_delta → larger negative term → lower prob → more likely to branch
            prob = random.random() - self.entropy_weight * entropy_delta
            prob = max(0.0, min(1.0, prob))

            if prob > self.branch_probability:
                return

            # Determine how many branches to create (up to beam_size, capped by budget)
            budget = self.n_rollouts - self._rollout_counts.get(sample_idx, 0)
            n_branches = min(self.beam_size, budget)

            for _ in range(n_branches):
                if self._rollout_counts.get(sample_idx, 0) >= self.n_rollouts:
                    break

                # Deep-copy agent_data for the branch
                branched_data = copy.deepcopy(agent_data)
                self._branch_queue.append(
                    BranchRequest(
                        sample_idx=sample_idx,
                        agent_data=branched_data,
                        parent_trajectory_id=trajectory_id,
                    )
                )
                self._rollout_counts[sample_idx] = self._rollout_counts.get(sample_idx, 0) + 1

                logger.info(
                    f"ARPO branch: sample={sample_idx}, parent={trajectory_id}, "
                    f"entropy={entropy:.4f}, delta={entropy_delta:.4f}, "
                    f"count={self._rollout_counts[sample_idx]}/{self.n_rollouts}"
                )

    def drain_branches(self) -> list[BranchRequest]:
        """Drain and return all pending branch requests."""
        branches = list(self._branch_queue)
        self._branch_queue.clear()
        return branches

    def has_pending_branches(self) -> bool:
        """Check if there are pending branch requests."""
        return len(self._branch_queue) > 0

    def all_samples_complete(self, sample_indices: list[int]) -> bool:
        """Check if all samples have reached their target rollout count."""
        return all(
            self._rollout_counts.get(idx, 0) >= self.n_rollouts
            for idx in sample_indices
        )
