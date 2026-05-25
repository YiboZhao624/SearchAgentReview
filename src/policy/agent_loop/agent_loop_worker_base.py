"""Shared AgentLoop worker/manager utilities."""

from __future__ import annotations

import os
import sys
from typing import Any

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.reward_loop import RewardLoopWorker


class BaseAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker with quiet stdout/stderr and optional RewardLoopWorker."""

    def __init__(self, *args, **kwargs):
        self._devnull = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        # Prevent base class from creating RewardLoopWorker with noisy logging.
        self.reward_loop_worker = None
        super().__init__(*args, **kwargs)
        if self.use_reward_loop:
            self.reward_loop_worker = RewardLoopWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
            ).remote(self.config, self.reward_router_address)

    def _inject_extra_info(self, trajectory: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        extra_info = kwargs.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info.setdefault("global_step", trajectory.get("step"))
        extra_info.setdefault("rollout_n", trajectory.get("rollout_n"))
        kwargs["extra_info"] = extra_info
        return extra_info


class BaseAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that binds a worker class via class attr."""

    worker_cls = None

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "agent_loop_workers_class"):
            if self.worker_cls is None:
                raise ValueError("worker_cls must be set for BaseAgentLoopManager.")
            self.agent_loop_workers_class = ray.remote(self.worker_cls)
        super().__init__(*args, **kwargs)


__all__ = ["BaseAgentLoopWorker", "BaseAgentLoopManager"]

