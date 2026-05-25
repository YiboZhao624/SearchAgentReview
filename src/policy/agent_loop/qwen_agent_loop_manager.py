"""Custom AgentLoopManager for quiet workers and dynamic prompt metadata."""

from __future__ import annotations

from src.policy.agent_loop.agent_loop_worker_base import BaseAgentLoopManager, BaseAgentLoopWorker


class QuietAgentLoopWorker(BaseAgentLoopWorker):
    """AgentLoopWorker that suppresses stdout/stderr noise and injects step info."""

    async def _run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace: bool = True,
        **kwargs,
    ):
        self._inject_extra_info(trajectory, kwargs)
        return await super()._run_agent_loop(
            sampling_params,
            trajectory,
            agent_name=agent_name,
            trace=trace,
            **kwargs,
        )


class QwenAgentLoopManager(BaseAgentLoopManager):
    """AgentLoopManager using QuietAgentLoopWorker."""

    worker_cls = QuietAgentLoopWorker


__all__ = ["QwenAgentLoopManager"]

