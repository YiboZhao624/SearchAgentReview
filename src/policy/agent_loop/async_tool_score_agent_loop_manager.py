"""AgentLoopManager/Worker with async tool-step score harvesting."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import ray

from src.policy.agent_loop.agent_loop_worker_base import BaseAgentLoopManager, BaseAgentLoopWorker

logger = logging.getLogger(__name__)


class AsyncToolScoreAgentLoopWorker(BaseAgentLoopWorker):
    """AgentLoopWorker that injects reward_loop_worker and harvests step scores."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Restore stdout/stderr so prints/logging are visible in this worker.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        self._inject_extra_info(trajectory, kwargs)

        agent_loop_config = self._get_agent_loop_config(agent_name)
        agent_loop = self._instantiate_agent_loop(agent_loop_config)
        agent_loop.reward_loop_worker = self.reward_loop_worker
        output = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs):
        return await super()._agent_loop_postprocess(output, **kwargs)

    async def generate_sequences(self, batch):
        output = await super().generate_sequences(batch)
        pending = output.non_tensor_batch.get("pending_step_scores")
        step_doc_ids_raw = output.non_tensor_batch.get("step_doc_ids")
        batch_size = len(output)
        existing_scores = output.non_tensor_batch.get("reward_scores")
        if existing_scores is not None and len(existing_scores) != batch_size:
            normalized_scores: list[dict[str, Any]] = []
            for idx in range(batch_size):
                if idx < len(existing_scores) and isinstance(existing_scores[idx], dict):
                    base = existing_scores[idx]
                else:
                    base = {}
                normalized_scores.append(base)
            output.non_tensor_batch["reward_scores"] = np.array(normalized_scores, dtype=object)
            existing_scores = output.non_tensor_batch["reward_scores"]

        if pending is None and step_doc_ids_raw is None:
            return output
        logger.debug("async_step_reward batch pending samples=%s", len(pending) if pending else 0)

        refs: list[Any] = []
        ref_meta: list[tuple[int, int, Any]] = []
        if pending is not None:
            for sample_idx, entries in enumerate(pending):
                if not entries:
                    continue
                for entry in entries:
                    ref = entry.get("ref") if isinstance(entry, dict) else None
                    if ref is None:
                        continue
                    refs.append(ref)
                    ref_meta.append((sample_idx, int(entry.get("idx", -1)), ref))

        if not refs and step_doc_ids_raw is None:
            output.non_tensor_batch.pop("pending_step_scores", None)
            logger.debug("async_step_reward no pending refs; skip wait")
            return output

        # Collect step doc_ids from reward_extra_info (from ray.get results)
        per_sample_doc_ids_from_reward: list[list[list[str]]] = [[] for _ in range(batch_size)]
        if refs:
            logger.debug("async_step_reward waiting refs=%s", len(refs))
            results = ray.get(refs)
            ref_to_result = {ref: result for ref, result in zip(refs, results, strict=True)}

            per_sample_entries: list[list[tuple[int, float, dict[str, Any]]]] = [[] for _ in range(batch_size)]
            for sample_idx, idx, ref in ref_meta:
                result = ref_to_result.get(ref)
                score = result.get("reward_score") if isinstance(result, dict) else result
                if score is None:
                    continue
                reward_extra_info = result.get("reward_extra_info") if isinstance(result, dict) else {}
                if not isinstance(reward_extra_info, dict):
                    reward_extra_info = {}
                per_sample_entries[sample_idx].append((idx, float(score), reward_extra_info))
                # Collect doc_ids from reward_extra_info for CalibAdv
                step_doc_ids = reward_extra_info.get("doc_ids", [])
                if step_doc_ids:
                    per_sample_doc_ids_from_reward[sample_idx].append(step_doc_ids)

            logger.debug(
                "async_step_reward collected refs=%s scored_samples=%s",
                len(refs),
                sum(1 for entries in per_sample_entries if entries),
            )

            merged_scores: list[dict[str, Any]] = []
            for sample_idx, sample_entries in enumerate(per_sample_entries):
                sample_entries.sort(key=lambda item: item[0])
                tool_step_scores = [score for _, score, _ in sample_entries]

                if existing_scores is not None and sample_idx < len(existing_scores):
                    base = existing_scores[sample_idx]
                    if not isinstance(base, dict):
                        base = {}
                else:
                    base = {}

                base.setdefault("tool_step_scores", [])
                base["tool_step_scores"].extend(tool_step_scores)
                if sample_entries:
                    extra_keys: set[str] = set()
                    for _, _, extra in sample_entries:
                        extra_keys.update(extra.keys())
                    for key in sorted(extra_keys):
                        if key == "score":
                            continue
                        series = [extra.get(key) for _, _, extra in sample_entries]
                        metric_key = f"tool_step_{key}"
                        base.setdefault(metric_key, [])
                        base[metric_key].extend(series)
                merged_scores.append(base)
        else:
            merged_scores = list(existing_scores) if existing_scores is not None else [{} for _ in range(batch_size)]
            per_sample_entries = [[] for _ in range(batch_size)]

        # Merge step_doc_ids from agent loop into reward_scores
        # step_doc_ids_raw is a list of lists (per sample, per step) from the agent loop
        if step_doc_ids_raw is not None:
            for sample_idx in range(batch_size):
                if sample_idx < len(step_doc_ids_raw):
                    agent_doc_ids = step_doc_ids_raw[sample_idx]
                    if isinstance(agent_doc_ids, list):
                        # Use agent loop doc_ids as primary source (more reliable)
                        merged_scores[sample_idx]["tool_step_doc_ids"] = agent_doc_ids
                elif per_sample_doc_ids_from_reward[sample_idx]:
                    # Fallback: use doc_ids from reward_extra_info
                    merged_scores[sample_idx]["tool_step_doc_ids"] = per_sample_doc_ids_from_reward[sample_idx]
        else:
            # No agent loop doc_ids — use reward_extra_info doc_ids as fallback
            for sample_idx in range(batch_size):
                if per_sample_doc_ids_from_reward[sample_idx]:
                    merged_scores[sample_idx]["tool_step_doc_ids"] = per_sample_doc_ids_from_reward[sample_idx]

        output.non_tensor_batch["reward_scores"] = np.array(merged_scores, dtype=object)
        output.non_tensor_batch.pop("pending_step_scores", None)
        output.non_tensor_batch.pop("step_doc_ids", None)
        # Score combination (outcome + step → final_score) is handled entirely
        # by the reward function (evolving_rubric_compute_score).  The combined
        # score is returned as result["score"] → rm_scores via agent_loop._postprocess.
        # No re-combination or rm_scores overwrite is needed here.
        logger.debug("async_step_reward merged scores into reward_scores")
        return output

    def _get_agent_loop_config(self, agent_name: str):
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry

        if agent_name not in _agent_loop_registry:
            raise ValueError(
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )
        return _agent_loop_registry[agent_name]

    def _instantiate_agent_loop(self, agent_loop_config):
        import hydra

        from verl.experimental.agent_loop.agent_loop import DictConfigWrap

        return hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            dataset_config=DictConfigWrap(self.config.data),
        )


class AsyncToolScoreAgentLoopManager(BaseAgentLoopManager):
    """AgentLoopManager using AsyncToolScoreAgentLoopWorker."""

    worker_cls = AsyncToolScoreAgentLoopWorker


__all__ = ["AsyncToolScoreAgentLoopManager", "AsyncToolScoreAgentLoopWorker"]

