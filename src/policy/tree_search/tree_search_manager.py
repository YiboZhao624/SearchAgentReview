"""Tree search AgentLoopWorker and AgentLoopManager.

Follows the same pattern as AsyncToolScoreAgentLoopManager:
- TreeSearchAgentLoopWorker overrides generate_sequences() to run tree search
- TreeSearchAgentLoopManager wires the worker class

Key difference from standard agent loop: instead of running the state machine for each
item independently, we group items by uid and run M tree searches per prompt, each
producing K sampled leaves.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    _InternalAgentLoopOutput,
)
from verl.protocol import DataProto
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.model import compute_position_id_with_mask

from src.policy.agent_loop.agent_loop_worker_base import BaseAgentLoopManager, BaseAgentLoopWorker
from src.policy.tree_search.tree_search_agent_loop import TreeSearchLoop

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TreeSearchAgentLoopWorker(BaseAgentLoopWorker):
    """AgentLoopWorker that runs tree search instead of standard chain rollouts.

    For each prompt group (M items sharing a uid = same prompt), runs M independent
    tree searches. Each tree produces K sampled leaves. Returns M*K outputs per prompt.

    Tools are initialized from the same config as the standard ToolAgentLoop, ensuring
    consistent search behavior with other baselines.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # Register custom tool parsers (search_r1, qwen) so ToolParser registry can find them
        from src.policy.tools import search_r1_tool_parser  # noqa: F401
        from src.policy.tools import verl_qwen_tool_parser  # noqa: F401

        # Initialize tools from config (same as ToolAgentLoop)
        tool_config_path = self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        logger.info(
            "TreeSearchAgentLoopWorker initialized with tools: %s",
            list(self.tools.keys()),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Override generate_sequences to run tree search.

        For each item in the batch, run a tree search and return K leaves.
        The K-expansion (repeating the original batch to match) is handled
        in ray_trainer.py fit() loop.
        """
        config = self.config.actor_rollout_ref.rollout
        custom = config.get("custom", {}) or {}
        tree_config = OmegaConf.to_container(
            custom.get("tree_search", {}), resolve=True
        ) if hasattr(custom, "get") else (custom if isinstance(custom, dict) else {}).get("tree_search", {})

        # If tree search not enabled or validation, fall back to standard single rollout
        if not tree_config.get("enable", False) or batch.meta_info.get("validate", False):
            return await super().generate_sequences(batch)

        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }

        # Override for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        ts_k = tree_config.get("ts_k", 3)

        # Create TreeSearchLoop with tools
        tree_loop = TreeSearchLoop(
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            tools=self.tools,
            config=self.config,
            tree_config=tree_config,
        )

        # Group by uid to identify which items are repeats of the same prompt
        uids = batch.non_tensor_batch.get(
            "uid", np.array([str(i) for i in range(len(batch))])
        )
        uid_groups: dict[str, list[int]] = {}
        for i in range(len(batch)):
            uid = str(uids[i])
            uid_groups.setdefault(uid, []).append(i)

        # Launch tree searches
        tasks: list[asyncio.Task] = []
        task_meta: list[tuple[str, str]] = []  # (uid, tree_uid) per task

        for uid, indices in uid_groups.items():
            for batch_idx in indices:
                tree_uid = uuid4().hex

                kwargs = {k: v[batch_idx] for k, v in batch.non_tensor_batch.items()}
                raw_prompt = kwargs.get("raw_prompt", [])
                extra_info = kwargs.get("extra_info", {})
                if not isinstance(extra_info, dict):
                    extra_info = {}
                ground_truth = extra_info.get("ground_truth", {"target": []})
                messages = list(raw_prompt) if raw_prompt is not None else []

                async def _run_tree(
                    msgs, gt, sp, tl, _tree_uid, _uid
                ):
                    t_start = time.perf_counter()
                    # Pass tool_schemas so chat template includes hermes tool descriptions
                    template_kwargs = {"add_generation_prompt": True}
                    if tl.tool_schemas:
                        template_kwargs["tools"] = tl.tool_schemas
                    prompt_ids = self.tokenizer.apply_chat_template(
                        msgs, **template_kwargs
                    )
                    leaves = await tl.run_tree_search(
                        prompt_ids=prompt_ids,
                        messages=msgs,
                        sampling_params=sp,
                        ground_truth=gt,
                    )
                    t_elapsed = time.perf_counter() - t_start
                    return [
                        (tl.leaf_to_agent_loop_output(leaf, prompt_ids), _tree_uid, _uid, t_elapsed)
                        for leaf in leaves
                    ]

                tasks.append(
                    asyncio.ensure_future(
                        _run_tree(messages, ground_truth, sampling_params, tree_loop, tree_uid, uid)
                    )
                )
                task_meta.append((uid, tree_uid))

        # Run all trees in parallel
        results = await asyncio.gather(*tasks)

        all_outputs: list[AgentLoopOutput] = []
        all_tree_uids: list[str] = []
        all_uids: list[str] = []
        all_elapsed: list[float] = []

        for tree_results in results:
            for output, tree_uid, uid, elapsed in tree_results:
                all_outputs.append(output)
                all_tree_uids.append(tree_uid)
                all_uids.append(uid)
                all_elapsed.append(elapsed)

        if not all_outputs:
            logger.warning("Tree search produced no outputs, falling back to standard")
            return await super().generate_sequences(batch)

        # Build DataProto from outputs
        return self._build_data_proto(all_outputs, all_uids, all_tree_uids, all_elapsed)

    def _build_data_proto(
        self,
        outputs: list[AgentLoopOutput],
        uids: list[str],
        tree_uids: list[str],
        elapsed_times: list[float],
    ) -> DataProto:
        """Convert list of AgentLoopOutput to DataProto.

        Handles padding, attention masks, and token_level_scores (EM reward).
        """
        prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        response_length = self.config.actor_rollout_ref.rollout.response_length
        pad_id = self.tokenizer.pad_token_id or 0
        bsz = len(outputs)

        prompts_list = []
        responses_list = []
        input_ids_list = []
        attention_mask_list = []
        response_mask_list = []
        logprobs_list = []
        has_logprobs = outputs[0].response_logprobs is not None

        for output in outputs:
            # Prompt: left-pad to prompt_length
            p_ids = list(output.prompt_ids)
            if len(p_ids) > prompt_length:
                p_ids = p_ids[-prompt_length:]
            pad_len = prompt_length - len(p_ids)
            p_padded = [pad_id] * pad_len + p_ids

            # Response: right-pad to response_length
            r_ids = list(output.response_ids[:response_length])
            r_mask = list(output.response_mask[:response_length])
            r_pad = response_length - len(r_ids)
            r_padded = r_ids + [pad_id] * r_pad
            m_padded = r_mask + [0] * r_pad

            # Input = prompt + response
            input_ids = p_padded + r_padded

            # Attention mask: 1 for non-pad prompt tokens + valid response tokens
            attn = [0] * pad_len + [1] * len(p_ids)
            for j in range(response_length):
                if j < len(r_ids):
                    attn.append(1)
                else:
                    attn.append(0)

            prompts_list.append(torch.tensor(p_padded, dtype=torch.long))
            responses_list.append(torch.tensor(r_padded, dtype=torch.long))
            input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            attention_mask_list.append(torch.tensor(attn, dtype=torch.long))
            response_mask_list.append(torch.tensor(m_padded, dtype=torch.long))

            if has_logprobs and output.response_logprobs:
                lp = list(output.response_logprobs[:response_length])
                lp_padded = lp + [0.0] * (response_length - len(lp))
                logprobs_list.append(torch.tensor(lp_padded, dtype=torch.float32))

        batch_dict = {
            "prompts": torch.stack(prompts_list),
            "responses": torch.stack(responses_list),
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "response_mask": torch.stack(response_mask_list),
        }

        # Position IDs
        batch_dict["position_ids"] = compute_position_id_with_mask(
            batch_dict["attention_mask"]
        )

        if logprobs_list:
            batch_dict["rollout_log_probs"] = torch.stack(logprobs_list)

        # Token-level scores: EM reward on last valid response token
        token_level_scores = torch.zeros(bsz, response_length, dtype=torch.float32)
        for i, output in enumerate(outputs):
            reward = output.extra_fields.get("tree_reward", 0.0)
            valid_indices = batch_dict["response_mask"][i].nonzero(as_tuple=True)[0]
            if len(valid_indices) > 0:
                last_idx = valid_indices[-1].item()
                token_level_scores[i, last_idx] = float(reward)
        batch_dict["token_level_scores"] = token_level_scores

        td = TensorDict(batch_dict, batch_size=bsz)

        non_tensor = {
            "uid": np.array(uids, dtype=object),
            "tree_uid": np.array(tree_uids, dtype=object),
            "multi_modal_inputs": np.array([{} for _ in range(bsz)], dtype=object),
        }

        # Propagate tree metadata
        for key in ("tree_reward", "tree_depth", "tree_terminal"):
            vals = [o.extra_fields.get(key, 0) for o in outputs]
            non_tensor[key] = np.array(vals, dtype=object)

        result = DataProto(batch=td, non_tensor_batch=non_tensor)
        # Metrics expected by agent_loop framework (generate_sequences, tool_calls, num_preempted per sample)
        # The framework computes timing from metrics in _performance_metrics(), so we only provide metrics.
        result.meta_info["metrics"] = [
            {
                "generate_sequences": elapsed_times[i] if i < len(elapsed_times) else 0.0,
                "tool_calls": elapsed_times[i] if i < len(elapsed_times) else 0.0,
                "num_preempted": 0,
            }
            for i in range(bsz)
        ]
        return result


class TreeSearchAgentLoopManager(BaseAgentLoopManager):
    """AgentLoopManager using TreeSearchAgentLoopWorker."""

    worker_cls = TreeSearchAgentLoopWorker


__all__ = ["TreeSearchAgentLoopManager", "TreeSearchAgentLoopWorker"]
