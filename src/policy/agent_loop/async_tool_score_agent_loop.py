"""Tool agent loop that submits async step scoring via Ray."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto

from src.policy.agent_loop.dynamic_qwen_tool_agent_loop import DynamicQwenToolAgentLoop
from src.policy.prompting.dynamic_prompt import update_messages

logger = logging.getLogger(__name__)


def _extract_doc_ids_from_tool_response(tool_response_text: str) -> list[str]:
    """Extract doc_ids from a tool response JSON string."""
    try:
        payload = json.loads(tool_response_text)
    except (json.JSONDecodeError, TypeError):
        return []
    doc_ids: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, list):
                for hit in item:
                    if isinstance(hit, dict):
                        did = hit.get("doc_id") or hit.get("id")
                        if did is not None:
                            doc_ids.append(str(did).strip())
            elif isinstance(item, dict):
                did = item.get("doc_id") or item.get("id")
                if did is not None:
                    doc_ids.append(str(did).strip())
    elif isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"doc_id", "id"}:
                if value is not None:
                    doc_ids.append(str(value).strip())
            if isinstance(value, (dict, list)):
                _collect_nested_doc_ids(value, doc_ids)
    return doc_ids


def _collect_nested_doc_ids(payload: Any, doc_ids: list[str]) -> None:
    """Recursively collect doc_ids from nested dicts/lists."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"doc_id", "id"}:
                if value is not None:
                    doc_ids.append(str(value).strip())
            if isinstance(value, (dict, list)):
                _collect_nested_doc_ids(value, doc_ids)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                _collect_nested_doc_ids(item, doc_ids)


class AsyncToolScoreQwenToolAgentLoop(DynamicQwenToolAgentLoop):
    """Dynamic Qwen tool agent loop with async tool-step scoring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_loop_worker = None
        self._step_reward_context: dict[str, Any] = {}
        self._pending_step_scores: list[dict[str, Any]] = []
        self._step_doc_ids: list[list[str]] = []  # accumulate doc_ids per step
        self._enable_async_step_reward = bool(
            self.config.actor_rollout_ref.rollout.multi_turn.get("async_step_reward", False)
        )
        self._is_validation: bool = False

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        # English annotations: mirror dynamic prompt update before capturing context.
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        global_step = extra_info.get("global_step")
        self._is_validation = kwargs.get("meta_info", {}).get("validate", False)
        if self._is_validation:
            logger.info("🔍 Validation mode: async step scoring disabled")
        else:
            logger.debug("🚀 Training mode: async step scoring enabled")

        prompt_config = self._get_dynamic_prompt_config()
        messages = update_messages(
            messages,
            global_step=global_step,
            extra_info=extra_info,
            prompt_config=prompt_config,
        )
        kwargs["raw_prompt"] = messages

        self._step_reward_context = dict(kwargs)
        self._pending_step_scores = []
        self._step_doc_ids = []
        self._anchor_obs_list: list[str] = []
        output = await super().run(sampling_params, **kwargs)
        if self._enable_async_step_reward:
            logger.debug(
                "async_step_reward completed run: pending_count=%s",
                len(self._pending_step_scores),
            )
        if self._pending_step_scores:
            output.extra_fields.setdefault("pending_step_scores", [])
            output.extra_fields["pending_step_scores"].extend(self._pending_step_scores)
        if self._step_doc_ids:
            output.extra_fields.setdefault("step_doc_ids", [])
            output.extra_fields["step_doc_ids"].extend(self._step_doc_ids)
        if self._anchor_obs_list:
            output.extra_fields["anchor_obs_list"] = self._anchor_obs_list
        return output

    async def _handle_processing_tools_state(self, agent_data):
        # English annotations: copy base logic and insert async scoring per tool response.
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with self._maybe_tool_timer(agent_data):
            responses = await asyncio.gather(*tasks)

        # Decode the latest assistant turn once (response_ids = current turn only).
        assistant_response = self.tokenizer.decode(
            agent_data.response_ids, skip_special_tokens=True
        ) if agent_data.response_ids else ""

        for idx, (tool_response, tool_reward, _) in enumerate(responses):
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                if isinstance(tool_response.image, list):
                    for img in tool_response.image:
                        if img is not None:
                            new_images_this_turn.append(img)
                else:
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            self._submit_step_score(
                agent_data=agent_data,
                tool_response_text=tool_response.text or "",
                tool_name=tool_call_names[idx] if idx < len(tool_call_names) else None,
                step_index=idx,
                assistant_response=assistant_response,
            )

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            tool_response_text = self._build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "search_r1":
            tool_response_text = self._build_search_r1_tool_response_text(add_messages)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=new_images_this_turn,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return self._terminate_state()

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return self._generating_state()

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination: bool = False):
        # Capture anchor_obs BEFORE the model generates this turn.
        # anchor_obs = full conversation context the model will see = everything accumulated so far.
        if not self._is_validation:
            self._anchor_obs_list.append(self._build_anchor_obs(agent_data))

        # English annotations: reuse base generation logic, but add a fallback step score
        # when no tool call is produced.
        from verl.experimental.agent_loop.tool_agent_loop import AgentState

        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination=ignore_termination)
        return state

    def _build_anchor_obs(self, agent_data) -> str:
        """Build a compact string representation of the current observation state.

        This is the full context the model sees before generating turn t, i.e.,
        the accumulated question + previous search queries and results.
        """
        parts: list[str] = []
        step_num = 0
        for msg in agent_data.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if not content:
                continue
            if role == "user":
                parts.append(f"[Question]\n{content}")
            elif role == "assistant":
                step_num += 1
                parts.append(f"[Step {step_num} - Search]\n{content}")
            elif role == "tool":
                parts.append(f"[Step {step_num} - Result]\n{content}")
        return "\n\n".join(parts)

    def _submit_step_score(
        self,
        *,
        agent_data,
        tool_response_text: str,
        tool_name: str | None,
        step_index: int,
        assistant_response: str,
    ) -> None:
        # ✅ 首先检查 validation 模式
        if self._is_validation:
            logger.debug("async_step_reward validation mode: skip submit")
            return

        # English annotations: fire-and-forget Ray request to reward loop worker.
        if not self._enable_async_step_reward:
            logger.debug("async_step_reward disabled: skip submit")
            return
        if self.reward_loop_worker is None:
            logger.debug("async_step_reward missing reward_loop_worker: skip submit")
            return
        if not agent_data.response_ids:
            logger.debug("async_step_reward empty response_ids: skip submit")
            return

        raw_prompt = list(self._step_reward_context.get("raw_prompt", agent_data.messages))
        extra_info = dict(self._step_reward_context.get("extra_info") or {})
        extra_info.setdefault("tool_response", tool_response_text)
        extra_info.setdefault("assistant_response", assistant_response)
        if tool_name is not None:
            extra_info.setdefault("tool_name", tool_name)
        extra_info.setdefault("tool_step_index", step_index)

        # Extract doc_ids from tool_response for CalibAdv step scoring
        step_doc_ids = _extract_doc_ids_from_tool_response(tool_response_text)
        extra_info.setdefault("doc_ids", step_doc_ids)
        self._step_doc_ids.append(step_doc_ids)

        # Build raw conversation history from PREVIOUS completed steps only.
        # Exclude the current step's assistant message (the last one) because
        # it will appear in ``new_step`` — including it here would cause the
        # judge to see the same content twice, blurring the boundary between
        # history and the step being evaluated.
        #
        # At this point agent_data.messages contains:
        #   [system, user, asst_1, tool_1, ..., asst_N]
        # where asst_N is the CURRENT step's assistant generation.
        # We want history = everything BEFORE asst_N.
        last_assistant_idx = -1
        for i in range(len(agent_data.messages) - 1, -1, -1):
            if agent_data.messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        history_parts: list[str] = []
        step_num = 0
        for i, msg in enumerate(agent_data.messages):
            if i >= last_assistant_idx and last_assistant_idx >= 0:
                break  # stop before current step's assistant message
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if not content:
                continue
            if role == "assistant":
                step_num += 1
                history_parts.append(f"[Step {step_num} - Assistant]\n{content}")
            elif role == "tool":
                history_parts.append(f"[Step {step_num} - Tool Response]\n{content}")
        extra_info["raw_history"] = "\n\n".join(history_parts)
        extra_info["current_step_number"] = step_num + 1

        responses = torch.tensor(agent_data.response_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(responses)
        batch = TensorDict({"responses": responses, "attention_mask": attention_mask}, batch_size=1)

        non_tensor_batch = {
            "raw_prompt": np.array([raw_prompt], dtype=object),
            "data_source": np.array([self._step_reward_context.get("data_source")], dtype=object),
            "reward_model": np.array([self._step_reward_context.get("reward_model")], dtype=object),
            "extra_info": np.array([extra_info], dtype=object),
        }

        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        obj_ref = self.reward_loop_worker.compute_score.remote(data)
        self._pending_step_scores.append({"idx": step_index, "ref": obj_ref})
        logger.debug(
            "async_step_reward submit: step=%s tool=%s response_len=%s",
            step_index,
            tool_name,
            len(agent_data.response_ids),
        )

    def _maybe_tool_timer(self, agent_data):
        return self._tool_timer_context(agent_data)

    def _tool_timer_context(self, agent_data):
        from verl.utils.profiler import simple_timer

        return simple_timer("tool_calls", agent_data.metrics)

    def _build_gpt_oss_tool_response_text(self, add_messages, tool_call_names):
        from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text

        return build_gpt_oss_tool_response_text(add_messages, tool_call_names)

    def _build_search_r1_tool_response_text(self, add_messages):
        from verl.experimental.agent_loop.utils import build_search_r1_tool_response_text

        return build_search_r1_tool_response_text(add_messages)

    def _terminate_state(self):
        from verl.experimental.agent_loop.tool_agent_loop import AgentState

        return AgentState.TERMINATED

    def _generating_state(self):
        from verl.experimental.agent_loop.tool_agent_loop import AgentState

        return AgentState.GENERATING


__all__ = ["AsyncToolScoreQwenToolAgentLoop"]

