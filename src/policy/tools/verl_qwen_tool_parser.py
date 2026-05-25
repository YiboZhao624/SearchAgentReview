"""Qwen tool parser registered into VERL ToolParser registry."""

from __future__ import annotations

import json
import logging
import os
import re

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import rollout_trace_op

from src.policy.tools.tool_parser import parse_qwen_tool_calls

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@ToolParser.register("qwen")
class QwenToolParser(ToolParser):
    """Parse Qwen tool calls from <tool_call>...</tool_call> blocks."""

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        content, function_calls = _parse_calls_from_text(text)
        return content, function_calls


@ToolParser.register("qwen_custom")
class CustomQwenToolParser(QwenToolParser):
    """Custom Qwen parser hook for user-owned parsing logic."""


@ToolParser.register("qwen_naive")
class NaiveQwenToolParser(ToolParser):
    """Naive parser that strips tool call blocks but never triggers tools."""

    @rollout_trace_op
    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)
        text = _THINK_PATTERN.sub("", text)
        content = _TOOL_CALL_PATTERN.sub("", text)
        return content, []


def _parse_calls_from_text(text: str) -> tuple[str, list[FunctionCall]]:
    text = _THINK_PATTERN.sub("", text)
    content, parsed_calls = parse_qwen_tool_calls(text)
    function_calls: list[FunctionCall] = []
    for call in parsed_calls:
        try:
            function_calls.append(
                FunctionCall(name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False))
            )
        except Exception as exc:
            logger.error("Failed to encode tool call: %s", exc)
    return content, function_calls


__all__ = ["QwenToolParser", "CustomQwenToolParser", "NaiveQwenToolParser"]

