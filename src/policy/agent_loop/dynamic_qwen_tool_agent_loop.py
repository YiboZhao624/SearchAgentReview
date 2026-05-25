"""Tool agent loop with dynamic prompt hook."""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.tool_parser import ToolParser

from src.policy.agent_loop.qwen_tool_agent_loop import QwenToolAgentLoop
from src.policy.prompting.dynamic_prompt import update_messages
from src.policy.tools.custom_tool_executor import execute_tool_call
from src.policy.tools.custom_tool_loader import load_tools


class DynamicQwenToolAgentLoop(QwenToolAgentLoop):
    """Qwen tool agent loop that updates prompt per training step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_custom_tooling()

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info") or {}
        global_step = extra_info.get("global_step")
        prompt_config = self._get_dynamic_prompt_config()
        messages = update_messages(
            messages,
            global_step=global_step,
            extra_info=extra_info,
            prompt_config=prompt_config,
        )
        kwargs["raw_prompt"] = messages
        return await super().run(sampling_params, **kwargs)

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        custom_tooling = self._get_custom_tooling_config()
        if custom_tooling.get("enable") and custom_tooling.get("use_custom_executor", False):
            return await execute_tool_call(
                tool_call=tool_call,
                tools=self.tools,
                tools_kwargs=tools_kwargs,
                agent_data=agent_data,
                max_tool_response_length=self.max_tool_response_length,
                tool_response_truncate_side=self.tool_response_truncate_side,
            )
        return await super()._call_tool(tool_call, tools_kwargs, agent_data)

    def _get_custom_tooling_config(self) -> dict[str, Any]:
        return self.config.actor_rollout_ref.rollout.multi_turn.get("custom_tooling", {}) or {}

    def _get_dynamic_prompt_config(self) -> dict[str, Any]:
        return self.config.actor_rollout_ref.rollout.multi_turn.get("dynamic_prompt", {}) or {}

    def _configure_custom_tooling(self) -> None:
        custom_tooling = self._get_custom_tooling_config()
        if not custom_tooling.get("enable", False):
            return

        parser_name = custom_tooling.get("tool_parser_name")
        if parser_name:
            self.tool_parser = ToolParser.get_tool_parser(parser_name, self.tokenizer)
            self.tool_parser_name = parser_name

        tool_config_path = custom_tooling.get("tool_config_path")
        override_tools = custom_tooling.get("override_tools")
        if tool_config_path or override_tools:
            tool_list = load_tools(tool_config_path, override_tools=override_tools)
            self.tools = {tool.name: tool for tool in tool_list}
            self.tool_schemas = [
                tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
            ]


__all__ = ["DynamicQwenToolAgentLoop"]

