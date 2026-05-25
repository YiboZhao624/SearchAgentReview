"""Custom agent loop that registers Qwen tool parser before use."""

from __future__ import annotations

from verl.experimental.agent_loop import ToolAgentLoop


class QwenToolAgentLoop(ToolAgentLoop):
    """ToolAgentLoop with Qwen tool parser registration."""

    def __init__(self, *args, **kwargs):
        # Ensure all custom parsers are registered before ToolAgentLoop initialization.
        from src.policy.tools import verl_qwen_tool_parser  # noqa: F401
        from src.policy.tools import search_r1_tool_parser  # noqa: F401

        super().__init__(*args, **kwargs)

        # Register ErrorToolCall as a hidden tool: available for dispatch (when the hermes
        # parser produces name="error_tool_call" on JSON parse failure) but NOT included in
        # self.tool_schemas, so its schema never appears in the system prompt.
        from src.policy.tools.local_search import ErrorToolCall
        _error_tool = ErrorToolCall(config={})
        self.tools[_error_tool.name] = _error_tool


__all__ = ["QwenToolAgentLoop"]

