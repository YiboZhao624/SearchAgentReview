"""Tree search agent loop using sglang server_manager.generate().

Adapted from Tree-GRPO (ICLR 2026). Instead of synchronous vLLM generate_sequences(),
uses the async sglang server for per-step generation. Each tree node expansion is an
independent async call, parallelized via asyncio.gather().

The tree search algorithm (matching original Tree-GRPO):
1. Step 1 — Initial chain: Run a full action chain from root until terminal or max_turns.
   This creates a single linear trajectory (the "trunk" of the tree).
2. Step 2 — Expansion (L iterations): Select N non-leaf nodes from ALL internal nodes
   (using random.choices with replacement), then run a full chain from each selected node.
   Each expansion creates a new branch from an intermediate state.
3. Step 3 — Sample K leaves per tree for training.

Key parameters:
- ts_n: number of nodes to expand per iteration (selected from non-leaf nodes)
- ts_l: number of expansion iterations
- ts_k: leaves to sample per tree
- max_turns: max assistant turns per trajectory (= max tree depth)

Key design:
- Uses LocalEmbeddingSearchTool (same tool as other baselines) for search
- Uses search_r1 format for consistent observation formatting
- The total assistant turns per trajectory = depth of the leaf in the tree
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AsyncLLMServerManager,
)
from verl.experimental.agent_loop.tool_agent_loop import _truncate_at_search_r1_tag
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.experimental.agent_loop.utils import build_search_r1_tool_response_text
from verl.tools.base_tool import BaseTool

from src.policy.tree_search.tree_node import TreeNode, new_node_id, reset_node_id_counter
from src.reward.exact_match_reward import compute_score as em_compute_score

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()
    return None


def _extract_search_queries(text: str) -> list[str]:
    """Extract search queries from <search>...</search> tags.

    Used by the search_r1 format: model outputs <search>query</search>.
    """
    pattern = r"<search>(.*?)</search>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


class TreeSearchLoop:
    """Runs tree search for a single prompt using sglang server.

    Uses the same LocalEmbeddingSearchTool and search_r1 format as the standard
    agent loop. Each node expansion = one assistant turn (generate → tool call → observe).

    Algorithm (matching original Tree-GRPO):
    1. Run a full chain from root (repeatedly expand until terminal/max_turns)
    2. For L iterations: pick N non-leaf nodes and run full chains from each
    3. Sample K leaves per tree
    """

    def __init__(
        self,
        server_manager: AsyncLLMServerManager,
        tokenizer: Any,
        tools: dict[str, BaseTool],
        config: DictConfig,
        tree_config: dict[str, Any],
    ):
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.tools = tools
        self.config = config
        self.topk = tree_config.get("topk", 3)
        self.ts_n = tree_config.get("ts_n", 2)  # nodes to expand per iteration
        self.ts_l = tree_config.get("ts_l", 2)  # expansion iterations
        self.ts_k = tree_config.get("ts_k", 4)  # leaves to sample per tree
        self.max_turns = tree_config.get("max_turns", 4)  # max assistant turns per trajectory
        self.max_obs_length = tree_config.get("max_obs_length", 500)
        self.expand_mode = tree_config.get("expand_mode", "random")

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.get(
            "max_tool_response_length", 4096
        )

        # Tool call format: hermes uses <tool_call> parser; search_r1 uses <search> regex
        self.format = config.actor_rollout_ref.rollout.multi_turn.get("format", "search_r1")
        self.tool_schemas = [
            tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
            for tool in tools.values()
        ] if self.format != "search_r1" else []
        self.tool_parser = (
            ToolParser.get_tool_parser(self.format, tokenizer)
            if self.format != "search_r1"
            else None
        )

    async def _execute_search(self, queries: list[str]) -> str:
        """Execute search using LocalEmbeddingSearchTool.

        Args:
            queries: List of search queries.

        Returns:
            Tool response text (JSON string from the tool).
        """
        tool = self.tools.get("local_search")
        if tool is None:
            # Fallback: try first available tool
            tool = next(iter(self.tools.values()), None)
        if tool is None:
            return json.dumps([])

        parameters = {"query_list": queries, "k": self.topk}
        try:
            instance_id, _ = await tool.create(create_kwargs={})
            tool_response, _, _ = await tool.execute(instance_id, parameters)
            await tool.release(instance_id)
            return tool_response.text or json.dumps([])
        except Exception as e:
            logger.warning(f"Tool search failed: {e}")
            return json.dumps([])

    async def _extract_queries(self, response_ids: list[int], response_text: str) -> list[str]:
        """Extract search queries from model output based on format.

        For search_r1: extracts from <search>...</search> tags.
        For hermes: uses ToolParser to parse <tool_call> and extracts query_list argument.
        """
        if self.format == "search_r1":
            return _extract_search_queries(response_text)

        # Hermes format: use ToolParser
        if self.tool_parser is None:
            return _extract_search_queries(response_text)  # fallback

        _, function_calls = await self.tool_parser.extract_tool_calls(response_ids)
        queries = []
        for fc in function_calls:
            try:
                args = json.loads(fc.arguments) if isinstance(fc.arguments, str) else fc.arguments
                # local_search tool accepts query_list
                query_list = args.get("query_list", [])
                if isinstance(query_list, list):
                    queries.extend(query_list)
                elif isinstance(query_list, str):
                    queries.append(query_list)
            except (json.JSONDecodeError, AttributeError):
                continue
        return queries

    def _build_observation_text(self, tool_response_text: str) -> str:
        """Format tool response based on configured format.

        search_r1: \\n\\n<information>Doc N(Title: X) text\\n...</information>\\n\\n
        hermes: <|im_start|>user\\n<tool_response>\\n...\\n</tool_response><|im_end|>\\n<|im_start|>assistant\\n
        """
        if self.format == "search_r1":
            messages = [{"role": "tool", "content": tool_response_text}]
            return build_search_r1_tool_response_text(messages)

        # Hermes format: Qwen wraps tool response in user turn with <tool_response> tags
        # and adds assistant prompt for next generation
        return (
            "<|im_start|>user\n<tool_response>\n"
            + tool_response_text
            + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
        )

    async def expand_node(
        self,
        node: TreeNode,
        sampling_params: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> TreeNode:
        """Expand a single non-terminal node by generating one assistant turn.

        One expansion = one step of the agent loop:
        1. Generate response from LLM (via sglang server_manager)
        2. Parse tool calls (hermes: <tool_call>, search_r1: <search>)
        3. If <answer>: terminal node with EM reward
        4. If tool call: execute tool, append observation, non-terminal
        5. If neither: terminal with 0 reward (malformed output)
        """
        request_id = uuid4().hex

        # Generate response
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=list(node.prompt_ids),
            sampling_params=sampling_params,
        )

        response_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else []

        # Truncate at first tool call / answer boundary
        if self.format == "search_r1":
            response_ids = _truncate_at_search_r1_tag(response_ids, self.tokenizer)
        # For hermes, we don't need to truncate since the model produces </tool_call> as stop

        if log_probs and len(log_probs) > len(response_ids):
            log_probs = log_probs[:len(response_ids)]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Create child node with LLM-generated tokens
        child = TreeNode(
            node_id=new_node_id(),
            response_ids=list(response_ids),
            response_mask=[1] * len(response_ids),
            response_logprobs=list(log_probs) if log_probs else [0.0] * len(response_ids),
        )
        node.add_child(child)

        # Build child's prompt_ids = parent's prompt_ids + response_ids
        child.prompt_ids = list(node.prompt_ids) + list(response_ids)

        # Check for answer
        answer = _extract_answer(response_text)
        if answer is not None:
            child.is_terminal = True
            result = em_compute_score(
                data_source="tree_grpo",
                solution_str=response_text,
                ground_truth=ground_truth,
                extra_info={},
            )
            child.reward = result["score"] if isinstance(result, dict) else float(result)
            child.messages = list(node.messages) + [
                {"role": "assistant", "content": response_text}
            ]
            return child

        # Extract search queries based on format
        queries = await self._extract_queries(response_ids, response_text)
        if not queries:
            # No tool call and no answer → terminal with 0 reward
            child.is_terminal = True
            child.reward = 0.0
            child.messages = list(node.messages) + [
                {"role": "assistant", "content": response_text}
            ]
            return child

        # Execute search using LocalEmbeddingSearchTool
        tool_response_text = await self._execute_search(queries)

        # Truncate tool response if too long
        if len(tool_response_text) > self.max_tool_response_length:
            tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"

        # Format observation in search_r1 format (same as standard agent loop)
        observation_text = self._build_observation_text(tool_response_text)

        # Encode observation tokens
        obs_ids = self.tokenizer.encode(observation_text, add_special_tokens=False)

        # Check total response length
        child_path = child.get_path_from_root()
        accumulated_response = sum(len(n.response_ids) for n in child_path[1:])  # skip root
        if accumulated_response + len(obs_ids) >= self.response_length:
            child.is_terminal = True
            child.reward = 0.0
            child.messages = list(node.messages) + [
                {"role": "assistant", "content": response_text}
            ]
            return child

        # Add observation to child node (mask=0 for tool response tokens)
        child.observation = observation_text
        child.prompt_ids = list(child.prompt_ids) + list(obs_ids)
        child.response_ids.extend(obs_ids)
        child.response_mask.extend([0] * len(obs_ids))
        if child.response_logprobs:
            child.response_logprobs.extend([0.0] * len(obs_ids))

        child.messages = list(node.messages) + [
            {"role": "assistant", "content": response_text},
            {"role": "tool", "content": tool_response_text},
        ]

        # Check max assistant turns (each expansion = one assistant turn)
        assistant_turns = sum(1 for m in child.messages if m.get("role") == "assistant")
        if assistant_turns >= self.max_turns:
            child.is_terminal = True
            child.reward = 0.0

        return child

    async def run_full_chain(
        self,
        node: TreeNode,
        sampling_params: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> TreeNode:
        """Run a full action chain from a node until terminal or max_turns.

        This is the equivalent of gen_action_chain() in the original Tree-GRPO.
        It repeatedly calls expand_node() on the current tip node until:
        - The node becomes terminal (found <answer> or malformed output), or
        - The trajectory reaches max_turns assistant turns.

        Args:
            node: Starting node (root or an intermediate non-leaf node).
            sampling_params: LLM sampling parameters.
            ground_truth: Ground truth for EM reward.

        Returns:
            The terminal (leaf) node at the end of the chain.
        """
        current = node
        for _ in range(self.max_turns):
            if current.is_terminal:
                break
            # Count assistant turns in the current trajectory
            assistant_turns = sum(1 for m in current.messages if m.get("role") == "assistant")
            if assistant_turns >= self.max_turns:
                current.is_terminal = True
                break
            child = await self.expand_node(current, sampling_params, ground_truth)
            current = child
        return current

    async def run_tree_search(
        self,
        prompt_ids: list[int],
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> list[TreeNode]:
        """Run tree search matching the original Tree-GRPO algorithm.

        Step 1 — Initial chain:
            Run a full action chain from root until terminal/max_turns.
            This creates the tree trunk: root → n1 → n2 → ... → leaf.

        Step 2 — Expansion (L iterations):
            For each iteration, select N non-leaf nodes from ALL internal nodes
            in the tree (using random.choices with replacement). For each selected
            node, run a full chain from that node. This creates new branches
            from intermediate states.

            Example with N=2, L=1:
              After Step 1: root → A → B → C (terminal)
              Step 2 selects e.g. [root, A] and runs full chains from each:
                root → D → E (terminal)     ← new branch from root
                A → F → G (terminal)         ← new branch from A
              Tree now has 3 leaves: C, E, G

        Step 3 — Sample K leaves:
            Sample K leaves from the tree for training.

        Args:
            prompt_ids: Initial prompt token IDs.
            messages: Initial conversation messages.
            sampling_params: LLM sampling parameters.
            ground_truth: Ground truth for EM reward.

        Returns:
            List of K sampled leaf nodes with accumulated trajectories.
        """
        reset_node_id_counter()

        root = TreeNode(
            node_id=new_node_id(),
            messages=list(messages),
            prompt_ids=list(prompt_ids),
        )

        # Step 1: Run full chain from root
        await self.run_full_chain(root, sampling_params, ground_truth)

        # Step 2: Expansion iterations
        for iteration in range(self.ts_l):
            # Select N non-leaf nodes from ALL internal nodes (with replacement)
            expand_nodes = root.get_expand_nodes(self.ts_n, mode=self.expand_mode)
            if not expand_nodes:
                logger.warning(f"No non-leaf nodes for expansion at iteration {iteration}")
                break

            # Run full chains from each selected node in parallel
            chain_tasks = [
                self.run_full_chain(node, sampling_params, ground_truth)
                for node in expand_nodes
            ]
            await asyncio.gather(*chain_tasks)

        # Step 3: Sample K leaves
        sampled = TreeNode.sample_leaves(root, self.ts_k)

        # Accumulate root-to-leaf trajectories
        for leaf in sampled:
            leaf.accumulate_trajectory()

        return sampled

    def leaf_to_agent_loop_output(
        self, leaf: TreeNode, prompt_ids: list[int]
    ) -> AgentLoopOutput:
        """Convert a sampled leaf to AgentLoopOutput format."""
        response_ids = leaf.full_response_ids[:self.response_length]
        response_mask = leaf.full_response_mask[:self.response_length]
        response_logprobs = leaf.full_response_logprobs[:self.response_length]

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=sum(1 for m in leaf.messages if m.get("role") in ("assistant", "tool")),
            metrics={},
            extra_fields={
                "tree_reward": leaf.reward,
                "tree_depth": leaf.depth,
                "tree_terminal": leaf.is_terminal,
            },
        )
