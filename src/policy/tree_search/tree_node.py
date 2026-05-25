"""TreeNode for tree-structured rollouts.

Ported from Tree-GRPO (ICLR 2026): Tree-GRPO/search_r1/llm_agent/tree_node.py
Each node represents a state in the search tree (a partial trajectory).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the tree search rollout.

    Attributes:
        node_id: Unique identifier for this node.
        parent: Parent node (None for root).
        children: Child nodes created by expansion.
        depth: Depth in the tree (root = 0).
        messages: Conversation history up to this node.
        response_ids: Token IDs of the LLM response at this node.
        response_mask: Mask for the response (1=LLM, 0=tool).
        response_logprobs: Per-token log probabilities.
        observation: Tool response text at this node.
        is_terminal: Whether this node represents a finished trajectory.
        reward: Reward score (set only for terminal nodes).
        prompt_ids: Full accumulated prompt token IDs.
    """

    node_id: int
    parent: Optional[TreeNode] = None
    children: list[TreeNode] = field(default_factory=list)
    depth: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    observation: str = ""
    is_terminal: bool = False
    reward: float = 0.0
    prompt_ids: list[int] = field(default_factory=list)

    # Accumulated trajectory data (root-to-this-node)
    full_response_ids: list[int] = field(default_factory=list)
    full_response_mask: list[int] = field(default_factory=list)
    full_response_logprobs: list[float] = field(default_factory=list)

    def add_child(self, child: TreeNode) -> None:
        """Add a child node to this node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if this node is the root."""
        return self.parent is None

    def get_path_from_root(self) -> list[TreeNode]:
        """Get the path from root to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes in the subtree rooted at this node."""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_terminal_leaves(self) -> list[TreeNode]:
        """Get all terminal leaf nodes."""
        return [leaf for leaf in self.get_leaves() if leaf.is_terminal]

    def accumulate_trajectory(self) -> None:
        """Accumulate response_ids, response_mask, and logprobs from root to this node.

        Populates full_response_ids, full_response_mask, full_response_logprobs.
        """
        path = self.get_path_from_root()
        self.full_response_ids = []
        self.full_response_mask = []
        self.full_response_logprobs = []
        for node in path:
            self.full_response_ids.extend(node.response_ids)
            self.full_response_mask.extend(node.response_mask)
            self.full_response_logprobs.extend(node.response_logprobs)

    @staticmethod
    def sample_leaves(root: TreeNode, k: int) -> list[TreeNode]:
        """Sample K leaves from the tree.

        Prefers terminal leaves. If not enough terminal leaves exist,
        falls back to non-terminal leaves. Never duplicates — returns
        fewer than K if the tree doesn't have enough leaves.

        Args:
            root: Root node of the tree.
            k: Number of leaves to sample.

        Returns:
            List of up to K sampled leaf nodes (no duplicates).
        """
        terminal = root.get_terminal_leaves()
        non_terminal = [leaf for leaf in root.get_leaves() if not leaf.is_terminal]

        # Prefer terminal leaves
        if len(terminal) >= k:
            return random.sample(terminal, k)

        # Not enough terminal leaves; take all terminal + sample non-terminal
        selected = list(terminal)
        remaining_k = k - len(selected)
        if non_terminal:
            remaining_k = min(remaining_k, len(non_terminal))
            selected.extend(random.sample(non_terminal, remaining_k))

        if len(selected) < k:
            logger.warning(
                f"Tree has only {len(selected)} leaves but K={k} requested. "
                f"Check tree_search params: need 1 + N*L >= K."
            )

        return selected

    def get_all_non_leaf_nodes(self) -> list[TreeNode]:
        """Get all non-leaf nodes in the subtree (including self if non-leaf).

        In Tree-GRPO, expansion candidates are ALL non-leaf nodes (internal nodes),
        not just leaf nodes. This matches the original get_expand_node() which uses
        `random.choices(candidate_set)` over all non-leaf nodes.
        """
        result = []
        if not self.is_leaf():
            result.append(self)
        for child in self.children:
            result.extend(child.get_all_non_leaf_nodes())
        return result

    def get_expand_nodes(self, n: int, mode: str = "random") -> list[TreeNode]:
        """Select N nodes for expansion (matching original Tree-GRPO algorithm).

        Uses random.choices (WITH replacement) from all non-leaf nodes in the tree.
        This means the same node can be selected multiple times, which creates
        multiple new branches from that node.

        Args:
            n: Number of nodes to select.
            mode: Selection strategy ("random" only for now).

        Returns:
            List of N selected nodes (may contain duplicates).
        """
        candidates = self.get_all_non_leaf_nodes()
        if not candidates:
            return []
        # random.choices allows replacement (same node selected multiple times)
        return random.choices(candidates, k=n)

    def __repr__(self) -> str:
        status = "terminal" if self.is_terminal else "active"
        return (
            f"TreeNode(id={self.node_id}, depth={self.depth}, "
            f"children={len(self.children)}, {status})"
        )


# Global node ID counter for unique IDs within a tree
_node_id_counter = 0


def new_node_id() -> int:
    """Generate a unique node ID."""
    global _node_id_counter
    _node_id_counter += 1
    return _node_id_counter


def reset_node_id_counter() -> None:
    """Reset the node ID counter (call at start of each tree)."""
    global _node_id_counter
    _node_id_counter = 0
