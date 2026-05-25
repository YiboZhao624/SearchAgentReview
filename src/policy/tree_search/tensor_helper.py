"""Tensor helper utilities for tree search.

Ported from Tree-GRPO: handles padding, stacking, and attention mask creation
for variable-length sequences from tree leaves.
"""

from __future__ import annotations

import torch


class TensorHelper:
    """Utility class for tensor operations on tree search outputs."""

    @staticmethod
    def pad_and_stack(
        sequences: list[list[int]],
        pad_value: int = 0,
        max_length: int | None = None,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        """Pad variable-length sequences and stack into a tensor.

        Args:
            sequences: List of token ID lists.
            pad_value: Value to use for padding.
            max_length: Maximum length to pad to. If None, uses max sequence length.
            dtype: Output tensor dtype.

        Returns:
            Tensor of shape (batch_size, max_length).
        """
        if not sequences:
            return torch.empty(0, 0, dtype=dtype)

        lengths = [len(seq) for seq in sequences]
        if max_length is None:
            max_length = max(lengths)

        padded = torch.full((len(sequences), max_length), pad_value, dtype=dtype)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            padded[i, :length] = torch.tensor(seq[:length], dtype=dtype)
        return padded

    @staticmethod
    def create_attention_mask(
        prompt_lengths: list[int],
        response_lengths: list[int],
        total_length: int,
    ) -> torch.Tensor:
        """Create attention masks for prompt + response sequences.

        Args:
            prompt_lengths: Length of prompt for each sequence.
            response_lengths: Length of response for each sequence.
            total_length: Total padded sequence length.

        Returns:
            Attention mask tensor of shape (batch_size, total_length).
        """
        batch_size = len(prompt_lengths)
        mask = torch.zeros(batch_size, total_length, dtype=torch.long)
        for i in range(batch_size):
            valid_length = prompt_lengths[i] + response_lengths[i]
            mask[i, :valid_length] = 1
        return mask

    @staticmethod
    def create_response_mask_tensor(
        masks: list[list[int]],
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Pad and stack response masks.

        Args:
            masks: List of response mask lists (1=LLM token, 0=tool/pad).
            max_length: Maximum length to pad to.

        Returns:
            Response mask tensor of shape (batch_size, max_length).
        """
        return TensorHelper.pad_and_stack(masks, pad_value=0, max_length=max_length)

    @staticmethod
    def truncate_or_pad(
        sequence: list[int], target_length: int, pad_value: int = 0
    ) -> list[int]:
        """Truncate or pad a single sequence to target length.

        Args:
            sequence: Input token ID list.
            target_length: Desired length.
            pad_value: Value for padding.

        Returns:
            Sequence of exactly target_length.
        """
        if len(sequence) >= target_length:
            return sequence[:target_length]
        return sequence + [pad_value] * (target_length - len(sequence))
