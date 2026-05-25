"""IGPO (Information Gain-based Policy Optimization) reward functions.

Two main components:
1. compute_info_gain_from_rollout() — post-rollout info gain computation via pseudo-sequence log probs
2. igpo_compute_score() — reward function for NaiveRewardManager (scalar F1)

Token placement of IG rewards is done post-reward in ray_trainer.py.
"""

import math
import re
import string
from collections import Counter
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# F1 / EM helpers (adapted from R1_searcher_reward.py)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")

# Format penalty for malformed tags (matching original IGPO baseline)
_FORMAT_PENALTY = -2.0

# Tags that must be properly balanced (open/close counts must match)
_BALANCED_TAGS = ("code", "tool_call", "think", "answer")


def _check_tags_balance(text: str) -> bool:
    """Return True if all relevant XML tags are properly paired."""
    for tag in _BALANCED_TAGS:
        if text.count(f"<{tag}>") != text.count(f"</{tag}>"):
            return False
    return True


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _f1_word_score(prediction: str, reference: str) -> float:
    pred_words = _normalize_text(prediction).split()
    ref_words = _normalize_text(reference).split()
    pn, rn = len(pred_words), len(ref_words)
    if pn + rn == 0:
        return 0.0
    common = sum((Counter(pred_words) & Counter(ref_words)).values())
    return (2.0 * common) / (pn + rn)


def _extract_answer(solution_str: str) -> str:
    """Extract text from the first <answer>...</answer> tag."""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _compute_f1(solution_str: str, ground_truth: str) -> float:
    # Check tag balance first — format penalty if malformed
    if not _check_tags_balance(solution_str):
        return _FORMAT_PENALTY
    predicted = _extract_answer(solution_str)
    if not predicted:
        return _FORMAT_PENALTY
    # Handle multi-label ground truth
    targets = ground_truth.split("<|answer_split|>") if "<|answer_split|>" in ground_truth else [ground_truth]
    return max(_f1_word_score(predicted, t) for t in targets)


def _compute_em(solution_str: str, ground_truth: str) -> float:
    if not _check_tags_balance(solution_str):
        return 0.0
    predicted = _normalize_text(_extract_answer(solution_str))
    if not predicted:
        return 0.0
    targets = ground_truth.split("<|answer_split|>") if "<|answer_split|>" in ground_truth else [ground_truth]
    return 1.0 if any(_normalize_text(t) == predicted for t in targets) else 0.0


# ---------------------------------------------------------------------------
# Token-position helpers
# ---------------------------------------------------------------------------

def _char_pos_to_token_idx(char_pos: int, offset_mapping: list) -> int:
    """Find token index for a character position using offset_mapping."""
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos < end:
            return i
        if char_pos < start:
            return max(0, i - 1)
    return len(offset_mapping) - 1


TURN_SEPARATOR = "\n<|im_start|>assistant\n"


def _find_turn_boundaries(response_str: str, tokenizer) -> tuple[list[int], int]:
    """Find turn-end token positions in response.

    Returns:
        turn_end_positions: list of token indices (in response token space) marking end of each non-final turn
        num_turns: total number of turns detected
    """
    encoding = tokenizer(response_str, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding["offset_mapping"]
    tokens_size = len(encoding["input_ids"])

    if tokens_size == 0:
        return [], 1

    # Find separator positions in character space
    sep_positions = []
    search_pos = 0
    while True:
        pos = response_str.find(TURN_SEPARATOR, search_pos)
        if pos == -1:
            break
        sep_positions.append(pos)
        search_pos = pos + 1

    if not sep_positions:
        return [], 1

    # Build turn structure
    turn_start_chars = []
    turn_end_chars = []

    if sep_positions[0] > 0:
        turn_start_chars.append(0)
        turn_end_chars.append(sep_positions[0])

    for i, sep_pos in enumerate(sep_positions):
        turn_start_chars.append(sep_pos + len(TURN_SEPARATOR))
        turn_end_chars.append(sep_positions[i + 1] if i + 1 < len(sep_positions) else len(response_str))

    num_turns = len(turn_start_chars)
    if num_turns <= 1:
        return [], num_turns

    # Map turn-end char positions to token positions (for non-final turns only)
    turn_end_token_positions = []
    for i in range(num_turns - 1):
        end_char = turn_end_chars[i]
        if end_char > 0:
            tok_idx = _char_pos_to_token_idx(end_char - 1, offset_mapping)
        else:
            tok_idx = 0
        turn_end_token_positions.append(min(tok_idx, tokens_size - 1))

    return turn_end_token_positions, num_turns


# ---------------------------------------------------------------------------
# Post-rollout info gain computation
# ---------------------------------------------------------------------------

# GT answer pseudo-response format (same as IGPO baseline)
# Turn 0 has no prior <think> open tag, so we must include it.
# Turn k>=1 already has an open <think> from the response, so we just close it.
_GT_PREFIX_TURN0 = "\n<think>\nNow there's enough information to answer\n</think>\n<answer>\n"
_GT_PREFIX = "\nNow there's enough information to answer\n</think>\n<answer>\n"
_GT_SUFFIX = "\n</answer><|im_end|>"


def _prepare_gt_tokens(ground_truth_dict: dict, tokenizer, *, is_turn0: bool = False):
    """Tokenize GT answer with prefix/suffix; return token IDs and GT token range."""
    gt_text = ground_truth_dict.get("target", [""])[0] if isinstance(ground_truth_dict.get("target"), list) else str(ground_truth_dict.get("target", ""))

    prefix = _GT_PREFIX_TURN0 if is_turn0 else _GT_PREFIX
    full_text = f"{prefix}{gt_text}{_GT_SUFFIX}"
    encoding = tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
    token_ids = encoding["input_ids"][0].tolist()
    offset_mapping = encoding["offset_mapping"][0].tolist()

    if not token_ids:
        return token_ids, 0, 0

    gt_char_start = len(prefix)
    gt_char_end = gt_char_start + len(gt_text)

    gt_token_start = None
    gt_token_end = None
    for idx, (cs, ce) in enumerate(offset_mapping):
        if gt_token_start is None and ce > gt_char_start:
            gt_token_start = idx
        if cs < gt_char_end and ce > 0:
            gt_token_end = idx + 1

    gt_token_start = gt_token_start if gt_token_start is not None else len(token_ids)
    gt_token_end = gt_token_end if gt_token_end is not None else len(token_ids)

    return token_ids, gt_token_start, gt_token_end


def compute_info_gain_from_rollout(
    batch,
    actor_rollout_wg,
    tokenizer,
    info_gain_type: str = "log_prob_diff",
    micro_batch_size: int = 64,
    dp_size: int = 8,
) -> list[dict[str, Any]]:
    """Compute info gain rewards post-rollout.

    For each (sample, turn), constructs a pseudo-sequence:
        context_up_to_turn + GT_answer_tokens
    and computes P(GT | context) via actor model log probs.

    Args:
        batch: DataProto with rollout results.
        actor_rollout_wg: Ray worker group for actor model.
        tokenizer: HuggingFace tokenizer.
        info_gain_type: "log_prob_diff" or "prob_diff".
        micro_batch_size: chunk size for compute_log_prob calls.
        dp_size: data-parallel world size (batch must be divisible by this).

    Returns:
        list[dict] per sample with keys:
            - "info_gains": list[float] — one IG value per intermediate turn
            - "turn_end_response_positions": list[int] — response-relative token positions for IG placement
    """
    from verl import DataProto

    input_ids = batch.batch["input_ids"]    # (bs, total_len)
    attention_mask = batch.batch["attention_mask"]  # (bs, total_len)
    responses = batch.batch["responses"]    # (bs, resp_len)
    prompts = batch.batch["prompts"]        # (bs, prompt_len)
    bsz = input_ids.shape[0]
    prompt_len = prompts.shape[-1]
    resp_len = responses.shape[-1]

    # --- 1. Decode responses and find turn boundaries ---
    results = []
    pseudo_sequence_specs = []  # (sample_idx, turn_idx, context_token_ids, gt_token_ids, gt_start, gt_end)

    for i in range(bsz):
        # Get valid response tokens
        resp_mask = batch.batch["attention_mask"][i, prompt_len:]
        valid_resp_len = int(resp_mask.sum().item())
        valid_resp_ids = responses[i, :valid_resp_len]

        # Decode with special tokens to find turn boundaries
        response_str = tokenizer.decode(valid_resp_ids, skip_special_tokens=False)

        # Find turn boundaries
        turn_end_positions, num_turns = _find_turn_boundaries(response_str, tokenizer)

        # Get ground truth — prepare GT tokens for turn 0 (with <think>) and others (without)
        gt_str = batch.non_tensor_batch["reward_model"][i]["ground_truth"]
        gt_token_ids_t0, gt_start_t0, gt_end_t0 = _prepare_gt_tokens(gt_str, tokenizer, is_turn0=True)
        gt_token_ids, gt_start, gt_end = _prepare_gt_tokens(gt_str, tokenizer, is_turn0=False)

        result = {
            "info_gains": [],
            "turn_end_response_positions": turn_end_positions,
            "num_turns": num_turns,
        }
        results.append(result)

        if num_turns <= 1 or not gt_token_ids or gt_start >= gt_end:
            # Single turn or empty GT — no info gain to compute
            continue

        # For each turn (including turn 0 for baseline value), construct pseudo-sequence
        # We need P(GT|context) at turn_start_0 (before any tool use) and at each subsequent turn start
        # Turn boundary positions in input_ids space:
        # Turn 0 starts at: prompt_len (start of response)
        # Turn k starts at: prompt_len + turn_end_positions[k-1] + offset

        # Get valid prompt tokens
        prompt_attn = attention_mask[i, :prompt_len]
        valid_prompt_len = int(prompt_attn.sum().item())
        valid_prompt_ids = prompts[i, prompt_len - valid_prompt_len:]  # remove left padding

        # We need num_turns context snapshots:
        # snapshot 0: just the prompt (before any generation)
        # snapshot k: prompt + response up to turn_end_positions[k-1]
        # The info_gain for turn k = value(snapshot k) - value(snapshot k-1)

        # Find turn boundary positions by looking for separator in token IDs
        # Use the same character-based approach as _find_turn_boundaries but map to input_ids positions
        sep_char_positions = []
        search_pos = 0
        while True:
            pos = response_str.find(TURN_SEPARATOR, search_pos)
            if pos == -1:
                break
            sep_char_positions.append(pos)
            search_pos = pos + 1

        # Token positions in the response for separator starts
        resp_encoding = tokenizer(response_str, return_offsets_mapping=True, add_special_tokens=False)
        resp_offset_map = resp_encoding["offset_mapping"]

        # Context snapshots: we take context at turn START positions
        # Snapshot 0: just prompt (before generation)
        # Snapshot k (k>=1): prompt + response up to separator k-1
        context_end_positions = [0]  # snapshot 0: 0 response tokens
        for sep_char_pos in sep_char_positions:
            # Map separator start char to token position
            if sep_char_pos > 0:
                tok_pos = _char_pos_to_token_idx(sep_char_pos - 1, resp_offset_map)
                context_end_positions.append(tok_pos + 1)  # include the token at sep_char_pos-1
            else:
                context_end_positions.append(0)

        gt_ids_tensor = torch.tensor(gt_token_ids, dtype=torch.long)
        gt_ids_tensor_t0 = torch.tensor(gt_token_ids_t0, dtype=torch.long)

        for turn_idx in range(min(len(context_end_positions), num_turns)):
            resp_end = context_end_positions[turn_idx]
            # Context = valid_prompt + response[:resp_end]
            if resp_end > 0:
                context_ids = torch.cat([valid_prompt_ids, valid_resp_ids[:resp_end]])
            else:
                context_ids = valid_prompt_ids.clone()

            # Turn 0 uses prefix with <think> (no prior open tag), others close existing <think>
            if turn_idx == 0:
                pseudo_sequence_specs.append((i, turn_idx, context_ids, gt_ids_tensor_t0, gt_start_t0, gt_end_t0))
            else:
                pseudo_sequence_specs.append((i, turn_idx, context_ids, gt_ids_tensor, gt_start, gt_end))

    if not pseudo_sequence_specs:
        return results

    # --- 2. Batch construct pseudo-sequences and compute log probs ---
    all_info_gains_by_sample = {i: {} for i in range(bsz)}  # sample_idx -> {turn_idx: value}

    # Process in chunks to avoid OOM
    for chunk_start in range(0, len(pseudo_sequence_specs), micro_batch_size):
        chunk = pseudo_sequence_specs[chunk_start:chunk_start + micro_batch_size]

        # Find max lengths for padding
        max_ctx_len = max(len(spec[2]) for spec in chunk)
        max_gt_len = max(len(spec[3]) for spec in chunk)
        max_total_len = max_ctx_len + max_gt_len
        chunk_size = len(chunk)

        # Construct padded tensors
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_input_ids = torch.full((chunk_size, max_total_len), pad_id, dtype=torch.long)
        padded_attn_mask = torch.zeros((chunk_size, max_total_len), dtype=torch.long)
        padded_position_ids = torch.zeros((chunk_size, max_total_len), dtype=torch.long)
        padded_responses = torch.full((chunk_size, max_gt_len), pad_id, dtype=torch.long)

        gt_ranges = []  # (gt_start, gt_end) per item in chunk

        for j, (sample_idx, turn_idx, ctx_ids, gt_ids, gt_s, gt_e) in enumerate(chunk):
            ctx_len = len(ctx_ids)
            this_gt_len = len(gt_ids)
            total_len = ctx_len + this_gt_len

            # Left-pad: place content at the end
            offset = max_total_len - total_len
            padded_input_ids[j, offset:offset + ctx_len] = ctx_ids
            padded_input_ids[j, offset + ctx_len:offset + total_len] = gt_ids
            padded_attn_mask[j, offset:offset + total_len] = 1
            padded_position_ids[j, offset:offset + total_len] = torch.arange(total_len)

            # Responses: left-pad GT tokens
            gt_offset = max_gt_len - this_gt_len
            padded_responses[j, gt_offset:] = gt_ids

            gt_ranges.append((gt_offset + gt_s, gt_offset + gt_e))

        # Pad batch to be divisible by dp_size (compute_log_prob splits across DP workers)
        pad_count = (dp_size - chunk_size % dp_size) % dp_size
        padded_size = chunk_size + pad_count
        if pad_count > 0:
            # Pad input_ids: clone first pad_count rows (content doesn't matter, attn_mask=0)
            pad_input = padded_input_ids[:1].expand(pad_count, -1).clone()
            padded_input_ids = torch.cat([padded_input_ids, pad_input], dim=0)
            # Pad attention_mask: all zeros so padded rows are ignored
            padded_attn_mask = torch.cat([padded_attn_mask, torch.zeros(pad_count, max_total_len, dtype=torch.long)], dim=0)
            # Pad position_ids: zeros
            padded_position_ids = torch.cat([padded_position_ids, torch.zeros(pad_count, max_total_len, dtype=torch.long)], dim=0)
            # Pad responses: clone first row
            pad_resp = padded_responses[:1].expand(pad_count, -1).clone()
            padded_responses = torch.cat([padded_responses, pad_resp], dim=0)

        assert padded_input_ids.shape[0] == padded_attn_mask.shape[0] == padded_position_ids.shape[0] == padded_responses.shape[0] == padded_size, (
            f"[IGPO] Padding mismatch: input_ids={padded_input_ids.shape}, attn_mask={padded_attn_mask.shape}, "
            f"pos_ids={padded_position_ids.shape}, responses={padded_responses.shape}, expected_size={padded_size}"
        )

        # Create DataProto for compute_log_prob
        pseudo_batch = DataProto.from_dict({
            "input_ids": padded_input_ids,
            "attention_mask": padded_attn_mask,
            "position_ids": padded_position_ids,
            "responses": padded_responses,
        })

        # Call compute_log_prob
        try:
            log_prob_output = actor_rollout_wg.compute_log_prob(pseudo_batch)
            old_log_probs = log_prob_output.batch["old_log_probs"][:chunk_size]  # strip padding rows
        except Exception as e:
            import traceback
            print(f"[IGPO] compute_log_prob failed (chunk {chunk_start}/{len(pseudo_sequence_specs)}): "
                  f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            continue

        # Extract mean log prob for GT tokens
        for j, (sample_idx, turn_idx, _, _, _, _) in enumerate(chunk):
            gt_s, gt_e = gt_ranges[j]
            if gt_s >= gt_e:
                continue
            log_probs = old_log_probs[j, gt_s:gt_e]
            mean_lp = log_probs.mean().item()

            if math.isnan(mean_lp) or math.isinf(mean_lp):
                continue

            if info_gain_type == "log_prob_diff":
                value = mean_lp
            else:  # prob_diff
                value = math.exp(mean_lp)

            all_info_gains_by_sample[sample_idx][turn_idx] = value

    # --- 3. Compute info gain differences ---
    for i in range(bsz):
        values = all_info_gains_by_sample[i]
        num_turns = results[i]["num_turns"]
        info_gains = []

        for t in range(1, num_turns):
            if t in values and (t - 1) in values:
                ig = values[t] - values[t - 1]
                if math.isnan(ig) or math.isinf(ig):
                    ig = 0.0
                info_gains.append(ig)
            else:
                info_gains.append(0.0)

        results[i]["info_gains"] = info_gains

    return results


# ---------------------------------------------------------------------------
# Reward function for NaiveRewardManager
# ---------------------------------------------------------------------------

def igpo_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """IGPO reward function — returns scalar F1 score for NaiveRewardManager.

    Token-level IG placement is handled separately in ray_trainer.py post-reward.
    This function just computes the outcome (F1/EM) score.
    """
    if solution_str is None:
        solution_str = ""

    # Handle ground_truth format
    if isinstance(ground_truth, dict):
        gt_str = ground_truth.get("target", [""])[0] if isinstance(ground_truth.get("target"), list) else str(ground_truth.get("target", ""))
    elif isinstance(ground_truth, str):
        gt_str = ground_truth
    else:
        gt_str = str(ground_truth)

    f1 = _compute_f1(solution_str, gt_str)
    em = _compute_em(solution_str, gt_str)

    return {"score": f1, "success": int(em == 1.0), "f1": f1, "em": em}


# ---------------------------------------------------------------------------
# Post-reward IG token placement (called from ray_trainer.py)
# ---------------------------------------------------------------------------

def place_igpo_token_rewards(batch, ig_results: list[dict]) -> None:
    """Place info gain rewards on turn-end tokens in token_level_rewards.

    Modifies batch.batch["token_level_rewards"] in-place.
    F1 is already on the last token (placed by NaiveRewardManager).
    This function adds IG values at intermediate turn-end positions.
    """
    token_level_rewards = batch.batch["token_level_rewards"]

    for i, ig_result in enumerate(ig_results):
        info_gains = ig_result.get("info_gains", [])
        positions = ig_result.get("turn_end_response_positions", [])

        for ig_val, pos in zip(info_gains, positions):
            if pos < token_level_rewards.shape[1]:
                # Use 1e-10 sentinel for exact-zero IG so the turn boundary is still
                # visible to the advantage computation (which uses != 0 to detect turns).
                if ig_val == 0.0:
                    ig_val = 1e-10
                token_level_rewards[i, pos] = ig_val
