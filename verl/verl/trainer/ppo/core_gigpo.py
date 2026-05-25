"""GiGPO advantage estimator adapted for per-trajectory batch format.

Faithful to the original GiGPO algorithm (https://arxiv.org/abs/2505.10978), with the
following adaptation: the original code operates on a per-step batch (each row = one
multi-turn step). Here we work with a per-trajectory batch (each row = full trajectory),
expanding to per-step internally and mapping advantages back to token positions.

Similarity-based anchor_obs grouping supports three backends:
  - "tfidf"              : TF-IDF cosine similarity (default, no extra deps)
  - "sequence_matcher"   : SequenceMatcher character ratio (original GiGPO default)
  - "embedding"          : semantic embedding via the local embedding HTTP service
"""

from __future__ import annotations

import uuid
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _sim_sequence_matcher(a: str, b: str, threshold: float) -> bool:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _build_sim_matrix_tfidf(texts: list[str]) -> np.ndarray:
    """Return cosine similarity matrix via TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf)
    except Exception:
        # Degenerate: treat every observation as unique
        return np.eye(len(texts))


def _build_sim_matrix_embedding(texts: list[str], service_url: str = "http://localhost:8000") -> np.ndarray:
    """Return cosine similarity matrix via the local embedding service."""
    import requests
    from sklearn.metrics.pairwise import cosine_similarity

    payload = {"input": texts, "model": "default"}
    resp = requests.post(f"{service_url}/v1/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]
    embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)
    return cosine_similarity(embeddings)


# ---------------------------------------------------------------------------
# Step-group construction  (faithful to GiGPO Eq. 6)
# ---------------------------------------------------------------------------

def build_step_group(
    anchor_obs: np.ndarray,          # (total_steps,)  strings
    episode_uid: np.ndarray,         # (total_steps,)  question-group IDs
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    similarity_method: str = "sequence_matcher",
    embedding_service_url: str = "http://localhost:8000",
    summarize: bool = False,
) -> np.ndarray:
    """Assign a step_group_uid to each step.

    Steps that share the same episode_uid AND whose anchor_obs are considered
    "the same state" (exact match, or similar under the chosen method) receive
    the same step_group_uid.
    """
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    group_sizes: list[int] = []

    for uid in np.unique(episode_uid):
        indices = np.where(episode_uid == uid)[0]
        obs_group = [str(anchor_obs[i]) for i in indices]
        n = len(obs_group)

        if n == 1 or not enable_similarity:
            # Exact-match clustering
            clusters: dict = defaultdict(list)
            for local_i, obs in enumerate(obs_group):
                clusters[obs].append(indices[local_i])
            for _, cluster_indices in clusters.items():
                cuid = str(uuid.uuid4())
                group_sizes.append(len(cluster_indices))
                for idx in cluster_indices:
                    step_group_uids[idx] = cuid
            continue

        # Similarity-based clustering
        if similarity_method == "sequence_matcher":
            # Greedy O(n^2) using SequenceMatcher
            cluster_reps: list[tuple[str, str]] = []  # (rep_obs, uid)
            obs_to_cuid: dict[int, str] = {}
            for local_i, obs in enumerate(obs_group):
                placed = False
                for rep_obs, cuid in cluster_reps:
                    if _sim_sequence_matcher(obs, rep_obs, similarity_thresh):
                        obs_to_cuid[local_i] = cuid
                        placed = True
                        break
                if not placed:
                    cuid = str(uuid.uuid4())
                    cluster_reps.append((obs, cuid))
                    obs_to_cuid[local_i] = cuid
        else:
            # Build similarity matrix first (tfidf or embedding)
            if similarity_method == "embedding":
                sim_matrix = _build_sim_matrix_embedding(obs_group, embedding_service_url)
            else:  # tfidf (default)
                sim_matrix = _build_sim_matrix_tfidf(obs_group)

            # Greedy clustering using the similarity matrix
            cluster_reps: list[tuple[int, str]] = []  # (rep_local_i, uid)
            obs_to_cuid: dict[int, str] = {}
            for local_i in range(n):
                placed = False
                for rep_local_i, cuid in cluster_reps:
                    if sim_matrix[local_i, rep_local_i] >= similarity_thresh:
                        obs_to_cuid[local_i] = cuid
                        placed = True
                        break
                if not placed:
                    cuid = str(uuid.uuid4())
                    cluster_reps.append((local_i, cuid))
                    obs_to_cuid[local_i] = cuid

        # Collect group sizes
        cuid_counts: dict[str, int] = defaultdict(int)
        for local_i in range(n):
            cuid_counts[obs_to_cuid[local_i]] += 1
        group_sizes.extend(cuid_counts.values())

        for local_i, original_idx in enumerate(indices):
            step_group_uids[original_idx] = obs_to_cuid[local_i]

    if None in step_group_uids or np.any(step_group_uids == None):  # noqa: E711
        missing = np.where(step_group_uids == None)[0]  # noqa: E711
        raise ValueError(f"Failed to assign step_group_uid at indices: {missing}")

    if summarize and group_sizes:
        counts = Counter(group_sizes)
        total = sum(counts.values())
        print("GiGPO step-group size summary:")
        for size in sorted(counts):
            print(f"  size={size}: {counts[size]} ({counts[size]/total:.1%})")
    avg = float(np.mean(group_sizes)) if group_sizes else 0.0
    print(f"GiGPO: avg step-group size = {avg:.2f}")
    return step_group_uids


# ---------------------------------------------------------------------------
# Episode-level advantage  (faithful to GiGPO Eq. 3)
# ---------------------------------------------------------------------------

def _episode_norm(
    outcome_scores: torch.Tensor,    # (bs,)  one per trajectory
    step_counts: list[int],          # [T_1, T_2, ..., T_bs]
    index: np.ndarray,               # (bs,)  question-group IDs
    compute_mean_std_cross_steps: bool = True,
    remove_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Returns per-trajectory episode advantage (bs,).

    With compute_mean_std_cross_steps=True (GiGPO default): each trajectory's
    outcome score is counted T_i times in the group statistics, matching the
    expanded per-step batch behaviour of the original implementation.
    """
    id2scores: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        bs = outcome_scores.shape[0]
        for i in range(bs):
            n_reps = step_counts[i] if compute_mean_std_cross_steps else 1
            for _ in range(n_reps):
                id2scores[index[i]].append(outcome_scores[i])

        for idx, vals in id2scores.items():
            t = torch.stack(vals)
            if len(vals) == 1:
                id2mean[idx] = t[0]
                id2std[idx] = torch.tensor(1.0, device=t.device)
            else:
                id2mean[idx] = t.mean()
                id2std[idx] = t.std()

        normed = torch.zeros(bs, dtype=outcome_scores.dtype, device=outcome_scores.device)
        for i in range(bs):
            idx = index[i]
            if remove_std:
                normed[i] = outcome_scores[i] - id2mean[idx]
            else:
                normed[i] = (outcome_scores[i] - id2mean[idx]) / (id2std[idx] + epsilon)
    return normed


# ---------------------------------------------------------------------------
# Step-level advantage  (faithful to GiGPO Eq. 7)
# ---------------------------------------------------------------------------

def _step_norm(
    step_rewards_flat: torch.Tensor,  # (total_steps,)
    step_group_uids: np.ndarray,      # (total_steps,)
    remove_std: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Returns per-step normalised advantage (total_steps,)."""
    id2scores: dict = defaultdict(list)
    id2mean: dict = {}
    id2std: dict = {}

    with torch.no_grad():
        n = step_rewards_flat.shape[0]
        for k in range(n):
            id2scores[step_group_uids[k]].append(step_rewards_flat[k])

        for gid, vals in id2scores.items():
            t = torch.stack(vals)
            if len(vals) == 1:
                id2mean[gid] = t[0]
                id2std[gid] = torch.tensor(1.0, device=t.device)
            else:
                id2mean[gid] = t.mean()
                id2std[gid] = t.std()

        normed = torch.zeros(n, dtype=step_rewards_flat.dtype, device=step_rewards_flat.device)
        for k in range(n):
            gid = step_group_uids[k]
            if remove_std:
                normed[k] = step_rewards_flat[k] - id2mean[gid]
            else:
                normed[k] = (step_rewards_flat[k] - id2mean[gid]) / (id2std[gid] + epsilon)
    return normed


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_grpo_gigpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    anchor_obs_per_traj: Optional[list] = None,
    step_scores_per_traj: Optional[list] = None,
    gamma: float = 0.9,
    step_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
    similarity_method: str = "sequence_matcher",
    embedding_service_url: str = "http://localhost:8000",
    compute_mean_std_cross_steps: bool = True,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GiGPO advantage for per-trajectory batch format.

    Differs from the original GiGPO only in that the input batch has one row
    per trajectory (not one row per step). Internally expands to per-step,
    runs GiGPO episode + step normalisation, then maps back.

    Args:
        token_level_rewards: (bs, seq_len) — outcome reward at last response token.
        response_mask:        (bs, seq_len) — 1 for LLM-generated tokens.
        index:                (bs,) — question-group IDs for GRPO normalisation.
        anchor_obs_per_traj:  list[list[str]] — anchor obs per step per trajectory.
                              If None, each step is treated as its own group.
        step_scores_per_traj: list[list[float]] — per-step LLM judge scores per trajectory.
                              If provided, used directly as step rewards instead of
                              discounted returns from the outcome score (Eq. 5).
        gamma:                Discount factor for step-level discounted returns.
                              Only used when step_scores_per_traj is None.
        step_advantage_w:     Weight for step-level advantage (GiGPO Eq. 8).
        mode:                 "mean_std_norm" (divide by std) or "mean_norm" (mean only).
        enable_similarity:    Use similarity-based step grouping (else exact-match).
        similarity_thresh:    Similarity threshold for grouping.
        similarity_method:    "tfidf" | "sequence_matcher" | "embedding".
        embedding_service_url: URL for embedding service (used when similarity_method="embedding").
        compute_mean_std_cross_steps: Match GiGPO's cross-steps statistics (default True).
    """
    from verl.trainer.ppo.core_algos import _get_step_boundaries

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'mean_std_norm' or 'mean_norm'.")

    bs, seq_len = token_level_rewards.shape
    device = token_level_rewards.device

    # Outcome score per trajectory: sum of token-level rewards
    outcome_scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Step boundaries: list[list[(start, end_incl)]] per trajectory
    step_boundaries = _get_step_boundaries(response_mask)

    # How many steps each trajectory has
    step_counts = [len(bounds) for bounds in step_boundaries]

    # ------------------------------------------------------------------ #
    # Episode-level advantage (Eq. 3)  — per trajectory, then broadcast  #
    # ------------------------------------------------------------------ #
    episode_adv_per_traj = _episode_norm(
        outcome_scores=outcome_scores,
        step_counts=step_counts,
        index=index,
        compute_mean_std_cross_steps=compute_mean_std_cross_steps,
        remove_std=remove_std,
        epsilon=epsilon,
    )  # (bs,)

    # ------------------------------------------------------------------ #
    # Expand to per-step representation for step-level processing         #
    # ------------------------------------------------------------------ #
    # flat lists: one entry per (traj, step)
    flat_episode_uids: list = []
    flat_traj_local_idx: list[tuple[int, int]] = []   # (traj_i, step_t)
    flat_step_rewards: list[float] = []
    flat_anchor_obs: list[str] = []

    use_process_reward = step_scores_per_traj is not None

    for i in range(bs):
        bounds = step_boundaries[i]
        T_i = len(bounds)
        if T_i == 0:
            continue
        r_o = outcome_scores[i].item()
        # Get per-step LLM judge scores for this trajectory (if available)
        traj_step_scores = None
        if use_process_reward and i < len(step_scores_per_traj):
            traj_step_scores = step_scores_per_traj[i]
            if not isinstance(traj_step_scores, (list, tuple)):
                traj_step_scores = list(traj_step_scores) if traj_step_scores is not None else None

        for t in range(T_i):
            if traj_step_scores is not None and t < len(traj_step_scores):
                # Use actual LLM judge per-step score
                G = float(traj_step_scores[t])
            else:
                # Fallback: discounted return from outcome (Eq. 5)
                G = (gamma ** (T_i - 1 - t)) * r_o
            flat_step_rewards.append(G)
            flat_episode_uids.append(index[i])
            flat_traj_local_idx.append((i, t))
            if anchor_obs_per_traj is not None and i < len(anchor_obs_per_traj):
                obs_list = anchor_obs_per_traj[i]
                obs = str(obs_list[t]) if t < len(obs_list) else ""
            else:
                # Fallback: use "question_id::step_t" as a unique key → no grouping
                obs = f"{index[i]}::step_{t}"
            flat_anchor_obs.append(obs)

    if not flat_step_rewards:
        # No steps found — return zero advantages
        zeros = torch.zeros(bs, seq_len, dtype=token_level_rewards.dtype, device=device)
        return zeros, zeros

    total_steps = len(flat_step_rewards)
    step_rewards_tensor = torch.tensor(flat_step_rewards, dtype=torch.float32, device=device)
    flat_episode_uids_arr = np.array(flat_episode_uids, dtype=object)
    flat_anchor_obs_arr = np.array(flat_anchor_obs, dtype=object)

    # ------------------------------------------------------------------ #
    # Anchor-obs grouping  (Eq. 6)                                         #
    # ------------------------------------------------------------------ #
    step_group_uids = build_step_group(
        anchor_obs=flat_anchor_obs_arr,
        episode_uid=flat_episode_uids_arr,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
        similarity_method=similarity_method,
        embedding_service_url=embedding_service_url,
    )

    # ------------------------------------------------------------------ #
    # Step-level advantage  (Eq. 7)                                        #
    # ------------------------------------------------------------------ #
    step_adv_flat = _step_norm(
        step_rewards_flat=step_rewards_tensor,
        step_group_uids=step_group_uids,
        remove_std=remove_std,
        epsilon=epsilon,
    )  # (total_steps,)

    # ------------------------------------------------------------------ #
    # Combine and map back to token positions  (Eq. 8)                    #
    # ------------------------------------------------------------------ #
    advantages = torch.zeros(bs, seq_len, dtype=token_level_rewards.dtype, device=device)

    for k, (i, t) in enumerate(flat_traj_local_idx):
        bounds = step_boundaries[i]
        if t >= len(bounds):
            continue
        start, end = bounds[t]
        ep_adv = episode_adv_per_traj[i].item()
        st_adv = step_adv_flat[k].item()
        advantages[i, start: end + 1] = ep_adv + step_advantage_w * st_adv

    advantages = advantages * response_mask
    return advantages, advantages
