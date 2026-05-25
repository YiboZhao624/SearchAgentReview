#!/usr/bin/env bash
# Tree-GRPO training script
# Advantage estimator: tree_grpo (tree-structured rollouts with shared prefix)
# Reward: EM/F1 (rule-based)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -euo pipefail

PROJECT_NAME="tree_grpo"
EXPERIMENT_NAME="$(date +%m%d)-Qwen3-8B-TreeGRPO"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROOT_DIR
CKPT_ROOT_DIR="${CKPT_ROOT_DIR:-${ROOT_DIR}/checkpoints}"

CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs}"
CONFIG_NAME="${CONFIG_NAME:-tree_grpo}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen3-8B}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT_DIR}/data/train.jsonl}"
VAL_FILE="${VAL_FILE:-${ROOT_DIR}/data/val.jsonl}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# Tree search parameters
# Leaves per tree: 1 (initial chain) + N × L = 1 + 2 × 2 = 5, sample K=4
# Training budget: M × K = 2 × 4 = 8 samples/prompt (matches group_size=8)
TS_M="${TS_M:-2}"       # Number of trees per prompt
TS_N="${TS_N:-2}"       # Nodes to expand per iteration
TS_L="${TS_L:-2}"       # Expansion iterations (1+N*L=5 leaves/tree)
TS_K="${TS_K:-4}"       # Leaves to sample per tree

RAY_PORT="${RAY_PORT:-6382}"
ray start --head --port="${RAY_PORT}" --dashboard-port 8269
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

cleanup_ray() {
    echo "Cleaning up Ray cluster on port ${RAY_PORT}..."
    ray stop --address="127.0.0.1:${RAY_PORT}" 2>/dev/null || true
    lsof -ti:${RAY_PORT} | xargs -r kill -9 2>/dev/null || true
}
trap cleanup_ray EXIT INT TERM

mkdir -p "${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

python -m verl.trainer.main_ppo \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.n="${TS_M}" \
  actor_rollout_ref.rollout.custom.tree_search.ts_m="${TS_M}" \
  actor_rollout_ref.rollout.custom.tree_search.ts_n="${TS_N}" \
  actor_rollout_ref.rollout.custom.tree_search.ts_l="${TS_L}" \
  actor_rollout_ref.rollout.custom.tree_search.ts_k="${TS_K}" \
  trainer.save_freq=20 \
  trainer.test_freq=20 \
  trainer.total_epochs=1 \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
  actor_rollout_ref.rollout.response_length=4096 \
  algorithm.adv_estimator=tree_grpo \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/rollout_data" \
  trainer.validation_data_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/validation_data" \
  trainer.log_val_generations=10 \
  "$@" \
  2>&1 | tee "${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/train.log"
