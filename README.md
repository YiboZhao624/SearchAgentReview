# SearchAgentReview

Code for the empirical review paper on search agent training methods. This repository provides unified implementations of **Search-R1**, **GiGPO**, **IGPO**, and **Tree-GRPO** within a single verl-based training framework.

## Methods

| Method | Advantage Estimator | Key Idea |
|--------|-------------------|----------|
| **Search-R1** | Standard GRPO | Multi-turn search agent with EM reward |
| **GiGPO** | `grpo_gigpo` | Discounted step-level returns + anchor observation grouping |
| **IGPO** | `grpo_igpo` | Information-gain weighted advantage estimation |
| **Tree-GRPO** | `tree_grpo` | Tree-structured rollouts with shared-prefix advantage |

## Setup

### Prerequisites

- 8× GPUs (A100/H100 recommended)
- Python 3.10+
- CUDA 12.x

### Installation

We suggest installing VeRL package following the official guidance. For the search service, we suggest create two conda environment for VLLM embedding service and the Faiss Search service, respectively.

```
conda create -n vllm11 python=3.12
conda activate vllm11
pip install vllm==0.11.0
```

```
conda create -n faiss-gpu python=3.12
conda activate faiss-gpu
pip install faiss-gpu
```

### Data Preparation

Place your training data in the `data/` directory:
- `data/train.jsonl` — training queries
- `data/val.jsonl` — validation queries

Each line should be a JSON object with fields: `question`, `answer`, `supporting_facts` (optional).

Use `src/data/multi_dataset_prep.py` to prepare multi-dataset mixtures from HotpotQA, 2WikiMultihopQA, MuSiQue, BamboogLe, and PopQA.

### Model

Download [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to `models/Qwen3-8B/`.

### Retrieval Service

1. Start the embedding backends (requires Qwen3-Embedding-8B model):
```bash
./scripts/start_embedding_service.sh
```

2. Start the FAISS search service:
```bash
./scripts/start_search_service.sh
```

## Training

Each method has a dedicated training script:

```bash
# Search-R1 (baseline)
./scripts/train_search_r1.sh

# GiGPO
./scripts/train_gigpo.sh

# IGPO
./scripts/train_igpo.sh

# Tree-GRPO
./scripts/train_tree_grpo.sh
```

### Configuration

Training configs are in `configs/`. Key parameters can be overridden via environment variables:

```bash
export MODEL_PATH=/path/to/model
export TRAIN_FILE=/path/to/train.jsonl
export VAL_FILE=/path/to/val.jsonl
export CKPT_ROOT_DIR=/path/to/checkpoints
export GPUS_PER_NODE=8
```

## Project Structure

```
SearchAgentReview/
├── configs/                  # Hydra training configs (one per method)
│   ├── search_r1.yaml
│   ├── gigpo.yaml
│   ├── igpo.yaml
│   ├── tree_grpo.yaml
│   └── agent_loop/           # Agent loop definitions
├── scripts/                  # Training & service scripts
├── src/
│   ├── data/                 # Dataset classes & preparation
│   ├── policy/
│   │   ├── agent_loop/       # Multi-turn agent loop implementations
│   │   ├── tools/            # Search tool definitions & embedding service
│   │   └── tree_search/      # Tree-GRPO tree search logic
│   └── reward/               # Reward functions (EM/F1, IGPO info-gain)
└── verl/                     # Training framework (forked verl with method implementations)
```

## License

Apache License 2.0
