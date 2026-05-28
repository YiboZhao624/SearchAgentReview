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
conda create -n vllm11 python=3.11.14
conda activate vllm11
pip install vllm==0.11.0
```

```
conda create -n faiss-gpu python=3.12.12
conda activate faiss-gpu
conda install -c pytorch -c nvidia faiss-gpu=1.9.0
```

### Data Preparation

We have placed our training data in the `data/` directory:
- `data/train_{m}.jsonl` — training queries, where m denotes the number of queries.
- `data/val.jsonl` — validation queries, 900 entries.
- `data/test_all.jsonl` - test queries mixed from 5 datasets, 4125 entries in total.

Each line should be a JSON object with fields: `question`, `answer`, `supporting_facts` (optional).

If you want to sample the training/validation dataset by yourself, download these original datasets and use `src/data/multi_dataset_prep.py` to prepare multi-dataset mixtures from HotpotQA, 2WikiMultihopQA, MuSiQue, BamboogLe, and PopQA.

For the corpus Wiki-Fixed, we cannot provide it without violating the double-blindness (as the file is too large to upload, and the huggingface link is unable to be anonymize.). We provide the pipeline to generate it here:

1. You need these raw dataset files under ./data/:                                                                                                                                                                                  
  - hotpotqa/train.json — HotpotQA train split                                                                                                                                                                                   
  - hotpotqa/hotpot_dev_distractor_v1.json — HotpotQA dev split                                                                                                                                                                    
  - 2wiki/train.json — 2WikiMultiHopQA train split                                                                                                                                                                                 
  - 2wiki/dev.json — 2WikiMultiHopQA dev split                                                                                                                                                                                     
  - musique/musique_full_v1.0_train.jsonl — MuSiQue train split                                                                                                                                                                    
  - musique/musique_full_v1.0_dev.jsonl — MuSiQue dev split                                                                                                                                                                        
  - wiki/wiki-18.corpus.jsonl — Original wiki-18 corpus (~21M rows)  

2. Run the command:

```
python src/data/multi_dataset_prep.py \                                                                                                                                                                                          
      --data_root ./data \                                                                                                                                                                                                         
      --output_dir ./data/{Your_Target_Data_Path} \
      --hierarchical_tiers 3000,5000,15000 \                                                                                                                                                                              
      --wiki_corpus ./data/wiki/wiki-18.corpus.jsonl \                                                                                                                                                                             
      --wiki_out ./data//wiki-fixed.jsonl  
```

This step samples the training set from the HotpotQA, 2Wiki, and Musique, where 3000, 5000, 15000 means the entries per original dataset. Further, this step will complement the missing documents of the Wiki-18.corpus.jsonl, and provide the Wiki-fixed corpus.

3. Embed the documents:

```
python src/policy/tools/build_faiss_index.py \                                                                                                                                                                                   
      --corpus-path ./data/{Your_Target_Corpus_Path}/wiki-fixed.jsonl \                                                                                                                                                                                --backend-url {Embedding_Service_Your_Started} \
      --model {Model_Name_of_Your_Embedding_Service} \
      --index-out ./data/{Your_Target_Embedding_Path}/ \
      --batchsize {Batchsize_for_Each_Worker} \
      --max-workers {How_Many_Workers_You_Launched} \
      --vectors-per-shard {Suggest_Value_5500000} \
```

**Notice:** The embedding step takes a long time and computational resource to embed ~21.3M documents. We leverage Qwen3-8B-Embedding as our embedding model, and this process takes ~30 hours on 8*H200.

This will generate the  `wiki-fixed.manifest.json` along with `wiki-fixed.partxxxxx.faiss`, where the json file records the sharding information.

### Model

Download [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to `models/Qwen3-8B/`, as our base model.

Download [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) to `models/Qwen3-Embedding-8B/`, as our embedding model.

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
├── data/                     # train/val/test datasets
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
