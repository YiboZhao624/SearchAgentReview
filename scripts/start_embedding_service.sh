#!/usr/bin/env bash
set -euo pipefail

# One-click startup for multi-GPU vLLM embedding backends + nginx load balancer.
# Replace MODEL and HOST_IP as needed.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_TORCH_COMPILE=0
MODEL="${MODEL:-${ROOT_DIR}/models/Qwen3-Embedding-8B}"
MODEL_NAME="${MODEL_NAME:-Qwen3-Embedding-8B}"
HOST_IP="${HOST_IP:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8010}"
NGINX_PORT="${NGINX_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-float32}"

NUM_GPUS="${NUM_GPUS:-8}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGINX_CONF="${WORK_DIR}/configs/nginx_embedding.conf"
NGINX_BODY_DIR="${WORK_DIR}/logs/nginx_body"

mkdir -p "${WORK_DIR}/logs"
mkdir -p "${NGINX_BODY_DIR}"

echo "Generating nginx config at ${NGINX_CONF} ..."
cat > "${NGINX_CONF}" <<EOF
worker_processes  1;
events { worker_connections 1024; }
http {
  client_body_temp_path ${NGINX_BODY_DIR} 1 2;
  upstream vllm_embedding_backends {
$(for i in $(seq 0 $((NUM_GPUS - 1))); do
    port=$((BASE_PORT + i))
    echo "    server ${HOST_IP}:${port};"
  done)
  }

  server {
    listen ${NGINX_PORT};
    location /v1/embeddings {
      proxy_pass http://vllm_embedding_backends;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host \$host;
      proxy_read_timeout 300;
    }
  }
}
EOF

echo "Starting ${NUM_GPUS} vLLM embedding backends with Auto-Restart..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
  port=$((BASE_PORT + i))
  log="${WORK_DIR}/logs/vllm_${port}.log"
  
  # 使用 nohup 启动一个后台 bash 进程，该进程内部是一个无限循环
  nohup bash -c "
    # 确保子进程使用正确的 GPU
    export CUDA_VISIBLE_DEVICES=${i}
    
    while true; do
      echo \"[$(date)] Starting vLLM server on port ${port} (GPU ${i})...\"
      
      vllm serve \"${MODEL}\" \
        --served-model-name \"${MODEL_NAME}\" \
        --host 0.0.0.0 --port \"${port}\" \
        --task embedding \
        --dtype \"${DTYPE}\" \
        --max-model-len \"${MAX_MODEL_LEN}\" \
        --gpu-memory-utilization 0.95 \
        --hf-overrides '{\"is_matryoshka\": true}' \
        --dimensions 1024

      EXIT_CODE=\$?
      echo \"[$(date)] vLLM server on port ${port} stopped with exit code \$EXIT_CODE. Restarting in 5 seconds...\"
      
      # 暂停 5 秒，避免瞬间频繁重启打满 CPU
      sleep 5
    done
  " >> "${log}" 2>&1 &

  echo "  - GPU ${i}: http://${HOST_IP}:${port}/v1/embeddings (Auto-restart enabled, log: ${log})"
done

echo "Starting nginx on port ${NGINX_PORT} ..."
nginx -c "${NGINX_CONF}"

echo "Done."
echo "Embedding LB endpoint: http://${HOST_IP}:${NGINX_PORT}/v1/embeddings"

