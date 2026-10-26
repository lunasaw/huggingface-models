#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if [[ "${1:-}" == --help ]]; then
  cat <<'EOF'
Usage: start-hauhau-vllm.sh [--dry-run] [extra vLLM arguments...]

Loads this project's .env (or HAUHAU_ENV_FILE), then starts the Hauhau compressed-tensors model.
See .env.example for GPU, port, model path and capacity settings.
--dry-run prints the command without loading weights or starting a server.
EOF
  exit 0
fi

ENV_FILE="${HAUHAU_ENV_FILE:-$SCRIPT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
cd -- "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Keep this host's validated runtime, while allowing other installations.
if [[ -z "${VLLM_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/vllm" ]]; then
    VLLM_BIN="$CONDA_PREFIX/bin/vllm"
  elif [[ -x /data/miniconda3/envs/qwen38-hauhau-vllm/bin/vllm ]]; then
    VLLM_BIN=/data/miniconda3/envs/qwen38-hauhau-vllm/bin/vllm
  else
    VLLM_BIN=vllm
  fi
fi
MODEL_DIR="${MODEL_DIR:-$ROOT/models/HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP}"
COMMAND=("$VLLM_BIN" serve "$MODEL_DIR" \
  --served-model-name "${MODEL_NAME:-qwen3.8-27b-hauhau-aggressive}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-18090}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --default-chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":"medium"}' \
  --override-generation-config '{"repetition_penalty":1.05}')

if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "${COMMAND[@]}" "$@"
  printf '\n'
  exit 0
fi
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  printf 'vLLM executable not found: %s (set VLLM_BIN)\n' "$VLLM_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  printf 'Model config not found: %s/config.json (download weights or set MODEL_DIR)\n' "$MODEL_DIR" >&2
  exit 1
fi
exec "${COMMAND[@]}" "$@"
