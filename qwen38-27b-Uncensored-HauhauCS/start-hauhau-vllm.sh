#!/usr/bin/env bash
# Compatibility entry point: the former quantized launcher now starts the
# official, unquantized deployment so it cannot accidentally relaunch AWQ.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../qwen38-27b-Base/start-qwen38-bf16-vllm.sh" "$@"
