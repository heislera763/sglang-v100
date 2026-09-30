#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# This project is restricted to the test endpoint and the first NUMA-local quad.
[[ ${PORT:-9001} == 9001 ]] || { echo 'This launcher only serves test port 9001' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SGLANG_V100_LITE=1 SGLANG_PLUGINS=v100_lite
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 OMP_NUM_THREADS=4 MAX_JOBS=4
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
export XDG_CACHE_HOME="$PWD/.cache/runtime"
export SGLANG_CACHE_DIR="$PWD/.cache/sglang"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit" TORCH_EXTENSIONS_DIR="$PWD/.cache/torch"
export TILELANG_CACHE_DIR="$PWD/.cache/tilelang" TVM_FFI_CACHE_DIR="$PWD/.cache/tvm-ffi"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export SGLANG_V100_NVFP4_MOE_BUILD_DIR="$PWD/.cache/nvfp4_moe"
export SGLANG_V100_DECODE_CUDA_BUILD_DIR="$PWD/.cache/longctx"
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm"
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
if [[ -z ${LLAMA_API_KEY:-} ]]; then
    set -a; source "${ENV_FILE:-$HOME/.llama-server/.env}"; set +a
fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY is required}"
model=${MODEL_PATH:-$HOME/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4}
exec numactl --cpunodebind=0 --preferred=0 .venv/bin/python -m sglang_v100_lite \
  --model-path "$model" --served-model-name "${SGLANG_SERVED_MODEL_NAME:-qwen3.8-flash-next-radixark-nvfp4}" \
  --trust-remote-code --host 0.0.0.0 --tensor-parallel-size 4 \
  --dtype float16 --quantization modelopt_fp4 --moe-runner-backend marlin \
  --sampling-backend pytorch \
  --reasoning-parser auto --tool-call-parser auto --mm-attention-backend sdpa --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 --mamba-ssm-dtype float16 --ple-offload-embedding \
  --mem-fraction-static 0.88 --context-length 262144 --max-running-requests 1 \
  --chunked-prefill-size 8192 --cuda-graph-bs-decode 1 --disable-prefill-cuda-graph --disable-custom-all-reduce \
  --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE --speculative-draft-model-path "$model" \
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics "$@" --port 9001
