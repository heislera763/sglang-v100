#!/usr/bin/env bash
# One-time, project-local build. The serving launcher performs no installations.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export SGLANG_BUILD_RUST_EXTS=none CUDA_VISIBLE_DEVICES=0,1,2,3
export XDG_CACHE_HOME="$PWD/.cache/runtime"
export SGLANG_CACHE_DIR="$PWD/.cache/sglang"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 NVCC_THREADS=1
export UV_LOCK_TIMEOUT=7200
uv venv --python 3.12 .venv --allow-existing
uv pip install --python .venv/bin/python setuptools setuptools-scm setuptools-rust wheel
uv sync --locked --no-install-package sglang-kernel
torch_cmake_dir=$(.venv/bin/python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')
# Reconfigure against this environment rather than a stale Torch probe path.
rm -f "$PWD/artifacts/aot-build/CMakeCache.txt"
uv sync --locked --no-build-isolation \
  --config-settings-package "sglang-kernel:build-dir=$PWD/artifacts/aot-build" \
  --config-settings-package "sglang-kernel:cmake.define.Torch_DIR=$torch_cmake_dir"
marlin="$PWD/artifacts/marlin-v100"
if [[ ! -d $marlin/.git ]]; then
  git clone https://github.com/zhinianqin/marlin_v100.git "$marlin"
  git -C "$marlin" checkout --detach 6d72a49939701d26b15b617a4cd2423174adb2d1
fi
[[ $(git -C "$marlin" rev-parse HEAD) == 6d72a49939701d26b15b617a4cd2423174adb2d1 ]]
for patch in "$PWD"/v100_lite/patches/marlin-v100-*.patch; do
  if ! git -C "$marlin" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$marlin" apply "$patch"
  fi
done
export CUTLASS_DIR=${CUTLASS_DIR:-$PWD/artifacts/cutlass}
if [[ ! -d $CUTLASS_DIR ]]; then
  git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi
[[ -f $CUTLASS_DIR/include/cute/tensor.hpp && -f $CUTLASS_DIR/include/cutlass/cutlass.h ]]
export CMAKE_ARGS="-DMARLIN_V100_NVFP4_ONLY=ON -DMARLIN_V100_FP8=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_70,code=sm_70"
build_python="$PWD/.venv/bin/python"
(cd "$marlin" && "$build_python" setup.py build_ext --inplace)
