# sglang-v100-plus

SM70 support for [SGLang](https://github.com/sgl-project/sglang), targeting V100
SXM2 GPUs arranged as two NVLink quads. This fork ports selected compatibility
work from [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) onto
recent mainline, using plugin hooks rather than carrying the original fork's
entire patch set. Source revisions and attribution are in
[v100_lite/provenance.json](v100_lite/provenance.json).

## Code organization

- `v100_lite/sglang_v100_lite/`: opt-in SM70 runtime adapters and CUDA/Triton
  kernels; no copied model implementations.
- `v100_lite/aot/`, `v100_lite/patches/`: selected native build and Marlin patches.
- `python/sglang/kernels/ops/`: shared operators where public kernel integration
  is appropriate, including GLM additions on its branch.
- `v100_lite/tests/gpu_checks.py` and `test/registered/`: numerical/operator and
  distributed regression tests.

Host services, deployment profiles, transfer scripts, investigation harnesses,
runbooks and experiment results belong outside this code repository.

## Architecture and branches

Each four-GPU NVLink quad has 128 GB of installed GPU memory. Communication
between quads uses PCIe/host links; weights share memory with caches and
workspaces. TP4 is the established Qwen profile. TP4×PP2 keeps frequent TP
collectives inside each quad for larger models; TP8 spans both quads.

`main` contains established Qwen/SM70 support. `glm-5.3-flash` adds experimental
GLM-5.3-Flash NVFP4 support, software FP8 indexing, sparse MLA, FP16 mHC and KDA
compatibility. GLM currently uses eager prefill and batch-one decode graphs;
MTP and multi-user throughput remain unvalidated.

Preliminary batch-one measurements on eight 32 GB V100s:

| Model/profile | Prompt tokens | PP tokens/s | TG tokens/s |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next, TP4 + MTP | 1,000 | 3,068 | 125.3 |
| GLM-5.3-Flash, TP8 | 2,048 | 670.7 | 33.09 |
| GLM-5.3-Flash, TP4×PP2 | 2,048 | 1,096.3 | 22.10 |

These are profile-specific development measurements. The matched GLM runs use
256-token prefill chunks, a 24/21 PP split and no speculation; they do not
predict concurrent-request throughput or imply cross-model comparisons.

## Build and use

Requires Python 3.12/uv, a CUDA 12.9 toolchain and SM70 GPUs. The project lock
pins the Volta-compatible CUDA-12 stack and selected `sglang-kernel` build.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
bash v100_lite/setup.sh
```

Use normal SGLang model/scheduling flags with the guarded plugin entry point:

```bash
SGLANG_PLUGINS=v100_lite SGLANG_V100_LITE=1 \
  uv run --no-project .venv/bin/python -m sglang_v100_lite --help
```

Refer to upstream SGLang for API usage. Configure models, GPU selection and
serving policy separately for your host.

## Upstream maintenance

The integrated upstream revision is recorded in `provenance.json`.
`v100_lite/core-patches.json` enumerates the remaining core exceptions; verify
`main` with `uv run --no-project .venv/bin/python v100_lite/check-core-diff.py`.
Update and validate `main` first, then integrate it into the GLM branch.
Prefer upstream implementations when equivalent fixes land. Keep fork-owned
hardware adaptations in the plugin/public operators and new changes covered
by focused numerical tests.

[Apache-2.0](LICENSE). Credit to SGLang, the original V100 fork,
[marlin_v100](https://github.com/zhinianqin/marlin_v100), and the kernel projects
identified in retained headers and provenance.
