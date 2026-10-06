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

`glm-5.3-flash-fast` builds on the GLM branch and prioritizes speed over exact
reference token/logit parity. Batch-one SM70 NVFP4 GEMV and fused mHC
projection/RMS are enabled by default for their supported shapes. Checkpoint
weights, quantization and model architecture are unchanged; accumulation order
changes can alter generated tokens. Development gates remain numerical
agreement with independent operator references, finite outputs, bounded
generation smoke checks and full-model performance measurements. These checks
do not establish broad quality equivalence.

Set `SGLANG_OPT_SM70_NVFP4_GEMV=0 SGLANG_OPT_SM70_MHC_PROJECTION=0` to compare
against the unfused paths. The alternative `SGLANG_OPT_SM70_MHC_POINTWISE=1`
requires `SGLANG_OPT_SM70_MHC_PROJECTION=0`; the two mHC implementations are
mutually exclusive. Existing shape/dtype/scheduling guards retain the fallback
paths for unsupported cases.

Preliminary batch-one measurements on eight 32 GB V100s:

| Model/profile | Prompt tokens | PP tokens/s | TG tokens/s |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next, TP4 + MTP | 1,000 | 3,068 | 125.3 |
| GLM-5.3-Flash, TP8 | 2,048 | 670.7 | 33.09 |
| GLM-5.3-Flash, TP4×PP2 | 2,048 | 1,096.3 | 22.10 |

These are profile-specific development measurements. The matched GLM runs use
256-token prefill chunks, a 24/21 PP split and no speculation; they do not
predict concurrent-request throughput or imply cross-model comparisons.

Initial matched TP8 offline Engine runs on the fast branch improve TG from
32.58 to 41.70 tokens/s at 128 prompt tokens and 32.50 to 41.52 at 2,048
(about 28%). Each uses three measured repetitions, an excluded warmup and 64
generated tokens. At 2K, PP remains approximately 674–675 tokens/s.

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
Keep speed experiments on `glm-5.3-flash-fast` and integrate validated changes
from `glm-5.3-flash` into it.
Prefer upstream implementations when equivalent fixes land. Keep fork-owned
hardware adaptations in the plugin/public operators and new changes covered
by focused numerical tests.

[Apache-2.0](LICENSE). Credit to SGLang, the original V100 fork,
[marlin_v100](https://github.com/zhinianqin/marlin_v100), and the kernel projects
identified in retained headers and provenance.
