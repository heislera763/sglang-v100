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

The fast branch supports GLM's native MTP with EAGLE, top-k one and linear
chains of up to six verification slots. TP8 and TP4×PP2 batch-one offline runs
have exercised target/draft graphs, rejection, request reuse and sampled decoding
at temperature 1/top-p 0.95. The SM70 cache writer consumes upstream plans;
the PP relay now preserves the exact draft proposal probabilities by request,
and greedy prefills use the same proposal policy as decode. Concurrent requests
remain unvalidated. MTP remains an explicit CLI choice:
`--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6`.
Sampled MTP tests add `--speculative-use-rejection-sampling`; aggregate PP+MTP
requires `SGLANG_ENABLE_PP_SPEC=1` and `--disable-overlap-schedule`.

Preliminary batch-one measurements on eight 32 GB V100s:

| Model/profile | Prompt tokens | PP tokens/s | TG tokens/s |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next, TP4 + MTP, sampled thinking | 1,000 | 3,229 | 126.0 |
| GLM-5.3-Flash fast, TP8 | 2,048 | 675.0 | 41.52 |
| GLM-5.3-Flash fast, TP4×PP2 | 2,048 | 1,079.2 | 25.00 |

These are profile-specific development measurements. The matched GLM runs use
256-token prefill chunks, a 24/21 PP split and no speculation; they do not
predict concurrent-request throughput or imply cross-model comparisons.
Greedy measurements are diagnostics; sampled performance is the primary tuning
target. Use the lab's settings for the exact model and thinking mode, including
penalties, rather than assuming checkpoint generation defaults cover them.
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage)
specifies T=1/top-p=0.95/top-k=20 with no presence penalty for thinking, and
T=0.7/top-p=0.80/top-k=20/presence penalty=1.5 for non-thinking; both use
min-p=0 and repetition penalty=1.
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes) publishes
task-specific sampled recipes; our T=1/top-p=0.95 profile matches the checkpoint
defaults and several lab evaluations. Speculative verification currently
broadcasts history penalties across each block, so nonzero penalties do not yet
have ordinary decoding's per-token semantics.

Matched Qwen TP4 offline runs use the lab's thinking settings, EAGLE3/1/4 with
classical rejection sampling, no overlap/radix caching, 8K prefill chunks,
16K cache capacity and a 0.88 static fraction. Median TG rises from 70.1 to
126.0 tokens/s at 1K and from 70.0 to 124.8 at 8K. Each uses three repetitions
after warmup and 128 output tokens. The former 125.3 tokens/s result used greedy
decoding and a different serving profile. These measurements establish a sampled
speed gain, not exact output parity or broad quality equivalence.

Initial matched TP8 offline Engine runs on the fast branch improve TG from
32.58 to 41.70 tokens/s at 128 prompt tokens and 32.50 to 41.52 at 2,048
(about 28%). Each uses three measured repetitions, an excluded warmup and 64
generated tokens. At 2K, PP remains approximately 674–675 tokens/s.
Extending GEMV dispatch to the measured TP4 projection shapes raises TP4×PP2 TG
from 23.45 to 25.00 tokens/s at 2K (6.6%); prefill remains approximately 1,080
tokens/s. TP8 leads single-request TG, while TP4×PP2 leads longer-prompt prefill.

Matched five-step MTP comparisons at 2K (TG tokens/s):

| Layout | Sampling | Without MTP | With MTP |
| --- | --- | ---: | ---: |
| TP8 | Greedy | 41.4 | 48.8 |
| TP4×PP2 | Greedy | 25.0 | 37.0 |
| TP8 | Temperature 1, top-p 0.95 | 39.8 | 19.8 |
| TP4×PP2 | Temperature 1, top-p 0.95 | 24.8 | 13.6 |

Five-step MTP helps greedy TG; it loses with sampling on these inputs.
Greedy PP drops from 675.5 to 611.7 tokens/s at TP8 and from 1,088.1 to 958.6
at TP4×PP2. Prefill cost offsets some decode saving on short answers.
Runs use three repetitions after warmup, 64 forced output tokens, batch one,
8,192-token cache capacity and a 0.92 static fraction. All 11 greedy output
sequences per layout match their non-MTP controls; sampler checks compare with
a CPU oracle and verify actual proposal probabilities. This does not establish
broad model-quality or sampled-output parity. Extra draft weights and graphs
need separate memory budgeting; the PP+MTP last quad has limited headroom.

## Build and use

For development, `SGLANG_DEBUG_V100_STRICT_DISPATCH=1` makes guarded V100
dispatch reject fallback paths with the operation name, expected contract and
tensor shape/dtype/device/layout. Native Qwen HC/small-GEMM verification now
covers one through four rows. All development benchmark and kernel runs use
strict mode; fix uncovered paths rather than disabling it. Default
`0` preserves normal optional dispatch.
The mode is deliberately strict, including guarded prefill calls and disabled
fast-path options: a full-model run can stop at the first coverage gap during
startup or execution. It covers the plugin's HC/mHC, dense linear, router,
NVFP4 MoE, QSA attention/indexer and E5M2 cache dispatch boundaries, rather than
every Torch operation or upstream backend. Explicit SM70 FP16 cuBLAS prefill,
embedding/vocabulary projections, native CUDA HC combine and SM70 Marlin
routes are valid primary backends, not implicit fallbacks. Native variants
within a selected path still use their existing scheduling. It does not change sampling,
precision or tensor contents.

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

### Qwen3.8-Flash-Next: thinking + MTP

Quick reference for `glm-5.3-flash-fast`, from the repository root. GPUs 4–7
are this machine's full-Gen3 NVLink quad; choose a single NVLink quad on other
hosts. This is the serving equivalent of the measured batch-one, 16K offline
profile above, with the user's LAN endpoint and API key.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 OMP_NUM_THREADS=4 MAX_JOBS=4
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
export SGLANG_PLUGINS=v100_lite SGLANG_V100_LITE=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit"
export SGLANG_V100_NVFP4_MOE_BUILD_DIR="$PWD/.cache/nvfp4_moe"
export SGLANG_V100_DECODE_CUDA_BUILD_DIR="$PWD/.cache/longctx"
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC

uv run --no-project .venv/bin/python -m sglang_v100_lite \
  --model-path "$HOME/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4" \
  --served-model-name qwen3.8-flash-next --trust-remote-code \
  --host 192.168.4.78 --port 9000 --api-key test-only \
  --tensor-parallel-size 4 --disable-overlap-schedule \
  --dtype float16 --quantization modelopt_fp4 \
  --moe-runner-backend marlin --fp4-gemm-backend marlin \
  --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto \
  --mm-attention-backend sdpa --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --mamba-ssm-dtype float16 --ple-offload-embedding --disable-radix-cache \
  --mem-fraction-static 0.88 --context-length 16384 --max-total-tokens 16384 \
  --max-running-requests 1 --chunked-prefill-size 8192 \
  --cuda-graph-bs-decode 1 --disable-prefill-cuda-graph \
  --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-use-rejection-sampling \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

OpenAI-compatible base URL: `http://192.168.4.78:9000/v1`, key `test-only`.
Client-supplied settings override these defaults: keep thinking enabled,
temperature 1, top-p 0.95, top-k 20, min-p 0, presence/frequency penalties 0
and repetition penalty 1 in WebUI. The sampling top-k of 20 is separate from
the MTP branch width of one. Larger contexts and concurrency need separate
memory/performance checks; the former 262K profile is a different configuration.

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
