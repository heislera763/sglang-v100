# sglang-v100-plus

SM70 support for [SGLang](https://github.com/sgl-project/sglang), tuned for V100
SXM2 GPUs arranged as two NVLink quads. We rebuilt the compatibility work from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
selectively porting its kernels into plugin hooks and shared operators. This
is effectively a rebase of the useful hardware adaptations into an independent
fork, with a smaller surface for future upstream updates. The integrated
upstream revision is `affa261e3d28`; exact origins and revisions are recorded in
[v100_lite/provenance.json](v100_lite/provenance.json).

## Architecture and scope

Each NVLink quad has 128 GB of installed GPU memory. TP4 keeps Qwen's frequent
collectives inside one quad. TP8 spans both quads over PCIe/host links.
TP4×PP2 confines TP collectives to each quad and passes activations between
stages; its potential prefill advantage does not imply lower single-request
decode latency. Models share memory with KV/recurrent caches, draft weights
and workspaces, so 256 GB installed is not 256 GB available for weights.

NVFP4 weights are unpacked into FP16 arithmetic through the SM70 Marlin/WMMA
adapter and selected GEMV kernels. FP8 cache/indexer formats use software
conversion on Volta. Qwen support includes QSA, GDN, gated residual connections
and pinned-host n-gram embeddings. GLM adds sparse MLA/K-pool indexing, KDA and
mHC, with FP16 activations/residuals and FP32 indexer head weights/mHC parameters.
Current measured profiles use eager prefill and batch-one decode graphs.

| Branch | Purpose |
| --- | --- |
| `main` | Established Qwen and SM70 compatibility |
| `glm-5.3-flash` | GLM compatibility and conservative operator paths |
| `glm-5.3-flash-fast` | GLM speed work; floating-point reduction differences are accepted |

On the fast branch, batch-one NVFP4 GEMV and fused mHC projection/RMS are on
by default for supported shapes. The checkpoint and model architecture stay
fixed, but reduction order can change routing, logits and generated tokens.
Operator references, finite-output checks and native model runs are development
gates; coherent prose is not a correctness test, and these gates do not prove
whole-model sampled-distribution or exact reference-output equivalence.

For an unfused comparison, set
`SGLANG_OPT_SM70_NVFP4_GEMV=0 SGLANG_OPT_SM70_MHC_PROJECTION=0`.
`SGLANG_OPT_SM70_MHC_POINTWISE=1` requires projection fusion to be off.
These alternatives have their own numerical/performance behavior.

## Preliminary performance

Maintainer measurements on eight 32 GB V100 SXM2s, **2026-10-07**, after correcting
the host's fan controller. All six cells use the same three coherent near-8K
inputs, concurrency one, a 512-token output cap and three measured passes.
These are this fork's preliminary results, not model-lab performance claims.

| Model / layout | MTP | Prefill tokens/s | Generation tokens/s |
| --- | --- | ---: | ---: |
| Qwen3.8-Flash-Next, TP4 | Off | 3,712 | 70.1 |
| Qwen3.8-Flash-Next, TP4 | 2 steps | 3,387 | 99.1 |
| GLM-5.3-Flash, TP8 | Off | 1,015 | 39.7 |
| GLM-5.3-Flash, TP8 | 3 steps | 925 | 40.1 |
| GLM-5.3-Flash, TP4×PP2 | Off | 1,719 | 24.8 |
| GLM-5.3-Flash, TP4×PP2 | 3 steps | 1,579 | 30.3 |

MTP improves generation by **41% for Qwen** and **22% for GLM PP2**.
GLM TP8's roughly 1% difference does not demonstrate a meaningful generation gain.
MTP reduces measured prefill throughput by roughly 8–9%. GLM TP4×PP2 prefills
about 70% faster than TP8, but TP8 generates faster for a single request.
The MTP depths are the best tested aggregate choices on these inputs;
small depth differences and this three-prompt sample do not establish a
universal optimum or multi-user throughput.

All profiles use context/cache 12,288, prefill chunks 2,048, eager prefill,
full batch-one decode graphs, strict SM70 dispatch, and no overlap/radix caching.
The shared chunk fits PP2's draft-stage workspace. Qwen uses the full-Gen3
NVLink quad at TP4, static memory fraction 0.88 and FP8 E5M2 KV; GLM uses both
quads, static fraction 0.92 and ordinary FP16 MLA KV. PP2 retains the 24/21-layer
split, with the draft on the last stage. Activations/state are FP16 and NVFP4
weights use Marlin. GPU order is `4,5,6,7` for Qwen and
`1,0,2,3,4,5,6,7` for both GLM layouts.

Sampling is T=1/top-p=0.95, min-p=0, additive penalties=0, repetition penalty=1
and request/server seed 531. Qwen uses thinking/xhigh and top-k=20; GLM uses
max effort/clear thinking and unrestricted top-k. MTP uses branch width one
and classical rejection sampling. A seed does not force identical sampled
sequences across speculative configurations.

Rates are total tokens divided by total native execution time across nine
requests per cell: prefill uses prompt tokens and the server's prefill interval;
generation uses completion tokens minus the first token and the decode interval.
Each prompt gets an excluded 32-token warmup. All measured outputs reached 512
tokens, with zero cached tokens and zero retractions. Native timing, sampling
parameters, prompt counts and speculation histograms were independently checked.
Timing/count validation does not establish numerical or distributional parity.
The SGLang HTTP client extracts
[`return_meta_info`](python/sglang/srt/entrypoints/openai/serving_chat.py);
llama.cpp's unmodified client expects its own `timings` fields.
Raw responses, commands and benchmark tooling stay outside this source repo.

<details>
<summary>Reproducibility identifiers</summary>

Runtime source:
[`cf7f7e9f`](https://github.com/heislera763/sglang-v100-plus/commit/cf7f7e9fa42baf4fe683ab0444183b7ca8e2e409),
PyTorch `2.13.0+cu126`, CUDA runtime 12.6.
Inputs come from [NVIDIA SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
configuration `throughput_8k`, revision
`454f88454792dfa3ccfd7ef15fff248efde44cd1`.
Use the first `turns` entry from these `question_id` values:

| Input | Dataset ID | Qwen tokens | GLM tokens |
| --- | --- | ---: | ---: |
| Code completion | `91d6ca2afe114d3c99312e8758b6f964` | 8,517 | 7,898 |
| Book continuation | `658dbd96a19e4b138e0aafe43eea1101` | 8,320 | 8,116 |
| Reference briefing | `9a936eebf8794621a11f963a56c92120` | 8,346 | 8,141 |

For the briefing input, preserve the complete reference excerpt and replace its
original instruction with:

```text
Read the reference excerpt below and write a concise briefing of roughly 300 words. Group the named characters and literary works into themes, explain key relationships, and support the briefing with concrete examples from the excerpt. Distinguish the reference author's claims from your interpretation.

REFERENCE EXCERPT:
```

Token counts include each model's chat template. No synthetic token IDs,
repeated padding or masked-placeholder rows are used.
Checkpoints:
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
and [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/tree/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35).
The transferred Qwen checkpoint has no retained Hub revision, so its exact
weight provenance is less complete than GLM's.

</details>

## Sampling and speculative decoding

Use the exact model's lab settings. The
[Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage)
recommends the thinking parameters above; non-thinking uses T=0.7/top-p=0.80,
top-k=20 and presence penalty=1.5. The
[GLM model card](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes)
publishes task-specific recipes, and its
[checkpoint defaults](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/blob/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35/generation_config.json)
use T=1/top-p=0.95. Our max-effort chat profile is one such sampled profile,
not a universal recommendation for every task.

One MTP head is called repeatedly: `D` draft steps means `D + 1` verification
positions, including the bonus-token opportunity. Branch width one is
`--speculative-eagle-topk 1`; it is separate from Qwen's sampling top-k of 20.
More acceptance or shorter drafts need not mean more tokens/s. Current
speculative penalties are broadcast across a verification block in
[`eagle_utils.py`](python/sglang/srt/speculative/eagle_utils.py); nonzero history
penalties do not yet have ordinary decoding's per-token semantics.

TP8 and TP4×PP2 GLM have realistic sampled performance evidence above.
TP4×PP2 MTP also has bounded earlier offline coverage of target/draft graphs,
request reuse, rejection and temperature-1 sampling. Its proposal-probability relay and greedy
prefill policy have focused regressions; eleven greedy sequences per layout
matched their non-MTP controls in that earlier check. These checks preceded
strict dispatch and do not establish concurrency, larger-context capacity or
TP8/TP4 output parity.

The near-8K matrix above includes fresh ordinary-decoding controls and
measured depth selection for each layout. Retain the 24/21 split; these
single-request results leave concurrency and broader workload coverage open.
The last quad also hosts the draft model and has limited memory headroom.
The experimental aggregate PP path requires `SGLANG_ENABLE_PP_SPEC=1`,
EAGLE/top-k one and `--disable-overlap-schedule`; adaptive depth and attention
DP are rejected by the current
[PP compatibility checks](python/sglang/srt/arg_groups/validation_hook.py).

## Build and development

Requires Python 3.12/uv, CUDA 12.9 and SM70 GPUs. The lock pins the CUDA-12
stack and selected `sglang-kernel` build. Setup builds the project-local SM70
Marlin extension and applies its checked-in patches.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
git switch glm-5.3-flash-fast
bash v100_lite/setup.sh
```

All development runs use `SGLANG_DEBUG_V100_STRICT_DISPATCH=1`. Guarded HC/mHC,
linear, router, NVFP4 MoE, QSA attention/indexer and E5M2 cache boundaries fail
with tensor metadata when coverage is missing. Native Qwen HC/small GEMM covers
one through four rows, allowing one/two/three draft steps. Declared SM70
cuBLAS projections, native CUDA HC combine and Volta Marlin are valid primary
backends. Strict mode does not promise the fastest variant at every shape or
instrument every upstream operation. Default 0 preserves optional dispatch;
development fixes gaps instead of turning the guard off.

The examples below target `glm-5.3-flash-fast`. Common environment, from the
repository root:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 OMP_NUM_THREADS=4 MAX_JOBS=4
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
export SGLANG_PLUGINS=v100_lite SGLANG_V100_LITE=1
export SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit"
export SGLANG_V100_NVFP4_MOE_BUILD_DIR="$PWD/.cache/nvfp4_moe"
export SGLANG_V100_DECODE_CUDA_BUILD_DIR="$PWD/.cache/longctx"
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC
```

### Qwen thinking + MTP, TP4

Choose one NVLink quad; `4,5,6,7` is the measured machine's quad. This example
uses the two-step coding/writing profile; three steps/four verification
positions is the measured QA/low-entropy alternative in this small suite.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run --no-project .venv/bin/python -m sglang_v100_lite \
  --model-path "$HOME/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4" \
  --served-model-name qwen3.8-flash-next --trust-remote-code \
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531 \
  --tensor-parallel-size 4 --disable-overlap-schedule \
  --dtype float16 --quantization modelopt_fp4 \
  --moe-runner-backend marlin --fp4-gemm-backend marlin \
  --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto \
  --mm-attention-backend sdpa --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --mamba-ssm-dtype float16 --ple-offload-embedding --disable-radix-cache \
  --mem-fraction-static 0.88 --context-length 12288 --max-total-tokens 12288 \
  --max-running-requests 1 --chunked-prefill-size 2048 \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 --disable-prefill-cuda-graph \
  --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 \
  --speculative-use-rejection-sampling \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

### GLM thinking + MTP, TP8

Map both NVLink quads to consecutive TP groups before evaluating PP. The order
below matches the benchmark host. Use a local copy of the linked NVFP4
checkpoint and the common environment above.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=0 SGLANG_DSA_FUSE_TOPK=1
export SGLANG_OPT_SM70_NVFP4_GEMV=1 SGLANG_OPT_SM70_MHC_PROJECTION=1
uv run --no-project .venv/bin/python -m sglang_v100_lite \
  --model-path "$HOME/models/sglang/RadixArk-GLM-5.3-Flash-NVFP4" \
  --served-model-name glm5.3-flash --trust-remote-code --language-only \
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531 \
  --tensor-parallel-size 8 --disable-overlap-schedule \
  --dtype float16 --quantization modelopt_fp4 \
  --moe-runner-backend marlin --fp4-gemm-backend marlin \
  --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto \
  --mm-attention-backend sdpa --attention-backend dsa \
  --dsa-prefill-backend triton --dsa-decode-backend triton \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype auto --mamba-ssm-dtype float16 \
  --mem-fraction-static 0.92 --context-length 12288 --max-total-tokens 12288 \
  --max-running-requests 1 --chunked-prefill-size 2048 --disable-radix-cache \
  --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --mamba-radix-cache-strategy no_buffer --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-use-rejection-sampling \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":true}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

These are manual launch references, not installed services. Bind address,
port, API key and model paths are host choices; the OpenAI base URL is
`http://<server>:9000/v1`. Clients can override sampling/template defaults.
For the tested experimental PP profile, change TP8 to
`--tensor-parallel-size 4 --pipeline-parallel-size 2`, set
`SGLANG_PP_LAYER_PARTITION=24,21 SGLANG_ENABLE_PP_SPEC=1`, and retain no-overlap
scheduling. Use 3 draft steps and 4 verification positions for the measured PP
choice. To run ordinary decoding, omit the speculative flags and leave
`SGLANG_ENABLE_PP_SPEC` unset. TP8 ordinary decoding is a reasonable default
here, given the negligible MTP generation gain and reduced prefill throughput.

## Code organization and upstream maintenance

- `v100_lite/sglang_v100_lite/`: opt-in runtime adapters and CUDA/Triton kernels.
- `v100_lite/aot/` and `v100_lite/patches/`: native build and Marlin patches.
- `python/sglang/kernels/ops/`: shared/public operators, including GLM additions.
- `v100_lite/tests/gpu_checks.py`, `test/registered/` and `test/manual/`: focused
  numerical, dispatch and distributed regressions.

Services, transfer scripts, host runbooks, benchmark harnesses and raw experiment
artifacts stay outside this source repository. The `v100_lite` package name is
retained for now; its eventual rename is separate work.

[v100_lite/core-patches.json](v100_lite/core-patches.json) enumerates the core
exceptions. Verify maintained `main` with
`uv run --no-project .venv/bin/python v100_lite/check-core-diff.py`.
Update and validate `main` against upstream first, then integrate it into the
GLM branches; integrate conservative GLM changes into the fast branch as well.
Prefer upstream implementations when equivalent fixes land. Keep hardware
adaptations in the plugin/public operators, with focused tests and provenance,
so an upstream update can review a small set of deliberate exceptions.

[Apache-2.0](LICENSE). Credit to SGLang, the original V100 fork,
[marlin_v100](https://github.com/zhinianqin/marlin_v100), and the kernel projects
identified in retained headers and provenance.
