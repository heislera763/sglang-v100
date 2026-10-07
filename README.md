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

Maintainer measurements on eight 32 GB V100 SXM2s, **2026-10-06**. Numbers below
are generation tokens/s on real chat prompts, using the pinned
[llama.cpp SPEED-Bench HTTP client](https://github.com/ggml-org/llama.cpp/blob/abeada335e2e78bd3fe63febafab7e900ce75810/tools/server/bench/speed-bench/README.md)
and [NVIDIA SPEED-Bench dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench).
These links identify the workload and methodology; the GPU results are this
fork's measurements, not NVIDIA or the model labs' published performance.

### Qwen3.8-Flash-Next NVFP4, TP4

| Category | 1 draft step | 2 draft steps | 3 draft steps |
| --- | ---: | ---: | ---: |
| Coding | 87.6 | 109.1 | 98.6 |
| QA | 92.8 | 107.9 | 93.2 |
| Writing | 91.1 | 106.4 | 102.8 |
| Low-entropy code | 88.3 | 101.4 | 107.3 |

Two steps leads these qualitative samples; three leads low-entropy code.
One-step acceptance is 72–82%, but produces fewer tokens per verification.
Its early/late control cycles stay at 19.51/19.58 ms. The old two-step collapse
(7.9–15.8 tokens/s) was missing three-row HC/small-GEMM coverage: verification
fell onto a slow generic route. Native one-through-four-row coverage removes
that gap. A prior no-MTP run was about 71 tokens/s; it predates strict dispatch
and is not a same-revision control for this table.

### GLM-5.3-Flash NVFP4, TP8

| Category | No MTP | 2 steps | 3 steps | 4 steps | 5 steps, repeated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coding | 39.8 | 40.1 | 32.6 | 45.7 | 38.0 |
| QA | 39.8 | 36.6 | 25.9 | 37.9 | 30.5 |
| Writing | 39.9 | 36.5 | 25.0 | 43.9 | 28.2 |
| Low-entropy code | 39.1 | 38.6 | 22.1 | 48.8 | 26.6 |

Four is the best tested draft depth for this suite, although QA still trails
ordinary decoding. Its early/late cycles stay at 80.93/80.77 ms. Three grows
from 92.78 to 158.73 ms, and a fresh five-step control from 101.42 to 186.61 ms;
the five-step slowdown was observed in two independent launches. Three accepts
more drafts than four in every category yet runs slower. These runs preceded
correction of a disabled fan controller on the benchmark host; the slower
depths need matched repeats before attributing cycle growth to software.
The conditional depth descent stopped when three regressed; GLM one-step has
not been tested. These TP8 results do not establish the best PP2 draft depth.

### GLM-5.3-Flash NVFP4, TP4×PP2

Sampled SPEED-Bench runs, 24/21 layers, strict dispatch enabled. Four-step
columns are matched repeats after restoring the host's fan controller;
the no-MTP baseline predates that correction:

| Category | No MTP, historical | 4 steps, original quads | 4 steps, reversed quads | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| Coding | 24.9 | 36.8 | 37.0 | 69% |
| QA | 24.9 | 32.4 | 32.5 | 58% |
| Writing | 24.9 | 32.5 | 32.8 | 58% |
| Low-entropy code | 24.8 | 37.2 | 37.7 | 70% |

Original order is `1,0,2,3,4,5,6,7`; reversed quads are
`4,5,6,7,1,0,2,3`. Each TP group retains its NVLink quad. Quad-order differences
are small in this single pass; all twelve messages and speculation counts
match between orders. Both runs have stable excluded early/late
verification cycles: 103.54/102.37 ms original and 101.53/101.82 ms reversed.
Post-suite warmed prefill of the same coherent 1,002-token prompt is
1.19/1.20 s respectively; the earlier no-MTP focused control was 1.03 s.

Earlier doubled prefill and 104–108 ms local attention stalls were measured
with the fan controller disabled. After its correction, unchanged runtime,
mapping and workload no longer reproduce those stalls. Treat the older
slowdowns as cooling-confounded, rather than inherent PP/MTP overhead. Four
post-suite profiled requests per order show attention below 9.3 ms and no
local kernel over 100 ms on any rank in the captured prefill windows.
These PP rates still trail the historical TP8 four-step rates; a matched
post-correction TP8 control is pending. Four is the only draft depth tested
in this PP suite; concurrency and larger-context capacity remain unmeasured.

### Measurement definition and provenance

All SPEED-Bench tables use concurrency 1, a 512-token output cap and twelve excluded
32-token warmups per profile. `qualitative` has two prompts each for coding,
QA and writing; `throughput_1k/low_entropy` has six repository-code completions.
Masked placeholders are excluded. Thinking consumes the output budget, and
sampled outputs/lengths can differ. Category rates are arithmetic means of
per-request native decode throughput, not a full-dataset leaderboard score.
Small single passes do not establish small speed gains or multi-user capacity.

The client was adapted to non-streaming SGLang
[`return_meta_info`](python/sglang/srt/entrypoints/openai/serving_chat.py):
`choices[0].meta_info.decode_throughput`, checked against
`(completion_tokens - 1) / (e2e_latency - first_token_latency)`.
Accepted/proposed draft counts exclude bonus tokens and are checked against
verification histograms. The upstream client expects llama.cpp-specific
`timings`; pointing it at SGLang without this extraction does not reproduce
these metrics. Timing runs capture no logits/probability tensors. Raw responses,
commands and validation remain in the maintainer's separate benchmark workspace;
this repository contains the summary and reproducibility identifiers.

| Profile | GPUs / layout | Context/cache | Static fraction | Prefill chunk | KV format |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen | Full-Gen3 NVLink quad, TP4 | 16,384 | 0.88 | 8,192 | FP8 E5M2 |
| GLM | Both quads, TP8 | 8,192 | 0.92 | 256 | Ordinary FP16 MLA |
| GLM PP | One TP4 group per quad, PP2 | 8,192 | 0.92 | 256 | Ordinary FP16 MLA |

All use FP16 activations/state, NVFP4 Marlin and no overlap/radix caching.
Speculative profiles use classical rejection sampling with branch width one.
Qwen uses thinking/xhigh,
T=1/top-p=0.95/top-k=20. GLM uses max effort/clear thinking,
T=1/top-p=0.95/unrestricted top-k. Both use min-p=0, additive penalties=0,
repetition penalty=1 and request/server seed 531. A request seed does not force
identical sampled sequences across speculative configurations.

Qwen two/three-step source:
[`ad9e6f8d`](https://github.com/heislera763/sglang-v100-plus/commit/ad9e6f8d326824a549317c196d5a8707e7b306ee).
Qwen one-step and GLM TP8 rows:
[`3eb3b844`](https://github.com/heislera763/sglang-v100-plus/commit/3eb3b84455e094f66c6d5738b8817e321e8c466f).
The intervening patch declares GLM library projections and changes diagnostic
text/tests/README; Qwen numerical kernels are unchanged. The PP baseline and
first MTP run use
[`41807a4a`](https://github.com/heislera763/sglang-v100-plus/commit/41807a4a0a457c0d51d44ef0f4ab230a702e32b0),
and the corrected-cooling PP runs use
[`3b41bc32`](https://github.com/heislera763/sglang-v100-plus/commit/3b41bc3260e1a5a0f70435d5b689cbaebc4cec51).
Both revisions change only the README from `3eb3b844`. Early/late controls accompany
Qwen one-step, GLM two/three/four/repeated-five-step and both PP runs; they are
excluded from category averages.

<details>
<summary>Pinned workload selection</summary>

Client revision: `abeada335e2e78bd3fe63febafab7e900ce75810`.
Dataset revision: `454f88454792dfa3ccfd7ef15fff248efde44cd1`.
Select these `question_id` values and send their first `turns` entry as a user
message; these rows each contain one real turn.

```json
{
  "qualitative": {
    "coding": ["0daf539b787c4dccbb547330a8b4c3d7", "135c7fe91faa48fd83ca5eac94c09f00"],
    "qa": ["a3ac2c931db84417a211b7d8756b5ff1", "2caa2289a2574756bb51967df812c420"],
    "writing": ["bec8b7f8659648118561e17008cc59cf", "5beba2e064e244a49a78aa83a0ae23e3"]
  },
  "throughput_1k/low_entropy": [
    "2afab7aa17f54849bf304a71cd57a6e6", "b67d4a675bc64caabe13fbf924998256",
    "ebe95e276bb242e3a85a447331e66ed4", "cfce22f2e1bd4d65b643630c5808ae4f",
    "67851b996d744da99bec992aafda6d5f", "6f219a885333415793f1e804cb273112"
  ]
}
```

Checkpoints:
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
and [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/tree/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35).
The transferred Qwen checkpoint has no retained Hub revision; it is identified
by repository name, so exact weight provenance is less complete than GLM's.

</details>

### Historical prefill diagnostics

Older offline Engine tests (2026-10-05, source `6b4bfc6b`) used 2,048 random
input token IDs, 64 forced output tokens, greedy decoding and three measured
repeats after warmup:

| GLM layout | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: |
| TP8 | 675.5 | 41.4 |
| TP4×PP2, 24/21 layers | 1,088.1 | 25.0 |

Both used cache/context 8,192, static fraction 0.92 and prefill chunks 256.
These are synthetic execution diagnostics, useful for the measured prefill
layout comparison. Their MTP acceptance is not representative of coherent
text, and they are not the SPEED-Bench results above. Earlier Qwen figures near
126 tokens/s used one padded natural web-service task, 128 forced output tokens
and a different repetition protocol; they are a separate workload too.

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

The next performance task is a matched TP8 control after the cooling correction,
then a PP draft-depth sweep using stable full-model timing.
Retain the 24/21 split. Draft depth must be measured for PP, rather than
inherited from TP8.
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
uses the two-step qualitative profile; use three steps/four verification
positions for the measured low-entropy code alternative.

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
  --mem-fraction-static 0.88 --context-length 16384 --max-total-tokens 16384 \
  --max-running-requests 1 --chunked-prefill-size 8192 \
  --cuda-graph-bs-decode 1 --disable-prefill-cuda-graph \
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
  --mem-fraction-static 0.92 --context-length 8192 --max-total-tokens 8192 \
  --max-running-requests 1 --chunked-prefill-size 256 --disable-radix-cache \
  --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --mamba-radix-cache-strategy no_buffer --mamba-full-memory-ratio 0.2 \
  --speculative-algorithm EAGLE --speculative-num-steps 4 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 \
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
scheduling. Four steps works in this bounded PP suite; it is not a measured
PP optimum.

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
