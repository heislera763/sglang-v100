# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) on Volta, optimized for two NVLink
quads of V100 SXM2 32 GB GPUs. We rebuilt useful adaptations from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline:
effectively rebasing the old hardware support into an independent fork, then
adding Qwen3.8-Flash-Next and GLM-5.3-Flash optimizations. **All profiles use `main`.**
Integrated upstream: `c892301ff76f`; [lineage](v100_plus/provenance.json),
[21 recorded core patches](v100_plus/core-patches.json).

## Architecture

- **TP4:** one 128 GB NVLink quad. **TP8:** parallel work across both quads, with
  collectives crossing host links. **TP4×PP2:** each stage owns one quad; better
  prefill here, but one request executes stages sequentially during decode.
  CUDA peer access is available inside each quad and unsupported across quads.
- **W4A16/W8A16:** NVFP4/FP8 weights stay compressed in GPU memory; SM70
  Marlin/WMMA and specialized vector kernels unpack/scale them for FP16
  activations. Volta has no native FP4/FP8 Tensor Core arithmetic. KV FP8 uses
  software conversion. Expert packing/scale contracts remain format-specific.
- **Qwen:** QSA/GDN, gated residuals and pinned-host PLE embeddings; vector
  experts for compatible 1–4-row target/draft operations, 32-row prefill routing,
  masked Tensor Core attention, and HC token partitions within each NVLink quad.
  HC quad groups preserve model TP8/EP8 attention/expert topology.
- **GLM:** sparse MLA/K-pool indexing, KDA and mHC. PP2 transports nested result
  tensors without losing proposal probabilities. Qwen PP2 captures recurrent/PLE
  commits for serialized 1–3-step chains, refreshing live request/acceptance buffers.
- **Maintenance:** adapters in `v100_plus/sglang_v100_plus/`, shared operators in
  `python/sglang/kernels/ops/`, builds/Marlin changes in `v100_plus/aot/` and
  `v100_plus/patches/`. Host services, transfer scripts, benchmark clients and
  experiment records stay outside Git. Prefer upstream fixes over duplicating
  model/scheduler implementations.

Numerical/operator references, graph/state/dispatch regressions and native model
runs are the validation boundary. They do not establish identical whole-model
outputs across quantization, GEMM shapes or TP/PP reduction orders. GLM's retained
GEMV/mHC paths are enabled by default; an unfused comparison sets
`SGLANG_OPT_SM70_NVFP4_GEMV=0 SGLANG_OPT_SM70_MHC_PROJECTION=0`.

## Build and common arguments

Python 3.12, `uv`, CUDA 12.9, SM70. Setup builds AOT kernels and patched Marlin;
the lock pins the CUDA 12 stack.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
bash v100_plus/setup.sh
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 OMP_NUM_THREADS=4 MAX_JOBS=4
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
export SGLANG_PLUGINS=v100_plus SGLANG_V100_PLUS=1
export SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit"
export SGLANG_V100_NVFP4_MOE_BUILD_DIR="$PWD/.cache/nvfp4_moe"
export SGLANG_V100_DECODE_CUDA_BUILD_DIR="$PWD/.cache/longctx"
export SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC
unset SGLANG_OPT_SM70_HC_PREFILL_SP SGLANG_ENABLE_METADATA_GLUE_GRAPH
unset SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH NCCL_TUNER_PLUGIN
sglang_args=(
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531
  --trust-remote-code --dtype float16 --moe-runner-backend marlin
  --disable-overlap-schedule --disable-radix-cache --sampling-backend pytorch
  --reasoning-parser auto --tool-call-parser auto --mm-attention-backend sdpa
  --mamba-ssm-dtype float16 --context-length 12288 --max-total-tokens 12288
  --max-running-requests 1 --chunked-prefill-size 2048
  --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1
  --mamba-full-memory-ratio 0.2
)
```

Manual launch references; adjust paths/order/endpoint for your host. API base:
`http://<server>:9000/v1`. Always keep strict dispatch on during development:
guarded missing coverage fails with tensor metadata. Declared Volta cuBLAS,
Torch CUDA and Marlin backends are valid implementations; the guard does not
instrument every upstream operation or establish which kernel is fastest.

## Qwen profiles

**NVFP4 TP4, thinking/two-step MTP:** select one NVLink quad.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC SGLANG_ENABLE_METADATA_GLUE_GRAPH
unset SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH NCCL_TUNER_PLUGIN
export SGLANG_OPT_SM70_HC_PREFILL_SP=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${sglang_args[@]}" \
  --model-path "$HOME/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4" \
  --served-model-name qwen3.8-flash-next --quantization modelopt_fp4 --fp4-gemm-backend marlin \
  --tensor-parallel-size 4 --mem-fraction-static 0.88 --chunked-prefill-size 4352 \
  --json-model-override-args '{"language_model_only":true}' --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --speculative-use-rejection-sampling \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

**Official FP8 TP4×PP2/EP4, ordinary:** all eight GPUs. EP keeps 640-wide experts
whole, preserving 128×128 checkpoint scale blocks; ordinary TP shards cross those
blocks. PLE embeddings stay in pinned host memory.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_PP_LAYER_PARTITION=24,24
export SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${sglang_args[@]}" \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" \
  --served-model-name qwen3.8-flash-next --quantization fp8 \
  --json-model-override-args '{"language_model_only":true}' \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --mem-fraction-static 0.88 --chunked-prefill-size 4352 --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

For **FP8 TP8**, use TP8/EP8, omit PP, unset its partition/metadata/PP-spec flags,
and retain HC/4352-token chunks. Consecutive groups of four launch ranks must
match the physical NVLink quads. For **FP8 MTP** on either eight-GPU layout, add
these arguments; PP2 also sets `SGLANG_ENABLE_PP_SPEC=1`:

```bash
--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 --speculative-use-rejection-sampling
```

Set `SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1` for TP8
and leave it off for PP2 at the selected three-step depth.
It preserves native tokens/RNG/q for serialized max-one complete linear chains;
penalties, greedy/min-p, grammar, logprobs, custom processors, deterministic seeded
sampling and unsupported schedules retain their existing sampling path. Linear
MTP penalties use every committed output and each row's causal draft prefix;
unsupported active-penalty branch trees fail explicitly.

The optional **TP8 small-allreduce tuner**, tested with NCCL 2.29.3, preserves
large-message and quad policies. Compile/load explicitly:

```bash
mkdir -p .cache
cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include \
  v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so
export NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so"
```

It chooses Tree/LL for 8-rank one-node AllReduce messages 8–256 KiB. Do not force
Tree globally: the large prefill transfers are slower with that policy here.

For **NVFP4 TP8**, use TP8/EP1 and the NVFP4 checkpoint/backend. The 80-wide expert
shard is padded exactly to 96 per gate/up half, including FC2/scales; no
requantization. For **NVFP4 PP2**, use TP4/PP2/EP1, the same 24,24 partition and
HC/metadata flags; two-step MTP also needs the PP-spec flag. These eight-GPU
NVFP4 profiles leave more cache memory but decode slower than the measured TP4.

Adaptive TP8 works with explicit `[1,2,3]` tiers, but its three-case screen
(79.6 generation tokens/s) did not beat fixed three-step (80.5). PP adaptive
needs an acknowledged relay tier before first-stage metadata/allocations;
the upstream guard remains. Qwen's JSON language-only
override skips vision loading; the CLI `--language-only` selects a different workflow.

## GLM profile

**NVFP4 TP8, ordinary:** current single-request default.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC
unset SGLANG_OPT_SM70_HC_PREFILL_SP SGLANG_ENABLE_METADATA_GLUE_GRAPH
unset SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH NCCL_TUNER_PLUGIN
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=0 SGLANG_DSA_FUSE_TOPK=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${sglang_args[@]}" \
  --model-path "$HOME/models/sglang/RadixArk-GLM-5.3-Flash-NVFP4" \
  --served-model-name glm5.3-flash --language-only --quantization modelopt_fp4 \
  --tensor-parallel-size 8 --mem-fraction-static 0.92 --fp4-gemm-backend marlin \
  --attention-backend dsa --dsa-prefill-backend triton --dsa-decode-backend triton \
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --page-size 256 --kv-cache-dtype auto --mamba-radix-cache-strategy no_buffer \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":true}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

PP2 uses TP4 and `SGLANG_PP_LAYER_PARTITION=24,21`. Three-step MTP adds EAGLE,
branch1/four verification positions/classical rejection; PP2 also needs
`SGLANG_ENABLE_PP_SPEC=1`. Keep serialized scheduling. TP8 MTP has no established
aggregate advantage. Experimental [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
with Triton draft attention works but measured ~19 generation tokens/s versus
~40 without speculation. Its adapter
keeps the overflowing BF16-trained output convolution/residual normalization in
FP32 until normalized FP16 projections. A local W8A16 drafter reduced memory,
without improving speed; GLM routed FP8 target experts alone need 283.5 GiB.

## Measured performance

Qwen FP8 on eight V100 SXM2 32 GB GPUs, six requests per mode across two fresh
processes. Same runtime
[`38f5a0f0`](https://github.com/heislera763/sglang-v100-plus/commit/38f5a0f0806883ec88da23ca3c91a46bec7ce12b),
2026-10-09; Torch 2.13.0+cu126/Transformers 5.19.0/NCCL 2.29.3. Near-8K coherent
inputs 8517/8320/8346, 512 sampled outputs, T1/top-p.95/top-k20/thinking-xhigh,
zero penalties, seed 531. Eager prefill/full decode graphs, HC, 4352-token chunks,
context/cache 12288, one request, no overlap/radix/profiling. Prefill divides input tokens
by native prefill time; generation excludes the first token. Warmups, cached
input and retractions are excluded. Rates aggregate total tokens/native seconds.

| FP8 layout | MTP steps | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| TP8/EP8 | Off | 5,002 | 50.8 |
| TP8/EP8 | 3 | 4,850 | 80.2 |
| TP4×PP2/EP4 | Off | 8,078 | 61.9 |
| TP4×PP2/EP4 | 3 | 7,619 | 72.3 |

MTP category rates (two requests each, same selected profiles):

| Category | TP8 prefill | TP8 generation | PP2 prefill | PP2 generation |
| --- | ---: | ---: | ---: | ---: |
| Coding | 4,868 | 87.8 | 7,716 | 78.5 |
| Book continuation | 4,870 | 77.6 | 7,620 | 73.4 |
| Document briefing | 4,812 | 76.2 | 7,522 | 66.2 |

TP8 MTP uses the bounded tuner plus captured sampler. PP2 uses the default sampler.
Three-step leads the one/two/three-step screen; it is workload-dependent.
The same-source three-step TP8 control without either optional optimization
measured 77.8 generation tokens/s (three requests); optimized confirmation is
80.2 (+3.0%; six requests). PP2 captured sampler 72.4 versus default 72.3
generation tokens/s;
no demonstrated gain at three steps, so leave its flag off for this profile.

Other preliminary profiles (different revisions/screens; not controlled A/Bs):

| Checkpoint/layout | MTP steps | Prefill tokens/s | Generation tokens/s | Evidence |
| --- | ---: | ---: | ---: | ---: |
| Qwen NVFP4/TP4 | 2 | 5,274 | 105.1 | [27e5127d](https://github.com/heislera763/sglang-v100-plus/commit/27e5127d) |
| Qwen NVFP4/TP8 | 2 | 4,433 | 83.3 | [38f5a0f0](https://github.com/heislera763/sglang-v100-plus/commit/38f5a0f0), 3-request screen |
| GLM NVFP4/TP8 | Off | 1,163 | 39.6 | [c7442393](https://github.com/heislera763/sglang-v100-plus/commit/c7442393) |
| GLM NVFP4/TP4×PP2 | 3 | 1,593 | 36.1 | [9212de6d](https://github.com/heislera763/sglang-v100-plus/commit/9212de6d) |

Inputs: [SPEED-Bench throughput_8k](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
first turns `91d6ca2afe114d3c99312e8758b6f964` (code),
`658dbd96a19e4b138e0aafe43eea1101` (book),
`9a936eebf8794621a11f963a56c92120` (adapted briefing). GLM used 7898/8116/8141
inputs, T1/top-p.95/unrestricted top-k/max effort, 2048-token chunks. Tooling/raw
responses stay outside Git; rates use native
[`return_meta_info`](python/sglang/srt/entrypoints/openai/serving_chat.py).

Independent numerical/operator controls retain existing tolerances; state,
proposal/payload and graph RNG comparisons are exact where specified. Native
EOS/logprob/penalty/cancel/reuse checks supplement operator tests. Seed 531 does
not guarantee deterministic sampled output. Profiler counters are separate from
these timings; kernel residence includes peer waiting and is not summed across GPUs.

Weights: [Qwen FP8 `236dfdf2`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13),
[Qwen NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(transferred snapshot without retained Hub revision),
[GLM NVFP4 `f46cf340`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/tree/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35).
Sampling: [Qwen thinking](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[GLM task settings](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes).

## Long context and multimodal

**Qwen FP8 1M YaRN:** ordinary PP2, HC/metadata disabled, fixed one-entry Mamba
cache, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`. Replace the override with
these [official settings](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices):

```bash
--context-length 1000000 --max-total-tokens 1000000 --max-mamba-cache-size 1 \
--json-model-override-args '{"language_model_only":true,"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
```

A native 977,312-input/512-output request completed at 2,948 prefill and 49.8
generation tokens/s, with unchanged checkpoint configuration. Independent token
counts agreed; three codes at widely separated positions were retrieved. This
is capacity/functional coverage, not a broad million-token quality evaluation.
Keep short-context profiles on their original RoPE settings.

**Images/video:** Qwen FP8 uses `{"language_model_only":false}`; GLM omits
`--language-only`. Use SDPA vision attention and disable Qwen HC/metadata.
Both models processed swapped-color images and a four-second H264 clip through
OpenAI `image_url`/`video_url`, including base64 data URLs. Qwen's optional video
size budget is roughly 224K tokens; set frame limits separately, for example:
`--mm-process-config '{"video":{"fps":2,"max_frames":64,"size":{"longest_edge":469762048,"shortest_edge":4096}}}'`.
A one-hour four-color fixture completed with 64 frames across the full duration,
3,624 input tokens (3,200 video), correct color order and plausible timestamps;
the 512-token thinking cap truncated its final summary. Set frame limits in startup
processor config: this Qwen processor currently ignores per-request `video_config`.
Frame caps/pixels, host decoding and
encoder/KV memory remain separate limits; this is functional coverage, not a
long-video quality evaluation. The lock includes CPU video decoding.

Before/after upstream updates, run
`uv run --no-project .venv/bin/python v100_plus/check-core-diff.py` plus affected
numerical/native tests. [Apache-2.0](LICENSE); credit to SGLang, the original
V100 fork, Marlin/vLLM and model authors. Third-party NCCL API declarations retain
their [license](v100_plus/nccl/LICENSE.txt).
