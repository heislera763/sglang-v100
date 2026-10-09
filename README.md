# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) for Volta, developed on **eight V100
SXM2 32 GB GPUs arranged as two NVLink quads**. We rebuilt useful SM70 adaptations
from [haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
effectively rebasing the hardware support into an independent fork. Qwen3.8-Flash-Next
and GLM-5.3-Flash work is consolidated on `main`.

The integrated upstream base is `c892301ff76f`, with targeted DFlash block-verification
backport [e902207a](https://github.com/sgl-project/sglang/commit/e902207a).
Adapters live in [v100_plus/sglang_v100_plus](v100_plus/sglang_v100_plus), reusable
operators in [python/sglang/kernels/ops](python/sglang/kernels/ops), and build/Marlin
changes in [v100_plus/aot](v100_plus/aot) and [v100_plus/patches](v100_plus/patches).
[Provenance](v100_plus/provenance.json) and [30 recorded core changes](v100_plus/core-patches.json)
keep upstream updates reviewable. Machine services, benchmark clients, traces and
project notes stay outside the fork.

## Checkpoints and provenance

Actual benchmark checkpoints; sizes are safetensors file totals, **not VRAM**.
Packing, replication, host PLE, vision, state and KV caches change the footprint.

| Checkpoint | Saved revision | Quantized scope / recipe | Files |
| --- | --- | --- | ---: |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Transferred snapshot; Hub revision not retained | ModelOpt 0.46.0; group-16 NVFP4 routed experts; BF16 attention/shared experts/MTP; FP8 PLE tables | 125.91 GiB |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13) | `236dfdf28582` | Official block-FP8, 128×128 scales; BF16 exclusions, FP8 PLE/MTP experts | 172.78 GiB |
| [RadixArk/GLM-5.3-Flash-NVFP4](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/tree/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35) | `f46cf340d35a2` | ModelOpt 0.46.0; abs-max group-16 NVFP4 routed/shared experts and dense MLPs; BF16 attention/router/vision/MTP | 188.98 GiB |
| [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) | `bf582e4eacc1` | Trained eight-position drafter; BF16 checkpoint, FP16 execution with range fixes | 2.18 GiB |
| DFlash2 W8A16 | Local conversion of preceding revision | Abs-max/448 block-FP8 linear weights, FP32 exported scales; convolution/norm execute in FP32 | 1.20 GiB |

RadixArk already uses **NVIDIA Model Optimizer**; the publisher alone does not identify
the calibration recipe or exclusions. Alternative checkpoints worth qualifying:

| Candidate, inspected 2026-10-09 | Weight files | Difference requiring qualification |
| --- | ---: | --- |
| [NVIDIA Qwen NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47) | 123.57 GiB | MSE-calibrated NVFP4; FP8 MTP experts/PLE. Mixed FP4/FP8 loading and block-preserving MTP EP need qualification. |
| [NVIDIA GLM NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) | 190.40 GiB | Shared experts remain BF16; different exclusions and FP8 KV metadata. Packing/scales/cache policy need SM70 checks. |

NVIDIA variants are untested here. Size suggests feasibility, not confirmed runtime
fit or improved quality/speed. GLM FP8 cannot be fully resident: routed experts alone
need ~283.5 GiB. Links pin available revisions; Qwen NVFP4's transferred revision is unknown.

## How SM70 is supported

Volta has FP16 Tensor Cores, **no native BF16/FP8/FP4 Tensor Core arithmetic**.
Checkpoint format and execution arithmetic are separate choices:

| Operation | Volta implementation | Contract / limitation |
| --- | --- | --- |
| NVFP4 linear/expert GEMMs | [marlin_v100](https://github.com/zhinianqin/marlin_v100) at `6d72a499`, plus local patches; unpack/scales → FP16 WMMA | W4A16 execution, not the checkpoint's advertised W4A4 activation arithmetic; distinct block/global-scale packing |
| Block-FP8 GEMMs | Patched Marlin plus small-row vector experts; software E4M3 decoding with FP16 activations | W8A16; complete 128×128 checkpoint blocks required. Qwen uses EP4/EP8 to keep experts whole. |
| Small projections / routed experts | CUDA/Triton GEMV, FP32 accumulation and defined FP16 output/activation boundaries | Selected by rows, geometry, packing and ownership; GLM optional routed-expert path supports 1–4 rows |
| Qwen QSA/GDN/HC | Masked FP16 Tensor Core sparse attention, SM70 GDN kernels and native gated-residual kernels | HC prefill partitions tokens within each quad; TP8 uses two independent quad groups without changing attention/expert TP |
| GLM sparse MLA / indexer | FP16 Triton attention; software FP8 indexer conversion/scoring; request-owned pool4 cache | Eager prefill/full decode graphs; optional two-query sharing preserves selected support but changes reduction order |
| GLM KDA / mHC | Range-safe recurrent KDA; FP32 mHC projection/RMS/Sinkhorn, fused projection for 1–8 rows | KDA is a remaining prefill cost; blindly casting BF16 chunk intermediates to FP16 overflows |
| KV / recurrent storage | Qwen software E5M2 KV conversion, FP16 QSA index cache; GLM FP16 sparse KV with separate FP8 indexer | KV/indexer/state dtype settings are independent; smaller storage does not imply native FP8 compute |
| Vision / DFlash2 | SDPA vision attention; DFlash2 FP32 output convolution/residual normalization before FP16 projections; explicit Torch CUDA softmax on SM70 | Image/video supported; DFlash2 remains experimental and slower than ordinary/MTP in current sampled tests |

Strict development dispatch rejects guarded unsupported shapes. Declared
Marlin/cuBLAS/Torch CUDA backends are valid; the guard does not profile or
instrument every upstream operator.

## Build and launch

Python 3.12, `uv`, CUDA 12.9. [Setup](v100_plus/setup.sh) builds SM70 AOT kernels and
pinned/patched Marlin; the lock pins the CUDA 12 stack. Examples assume a fresh shell
and local checkpoints at the paths below. Set device order from your host topology.
For Hub loading, substitute the repo ID and full `--revision` from the pinned snapshot link.
The local `org-model` directory names are storage conventions. API: `http://<server>:9000/v1`;
`test-only` is the example API key. These are manual launch references.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
bash v100_plus/setup.sh
export CUDA_HOME=/usr/local/cuda-12.9 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH" OMP_NUM_THREADS=4
export SGLANG_PLUGINS=v100_plus SGLANG_V100_PLUS=1 SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm" SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
common=(
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531 --dtype float16 --moe-runner-backend marlin
  --disable-overlap-schedule --disable-radix-cache --sampling-backend pytorch
  --reasoning-parser auto --tool-call-parser auto --mm-attention-backend sdpa
  --mamba-ssm-dtype float16 --mamba-full-memory-ratio 0.2
  --context-length 12288 --max-total-tokens 12288 --max-running-requests 1
  --disable-prefill-cuda-graph --cuda-graph-backend-decode full --cuda-graph-bs-decode 1
)
```

**Qwen official FP8, TP4×PP2/EP4, ordinary:**

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7 SGLANG_PP_LAYER_PARTITION=24,24
export SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" --served-model-name qwen3.8-flash-next --quantization fp8 \
  --json-model-override-args '{"language_model_only":true}' --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --mem-fraction-static 0.88 --chunked-prefill-size 4352 --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

**GLM RadixArk NVFP4, TP8, ordinary:** use the common arguments from a fresh shell.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_FUSE_MHC_POST_PRE=0
export SGLANG_OPT_SM70_SPARSE_PREFILL_UNION=1 SGLANG_OPT_SM70_NVFP4_MOE_GEMV=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/RadixArk-GLM-5.3-Flash-NVFP4" --served-model-name glm5.3-flash --language-only \
  --quantization modelopt_fp4 --fp4-gemm-backend marlin --tensor-parallel-size 8 --expert-parallel-size 1 \
  --mem-fraction-static 0.92 --chunked-prefill-size 2048 --attention-backend dsa \
  --dsa-prefill-backend triton --dsa-decode-backend triton --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --page-size 256 --kv-cache-dtype auto --mamba-radix-cache-strategy no_buffer \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":true}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

| Profile change | Values relative to the recipes | Reason |
| --- | --- | --- |
| Qwen FP8 TP8 | TP8/EP8; omit PP and unset its partition/metadata flags; retain HC/chunk4352 | Stronger sampled MTP decode; cross-quad allreduces remain |
| Qwen RadixArk NVFP4 TP4 | NVFP4 checkpoint, `--quantization modelopt_fp4 --fp4-gemm-backend marlin`; TP4/EP1, no PP; select one quad | Fits a quad with host PLE; fastest measured Qwen single-request profile; select MTP2 |
| Qwen NVFP4 PP2 / TP8 | PP2: TP4/EP1, partition24,24; TP8: TP8/EP1, no PP | TP8 width80 expert shards are zero-padded to96 without requantization; preliminary profiles |
| GLM PP2 | TP4/PP2/EP1; `SGLANG_PP_LAYER_PARTITION=24,21`; MTP table also enables `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | Quad-local TP; faster prefill, sequential single-request decode |
| MTP, either model | `--speculative-algorithm EAGLE --speculative-num-steps N --speculative-eagle-topk 1 --speculative-num-draft-tokens N+1 --speculative-use-rejection-sampling` | N draft steps plus one target/bonus position; built-in MTP head, linear branch, classical sampled rejection. PP2 also sets `SGLANG_ENABLE_PP_SPEC=1`. |

Use N=2 for Qwen NVFP4 TP4, N=3 for Qwen FP8 and GLM; substitute integer counts.
Sampling recommendations come from
[Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage) and
[ZAI](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes); greedy decoding is not the
performance baseline. GLM's template supports `reasoning_effort`, not Qwen's
`enable_thinking=False` switch.

### Why these settings exist

These are qualified **single-request benchmark profiles**, not an exhaustive optimum
or a deployment preset. Some choices deliberately disable useful multi-user features.

| Setting / group | Value | Justification and status |
| --- | --- | --- |
| Plugin / Marlin path / CUDA 12 ptxas | Exports above | Required SM70 compatibility/build selection; build/cache-directory housekeeping is separate from tuning |
| FP16 model/conv/state | Explicit FP16 | Tested compute/storage policy; BF16 range fixes are operation-specific. FP32 recurrent state is a separate precision/memory choice. |
| Attention / linear / MoE backends | Qwen Triton + `tilelang_v100`; GLM DSA/Triton; Marlin | Explicit supported implementations; do not substitute Blackwell FlashInfer/TRT-LLM recipes |
| GLM TileLang mHC/fusion flags | All0; `SGLANG_DSA_FUSE_TOPK` stays its default1 | Select supported FP32/FP16 mHC implementation instead of newer-hardware paths |
| GPU order / PP partition / EP | Profile table | Quad-local P2P/NVLink; cross-quad peer access unsupported here. FP8 EP preserves scale blocks. |
| NCCL / CPU threads | PHB, NVLS0, OMP4 | Retained host policy; no evidence all overrides beat resolved defaults on every workload |
| Serialized/max1; radix off | Above | Current graph/state ownership and uncached comparison; overlap and cache reuse require separate qualification |
| Eager prefill / full decode graph | Above | Qualified native paths; prefill-graph screens did not improve Qwen, GLM prefill capture remains unsupported |
| Chunk size | Qwen4352 / GLM2048 | Measured choices; GLM2048 selected jointly across TP8/PP2, not each layout's isolated maximum |
| Memory fraction / pool ratio / context | .88 or .92 / .2 /12288 | Capacity for near8K tests; these are workload limits, not demonstrated universal speed optimizations |
| Qwen PLE offload / cache policy | Host PLE / `extra_buffer` | Keep large n-gram tables in pinned host RAM; preserve Qwen recurrent/PLE state contracts |
| GLM cache policy | `auto` KV / `no_buffer` state | FP16 sparse KV; preserve qualified KDA snapshot/commit path |
| Strict dispatch | `SGLANG_DEBUG_V100_STRICT_DISPATCH=1` | Required development check for missing coverage; not a speed toggle |
| CPU-only tests | `CUDA_VISIBLE_DEVICES=999` | Hides CUDA devices for tests that must not allocate GPU memory; never a serving setting |

Optional optimization flags (default0 unless specified):

| Flag | Qualified use / status |
| --- | --- |
| `SGLANG_OPT_SM70_HC_PREFILL_SP=1` | Qwen short-context text-only serialized eager prefill; quad-local partitions across TP4/TP8/PP2 |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | First PP stage's exact recurrent/PLE commit copies; max-one serialized EAGLE chains of1–3 steps. GLM requires full-snapshot Triton KDA, not fused-accept. |
| `SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1` | Qwen FP8 TP8 MTP3; no demonstrated win on PP2. Unsupported sampling/features retain their declared original path. |
| `SGLANG_OPT_SM70_SPARSE_PREFILL_UNION=1` | GLM <=8K: two-query shared keys, exact support/ownership masks; later16-head/full-history rows retain ordinary attention |
| `SGLANG_OPT_SM70_NVFP4_MOE_GEMV=1` | GLM 288-expert gated widths256/512,1–4 rows; independent/native controls pass. Enabled in the measured matrix; small aggregate gain remains optional. |
| `SGLANG_OPT_SM70_NVFP4_GEMV`, `SGLANG_OPT_SM70_MHC_PROJECTION` | Default1; retained GLM fast paths change accumulation order. Set0 for the corresponding unfused comparison. |

Qwen TP8 MTP's optional [small-allreduce tuner](v100_plus/nccl/quad_tuner.c) is built
with `mkdir -p .cache` then `cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so`
and selected by `NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so"`.
It chooses Tree/LL for 8–256 KiB one-node eight-rank allreduces; forcing Tree
globally slows large prefill transfers. GLM showed no useful combined win.
PP adaptive speculation and PP DFlash remain guarded unsupported combinations.

## Performance and validation

Eight V100 SXM2 32 GB; Torch 2.13.0+cu126, Transformers 5.19.0, NCCL 2.29.3.
Six uncached coherent near-8K requests/profile across two fresh processes; 512 sampled
output tokens, one request, eager prefill/full decode graphs, zero penalties.
Qwen: 8517/8320/8346 input tokens, T1/P.95/K20/xhigh; GLM: 7898/8116/8141, T1/P.95/K−1/max.
Same chunk/backend within each model's matrix. Seed531 is a comparison setting,
not a guarantee of sampled determinism. PP = input tokens / native prefill seconds;
TG = generated tokens excluding the first / native decode seconds. Warmups, cached
input, retractions and profiling are excluded; rates aggregate tokens/seconds.

| Model / layout | MTP steps | Prefill tokens/s | Generation tokens/s | Runtime |
| --- | ---: | ---: | ---: | --- |
| Qwen official FP8 TP8/EP8 | Off | 5,002 | 50.8 | [38f5a0f0](https://github.com/heislera763/sglang-v100-plus/commit/38f5a0f0806883ec88da23ca3c91a46bec7ce12b) |
| Qwen official FP8 TP8/EP8 | 3 | 4,850 | 80.2 | Same |
| Qwen official FP8 TP4×PP2/EP4 | Off | 8,078 | 61.9 | Same |
| Qwen official FP8 TP4×PP2/EP4 | 3 | 7,619 | 72.3 | Same |
| GLM NVFP4 TP8/EP1 | Off | 1,453 | 39.7 | [5fae9db0](https://github.com/heislera763/sglang-v100-plus/commit/5fae9db01a89abe2362b10f79f030e3a3bfa3ed9) |
| GLM NVFP4 TP8/EP1 | 3 | 1,352 | 41.1 | Same |
| GLM NVFP4 TP4×PP2/EP1 | Off | 1,856 | 24.8 | Same |
| GLM NVFP4 TP4×PP2/EP1 | 3 | 1,704 | 37.1 | Same |

Qwen TP8 MTP includes tuner/captured sampling; a three-request same-source control
measured 77.8 TG versus 80.2 (+3.0%, six-request confirmation). PP2 captured sampling
72.4 versus72.3: leave it off. GLM's matrix enables sparse union and optional expert
GEMV; PP2 MTP also enables the commit graph. Three-request same-source controls with
expert GEMV/commit capture off measured 43.0 TG (TP8) and 36.4 (PP2), versus 41.1/37.1
in the six-request matrix. Sampling/acceptance varies substantially by content/run;
these extras have no established repeatable aggregate advantage and remain optional.
Three-step leads the shorter depth screen, but TP8 MTP's gain over ordinary is modest.
Qwen NVFP4 TP4/MTP2 preliminarily measured 5,274 PP/105.1 TG at
[27e5127d](https://github.com/heislera763/sglang-v100-plus/commit/27e5127d), a separate
checkpoint/revision study. DFlash2 FP16/W8A16 token/block screens measured ~21 TG;
the smaller drafter did not improve speed.

MTP3 category rates (two requests/category; PP/TG tokens/s):

| Model / category | TP8 PP | TP8 TG | PP2 PP | PP2 TG |
| --- | ---: | ---: | ---: | ---: |
| Qwen coding | 4,868 | 87.8 | 7,716 | 78.5 |
| Qwen book continuation | 4,870 | 77.6 | 7,620 | 73.4 |
| Qwen document briefing | 4,812 | 76.2 | 7,522 | 66.2 |
| GLM coding | 1,354 | 48.1 | 1,701 | 42.2 |
| GLM book continuation | 1,353 | 33.3 | 1,702 | 33.2 |
| GLM document briefing | 1,348 | 45.0 | 1,707 | 37.0 |

Inputs: [SPEED-Bench throughput_8k](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
first turns `91d6ca2afe114d3c99312e8758b6f964` (code),
`658dbd96a19e4b138e0aafe43eea1101` (book), `9a936eebf8794621a11f963a56c92120`
(adapted briefing); native timing via [return_meta_info](python/sglang/srt/entrypoints/openai/serving_chat.py).
Independent operator references keep existing numerical bounds; state, graph RNG,
proposal/payload identity checks are exact where specified. EOS/logprob/penalty/cancel/reuse
and native dispatch checks supplement them. Exact whole-model output parity across
quantization/GEMM/TP/PP reduction orders is not established. Nsight timelines are
separate measurements; collective residence includes peer waiting and is not summed
across GPUs as available wall-time savings.

## Context and multimodal coverage

| Feature | Configuration / established coverage |
| --- | --- |
| Qwen FP8 1M YaRN | Ordinary PP2; HC/metadata off, `--max-mamba-cache-size 1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; [official RoPE settings](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices): factor4, original262144, theta10000000, partial.25, interleaved MRoPE[11,11,10]. Native977,312-input/512-output request:2,948PP/49.8TG; three widely separated planted codes retrieved. Capacity/functional coverage only. |
| Images / short video | Qwen `language_model_only:false` with HC/metadata off; GLM omits `--language-only`; SDPA vision. Both processed swapped-color images and a four-second H264 clip through `image_url`/`video_url`, including base64. |
| Qwen long video | Startup `--mm-process-config` frame limits:2fps/max64frames; optional size budget `{longest_edge:469762048,shortest_edge:4096}`. One-hour fixture sampled the full duration:3,624 input tokens/3,200video, correct color order/plausible timestamps;512 thinking cap truncated summary. Current processor ignores per-request `video_config`. Functional coverage only. |

Keep original RoPE for short-context results. Frame/pixel budgets, host decoding,
encoder memory and KV capacity are separate limits. The lock includes CPU video decoding.
Before/after upstream updates, run `uv run --no-project .venv/bin/python v100_plus/check-core-diff.py`
and affected numerical/native checks. Broader defaults/CP/DCP/DP/EP/overlap/cache
configuration review remains open; current profiles are measured points.

[Apache-2.0](LICENSE). Credit to SGLang, the original V100 fork, Marlin/vLLM and model
authors; NCCL API declarations retain their [license](v100_plus/nccl/LICENSE.txt).
