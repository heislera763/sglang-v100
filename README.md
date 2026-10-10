# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) adapted for **eight V100 SXM2 32 GB GPUs
in two NVLink quads**. We rebuilt useful SM70 support from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline,
effectively rebasing it into an independent fork. Qwen3.8-Flash-Next FP8 and
GLM-5.3-Flash NVFP4 work is consolidated on `main`.

`main` contains validated changes; `dev` is the integration branch. Keep upstream
updates and experiments on `dev`, then advance `main` after relevant checks.
Clone remotes: `origin` is this fork; `upstream` is `sgl-project/sglang`.

Integrated upstream: [3831e7e0](https://github.com/sgl-project/sglang/commit/3831e7e0918052be50ef38b3ff62154573e213d7), pinned 2026-10-10.
Hardware adapters live in [v100_plus/sglang_v100_plus](v100_plus/sglang_v100_plus),
reusable operators in [python/sglang/kernels/ops](python/sglang/kernels/ops), and
build/Marlin changes in [v100_plus/aot](v100_plus/aot) and [v100_plus/patches](v100_plus/patches).
[Provenance](v100_plus/provenance.json) and [28 recorded core changes](v100_plus/core-patches.json)
make upstream updates auditable. Services, benchmark clients, traces and project notes
stay outside Git. This assumes familiarity with SGLang.

## Checkpoints

Pinned benchmark sources; sizes are weight-file totals, **not runtime VRAM**.

| Checkpoint | Revision | Composition | Files |
| --- | --- | --- | ---: |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13) | `236dfdf28582` | Official 128×128 block FP8, BF16 exclusions; FP8 PLE/MTP experts | 172.78 GiB |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/tree/da920bb0b9f4a06727223a349e55468e38352348) | `da920bb0b9f4` | ModelOpt abs-max NVFP4; main shared experts remain BF16; FP8 KV metadata | 190.40 GiB |

NVIDIA GLM is the active GLM checkpoint. It uses an abs-max recipe, with BF16
shared experts and MTP exclusions; publisher alone does not prove better quality.
TP8 ordinary/MTP and PP2 ordinary/MTP are the reference layouts. The PP2 MTP
module-load OOM is handled by cold-load allocator reclamation, without changing
checkpoint, arithmetic or workload limits. NVIDIA Qwen NVFP4 remains deferred;
full GLM FP8 routed weights alone need roughly 283.5 GiB.

At checkpoint-native context, cache estimates below exclude weights, state,
execution scratch and required page/sentinel/speculative overhead:

| Model | Native tokens | TP8 main KV + index / GPU | PP2 largest main KV + index / GPU |
| --- | ---: | ---: | ---: |
| Qwen FP8 weights, FP16 KV | 262,144 | 3.19 GiB | 1.59 GiB |
| NVIDIA GLM | 1,048,576 | 11.35 GiB | 6.19 GiB |

Qwen MTP adds about0.27GiB on its owning stage. Native FP16 PP2/MTP3
capacity262106input+32output and request reuse pass. GLM's FP16 latent KV is replicated
across TP ranks. The current GLM PP2/MTP3 session limit is **262,144** with a
262,656-slot pool (~1.55GiB/GPU including target/draft KV and index storage).
A 262,106-input/32-output sampled capacity probe and subsequent request reuse passed;
this is functional capacity coverage, not a long-context quality claim. Larger
MTP windows and native1M remain unqualified. The TP8 launch below is a short-context
reference; the PP2 variation gives the working session configuration.

## SM70 execution

Volta supports FP16 Tensor Cores. FP4/FP8/BF16 checkpoint storage is decoded or cast
in software; it does not imply native arithmetic in those formats.

| Operation | Implementation / contract |
| --- | --- |
| NVFP4 GEMMs | [marlin_v100](https://github.com/zhinianqin/marlin_v100) at `6d72a499` plus local patches: W4A16, FP16 WMMA with FP32 accumulation; block/global scales remain distinct. Unused alternate-backend MoE scale buffers alias active scales, saving17.72GiB across GLM's8GPUs. |
| Block-FP8 GEMMs | Patched Marlin and small-row vector kernels: W8A16, complete 128×128 scale blocks. Qwen EP4/EP8 preserves whole experts. |
| Qwen QSA/GDN/HC | Direct FP16-cache sparse prefill/decode/verification, masked Tensor Core prefill, SM70 GDN and gated-residual kernels; HC eager prefill partitions within each quad, including TP8. |
| GLM sparse MLA / indexer | Explicit Volta Tensor Core FP16 sparse prefill for128+queries/H8/H16/latent512/no-tail; direct selected-KV gather, FP32 reductions and FP16 probabilities. Small-row Triton split attention keeps16-row KV tiles. Software FP8 learned indexer, request-owned pool4 compression and bounded FP32 scoring workspace. |
| GLM KDA / mHC | Range-safe KDA; FP32 mHC projection/RMS/Sinkhorn, private prefill buffer reused for square/mean (~128MiB lower peak at2048rows). BF16-to-FP16 conversion needs operation-specific range handling. |
| Storage | Unquantized FP16 attention KV for both models; Qwen FP16 QSA index, GLM's separate model-native FP8 indexer. KV-cache quantization is excluded from the project recipes. FP16 recurrent state is retained. |
| Cold module loading | Return idle Torch allocator blocks to CUDA when first-use Triton loads have little headroom; leave live tensors unchanged. Skip capture/custom arenas; steady kernel replay has no callback. |
| Vision / DFlash2 | SDPA vision. [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a) FP16/local W8A16 have prior experimental coverage; NVIDIA-target qualification is pending. |

For FP16 GLM, the plugin resolves checkpoint-inferred `auto` sparse KV to FP16,
including NVIDIA's FP8 KV metadata; weight quantization is unchanged. Explicit cache
choices remain explicit. FP32 storage does not remove FP16 internal arithmetic:
Qwen's current FP32-state path fails an existing operator bound; GLM's passes but
showed no speed gain. Strict dispatch rejects unsupported guarded paths; declared
Marlin/cuBLAS/Torch CUDA backends remain valid.

## Build and reference launches

Python 3.12, `uv`, CUDA 12.9; [setup](v100_plus/setup.sh) builds pinned SM70 AOT/Marlin
kernels. Use a fresh shell and the common block below for each recipe. Device order
must follow your topology. Local directory names are conventions; for Hub loading,
use the checkpoint ID and full `--revision` above. These are manual single-request
references, not installed services.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
git remote add upstream https://github.com/sgl-project/sglang.git
bash v100_plus/setup.sh
export CUDA_HOME=/usr/local/cuda-12.9 TRITON_PTXAS_PATH=/usr/local/cuda-12.9/bin/ptxas
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH" OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_PLUGINS=v100_plus SGLANG_V100_PLUS=1 SGLANG_DEBUG_V100_STRICT_DISPATCH=1
export SGLANG_V100_MARLIN_DIR="$PWD/artifacts/marlin-v100/vllm" SGLANG_SM70_DENSE_GEMV=1
export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
export NCCL_P2P_LEVEL=PHB NCCL_NVLS_ENABLE=0
common=(
  --host 0.0.0.0 --port 9000 --api-key test-only --random-seed 531 --dtype float16 --trust-remote-code
  --moe-runner-backend marlin --sampling-backend pytorch --reasoning-parser auto --tool-call-parser auto
  --disable-overlap-schedule --disable-radix-cache --mm-attention-backend sdpa
  --mamba-ssm-dtype float16 --max-running-requests 1 --disable-prefill-cuda-graph
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1
)
```

**Qwen FP8, TP4×PP2/EP4, ordinary** (`http://<host>:9000/v1`, key `test-only`):

Native QSA prefill/decode/verification read FP16 KV directly. For this pinned
checkpoint, `--dtype float16 --kv-cache-dtype auto` allocates FP16 KV; the
indexer and recurrent state also use FP16. Strict dispatch remains enabled.

```bash
export SGLANG_PP_LAYER_PARTITION=24,24 SGLANG_OPT_SM70_HC_PREFILL_SP=1
export SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" --served-model-name qwen3.8-flash-next \
  --quantization fp8 --json-model-override-args '{"language_model_only":true}' \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --context-length 262144 --max-total-tokens 262272 \
  --chunked-prefill-size 4352 --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype auto --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

**GLM NVIDIA NVFP4, TP8/EP1, ordinary** (fresh shell plus common block):

```bash
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_FUSE_MHC_POST_PRE=0
uv run --no-project .venv/bin/python -m sglang_v100_plus "${common[@]}" \
  --model-path "$HOME/models/sglang/nvidia-GLM-5.3-Flash-NVFP4" --served-model-name glm5.3-flash --language-only \
  --quantization modelopt_fp4 --fp4-gemm-backend marlin --tensor-parallel-size 8 --expert-parallel-size 1 \
  --context-length 12288 --max-total-tokens 12288 \
  --mem-fraction-static 0.92 --chunked-prefill-size 2048 --attention-backend dsa \
  --dsa-prefill-backend triton --dsa-decode-backend triton --linear-attn-prefill-backend triton --linear-attn-decode-backend triton \
  --page-size 256 --kv-cache-dtype auto --mamba-radix-cache-strategy no_buffer \
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":true}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":-1,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

| Variation | Change from reference |
| --- | --- |
| Qwen TP8 | TP8/EP8; omit PP and unset `SGLANG_PP_LAYER_PARTITION`/metadata flag. Keep HC/chunk4352. |
| GLM PP2 MTP | TP4/PP2/EP1; `SGLANG_PP_LAYER_PARTITION=24,21`; context **262144**, total tokens **262656**, fraction **.95**, chunk2048. Add the MTP flags below. Omit `--language-only` for images; vision weights reside on the first PP stage. |
| MTP, either model | Add `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1 --speculative-use-rejection-sampling`. PP2 also exports `SGLANG_ENABLE_PP_SPEC=1`. |

Three draft steps plus one target/bonus position; built-in MTP head, one linear branch,
classical sampled rejection. Lab sampling sources: [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage),
[ZAI](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes). Greedy decoding is not the benchmark baseline.

## Settings rationale

| Setting | Reason / established limit |
| --- | --- |
| Explicit backends/dtypes/strict | Supported SM70 implementations and precision policy; strict is required during development, not a speed switch. |
| TP/PP/EP and device order | Quad-local NVLink/P2P; cross-quad peer access is unavailable here. PP improves prefill but serializes single-request stages; TP8 MTP decodes faster on Qwen. |
| NCCL PHB/NVLS0, OMP4 | Retained host policy; removing PHB did not help the fresh Qwen screen. NVLS is unavailable on SM70. No universal-optimum claim. |
| Max1/serialized/radix off | Qualified graph/scratch/state ownership and uncached timing. Overlap/concurrency is separate work. |
| Eager prefill/full decode graphs | Qualified native paths; earlier Qwen prefill-graph screens lost. Chunk4352/2048 are measured per-model choices. |
| Context and pool capacity | Qwen targets native262144 plus128page/workspace slots. FP16 PP2/MTP3 allocation,262106-input/32-output generation and request reuse pass. Explicit uncached max1 reservations fail instead of shrinking. |
| Static fraction/state pool | Qwen uses the automatic fraction estimate. GLMTP8.92 is the short reference; GLMPP2MTP.95 qualifies the262K session. The fraction is a profiled allowance, not a physical VRAM allocation or speed setting. With radix off/max1, state slots derive from the request count; a state-memory ratio is redundant. Host PLE saves device memory. |
| GLM sparse prefill | The SM70 hook routes supported large FP16 rows to explicit Volta MMA under the `triton` DSA entry point; no extra flag or8K query-union restriction. Small-row decode retains its split policy. |
| GLM mHC flags0 | Select supported mHC paths; fused DSA top-k retains its default1. |
| CPU-only tests | `CUDA_VISIBLE_DEVICES=""`; numeric/disabled/UUID port allocation is supported. No999 workaround. |

| Optional flag | Recommendation |
| --- | --- |
| `SGLANG_OPT_SM70_HC_PREFILL_SP=1` | Qwen serialized text-only eager prefill, quad-local TP4/TP8/PP2. |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH=1` | Qwen PP2 fixed EAGLE1–3 commit copies; GLM fixed chains/full KDA snapshots supported; NVIDIA speed benefit unmeasured, leave off. |
| `SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1` | Qwen TP8 MTP3; leave off on PP2. Unsupported sampling features use their declared original path. |
| `SGLANG_OPT_SM70_NVFP4_MOE_GEMV=1` | GLM opt-in supports1–4 rows; prior publisher screen found no aggregate win, NVIDIA benefit unmeasured. Leave off. |
| `SGLANG_OPT_SM70_NVFP4_GEMV`, `SGLANG_OPT_SM70_MHC_PROJECTION` | Default1 GLM paths; set0 for corresponding unfused controls. |

Qwen TP8 MTP also uses the optional [small-allreduce tuner](v100_plus/nccl/quad_tuner.c):

```bash
mkdir -p .cache
cc -std=c11 -O2 -fPIC -shared -I v100_plus/nccl/include v100_plus/nccl/quad_tuner.c -o .cache/libnccl-v100-quad-tuner.so
export NCCL_TUNER_PLUGIN="$PWD/.cache/libnccl-v100-quad-tuner.so" SGLANG_OPT_SM70_SPEC_SAMPLE_GRAPH=1
```

It selects Tree/LL for small eight-rank transfers, leaving large prefill collectives
alone. Forcing Tree globally slows prefill. PP adaptive/DFlash remain guarded;
CP/DCP needs SM70 compression/KV ownership support, and overlap needs buffer leases.
These are implementation gaps, not unused free speed flags.

## Performance and checks

Three uncached coherent near8K requests/profile,512 sampled output tokens, one request,
eager prefill/full decode graphs. GLM PP2 MTP uses two fresh processes, images enabled
and262K context. Qwen MTP rows use fresh FP16-KV native-context screens;
GLM TP8/ordinary rows remain earlier one-process screens. Seed531 does not guarantee
sampled determinism. PP = input tokens/native prefill seconds; TG = output tokens
excluding the first/native decode seconds. Warmups and profiling are excluded.
These are preliminary throughput measurements. Qwen rows now use FP16 KV;
the earlier E5M2 results are historical controls. Earlier GLM rows:
[ce60d8c9](https://github.com/heislera763/sglang-v100-plus/commit/ce60d8c91c7a0b1c92b1c35059f0bfb7e492d8da)
compatible post-sync paths; NVIDIA PP2 MTP additionally uses the allocator fix
[a85513f5](https://github.com/heislera763/sglang-v100-plus/commit/a85513f5) and exact mHC buffer reuse
[9f057466](https://github.com/heislera763/sglang-v100-plus/commit/9f057466).
GLM PP2 MTP now uses [explicit Volta MMA prefill](python/sglang/kernels/ops/attention/dsa/sm70_sparse_prefill.py):
fresh A/B/B/A processes on 2026-10-10 gave 1,713 → 2,359 prefill tokens/s (+37.7%)
with the same 262K context, 2048-token chunks and FP16 KV. Decode kernels are unchanged; sampled TG varies
with outputs/acceptance. [Operator tests](test/registered/kernels/ops/attention/test_triton_sparse_mla_fp16.py)
cover reference math, ragged/empty selections, large physical slots and graph refresh.
MMA/softmax reduction order changes; this preserves support and precision boundaries,
not bit-exact scalar outputs. Images, 262106-input/32-output capacity and reuse pass;
full-context quality is separate. Earlier PP2 MTP at [aa02d9b3](https://github.com/heislera763/sglang-v100-plus/commit/aa02d9b3)
was 1714 PP / 42.5 TG. TP8 below remains an earlier screen without this prefill implementation.

| Model / layout | MTP steps | Prefill tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| Qwen FP8 TP8/EP8 | 3 | 4,831 | 79.4 |
| Qwen FP8 TP4×PP2/EP4 | 3 | 7,588 | 72.5 |
| NVIDIA GLM TP8/EP1 | 3 | 1,366 | 42.7 |
| NVIDIA GLM TP4×PP2/EP1 | Off | 1,882 | 28.2 |
| NVIDIA GLM TP4×PP2/EP1 | 3 | 2,359 | 43.6 |

MTP3 categories (PP/TG tokens/s; GLM PP2 averages two processes, others one):

| Model / category | TP8 PP | TP8 TG | PP2 PP | PP2 TG |
| --- | ---: | ---: | ---: | ---: |
| Qwen coding | 4,882 | 82.3 | 7,668 | 82.2 |
| Qwen book continuation | 4,831 | 82.6 | 7,607 | 66.3 |
| Qwen document briefing | 4,779 | 73.3 | 7,489 | 69.0 |
| NVIDIA GLM coding | 1,370 | 48.6 | 2,353 | 47.1 |
| NVIDIA GLM book continuation | 1,367 | 37.6 | 2,358 | 39.5 |
| NVIDIA GLM document briefing | 1,360 | 43.2 | 2,365 | 44.4 |

Inputs: [SPEED-Bench throughput_8k](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
first turns `91d6ca2afe114d3c99312e8758b6f964` (code),
`658dbd96a19e4b138e0aafe43eea1101` (book), `9a936eebf8794621a11f963a56c92120`
(adapted briefing). Qwen8517/8320/8346 input tokens, T1/P.95/K20/xhigh;
GLM7898/8116/8141, T1/P.95/K−1/max. Native metadata via
[return_meta_info](python/sglang/srt/entrypoints/openai/serving_chat.py).
Earlier six-request Qwen ordinary/MTP matrices at
[38f5a0f0](https://github.com/heislera763/sglang-v100-plus/commit/38f5a0f0806883ec88da23ca3c91a46bec7ce12b)
measured TP8 5002/4850PP and50.8/80.2TG; PP2 8078/7619PP and61.9/72.3TG.
Those used E5M2 KV and precede this FP16-cache screen. Current Qwen screens
use HC on, PP2 metadata on, TP8 metadata off, sampling graph/tuner off;
[FP16 QSA numerical/graph tests](test/registered/kernels/ops/attention/qsa/test_sm70_fp16_kv.py) cover the cache port.

Matched independent-process A/B/B/A against [f0245fb0](https://github.com/heislera763/sglang-v100-plus/commit/f0245fb0)
raises GLM PP2 MTP generation 32.2→42.5tokens/s on average: coding+30%,book+30%,
briefing+36%. Prefill is unchanged; speculative cycles are~23% shorter.

The small-row tile changes FP16 reduction rounding; independent FP64 probes found
max absolute error0.00044 versus the reference, old/new difference0.00049. It preserves
selected sparse support and is not bit-exact to the wider tile.

Independent operator bounds plus exact state/payload/RNG checks where specified;
native strict dispatch and EOS/logprob/penalty/cancel/reuse checks supplement them.
Exact full-model token parity across quantizers or TP/PP reductions is not established.
Nsight timelines are separate from timing runs; overlapping GPU durations and
collective waiting are not summed as recoverable wall time.

## Context and multimodal

Long-context and long-video coverage comes from earlier bounded studies; this pass
rechecked Qwen request-level video configuration. These are functional, not broad quality claims.

| Feature | Coverage / configuration |
| --- | --- |
| Qwen1M YaRN | Ordinary PP2, HC/metadata off, `--max-mamba-cache-size 1`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`; [official RoPE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices): factor4/original262144/theta10000000/partial.25/interleaved MRoPE[11,11,10]. Native977,312-input/512-output functional run; three separated planted codes retrieved. |
| Qwen images/video | `language_model_only:false`, HC/metadata off; SDPA vision. Historical image/H264 and request-level4/8/4-frame checks pass; current unquantized-KV qualification is pending. Request `video_config` overrides startup values without shared mutation. |
| NVIDIA GLM images | PP2/MTP3,262K context allocation,FP16 KV,SDPA vision; omit `--language-only`. Two swapped images, repeated-image reuse and subsequent8K/512-token text inference pass. Vision weights load only on the first PP stage; image embeddings reach last-stage MTP. Full-context image requests and NVIDIA video remain unqualified. |
| Qwen long video | Startup `--mm-process-config`:2fps/max64frames; optional size budget `{longest_edge:469762048,shortest_edge:4096}`. One-hour fixture sampled full duration; bounded functional coverage, not general video quality. |

Keep original RoPE for short-context benchmarks. Frame/pixel/encoder/KV budgets are
separate limits. Before/after upstream updates run
`uv run --no-project .venv/bin/python v100_plus/check-core-diff.py` and affected checks.

[Apache-2.0](LICENSE). Credit: SGLang, original V100 fork, Marlin/vLLM and model authors;
NCCL declarations retain their [license](v100_plus/nccl/LICENSE.txt).
