# sglang-v100-plus

[SGLang](https://github.com/sgl-project/sglang) on Volta, optimized for two NVLink
quads of V100 SXM2 32 GB GPUs. We rebuilt the useful adaptations from
[haohervchb/sglang-V100](https://github.com/haohervchb/sglang-V100) on mainline:
effectively rebasing the hardware support into an independent fork, then adding
Qwen3.8-Flash-Next and GLM-5.3-Flash optimizations. **Use `main` for all profiles.**
The integrated upstream revision is `c892301ff76f`; see
[provenance](v100_plus/provenance.json) and the
[20 deliberate core patches](v100_plus/core-patches.json).

## Approach

- **TP4:** collectives stay inside one 128 GB NVLink quad. **TP8:** collectives
  cross the host links. **TP4×PP2:** each stage uses its own quad, with activations
  crossing between stages. This fits the topology well for prefill; one request
  still executes pipeline stages sequentially during ordinary decode.
- **NVFP4/W4A16 and FP8/W8A16:** keep compressed weights resident, unpack/scale
  them inside SM70 Marlin/WMMA or specialized GEMV kernels, and multiply FP16
  activations. Volta has no native FP8/FP4 Tensor Core arithmetic. KV FP8 formats
  use software conversion.
- **Architecture coverage:** Qwen QSA/GDN, gated residuals and pinned-host PLE
  embeddings; GLM sparse MLA/K-pool indexing, KDA and mHC. Qwen FP8 PP2 adds
  vector experts for ordinary and MTP decode, block-scale reuse, partitioned HC
  prefill, masked Tensor Core sparse attention and GPU metadata graphs. Sampled
  MTP keeps exact draft probabilities locally, packs nested PP result tensors
  for both models and retains Tensor Core Qwen target/draft prompt attention.
  Qwen's two-step PP2 profile also captures accepted recurrent/PLE state copies.
- **Upstream maintenance:** adapters live in `v100_plus/sglang_v100_plus/`;
  shared operators live in `python/sglang/kernels/ops/`; native builds and Marlin
  patches live in `v100_plus/aot/` and `v100_plus/patches/`. Keep host services,
  transfer scripts, benchmark tools and experiment records outside this repo.

Reduction order and quantization can change logits, expert choices and tokens.
Numerical references, graph/dispatch regressions and native model runs are our
checks; they do not establish exact whole-model reference equivalence. GLM's
retained GEMV/mHC optimizations are enabled by default. An unfused comparison
uses `SGLANG_OPT_SM70_NVFP4_GEMV=0 SGLANG_OPT_SM70_MHC_PROJECTION=0`.

## Build

Python 3.12, `uv`, CUDA 12.9 and SM70 GPUs. The lock pins the CUDA 12 stack;
setup builds the local AOT kernels and patched SM70 Marlin, including FP8.

```bash
git clone https://github.com/heislera763/sglang-v100-plus.git
cd sglang-v100-plus
bash v100_plus/setup.sh
```

Common environment and arguments, from the repository root:

```bash
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

These are manual launch references. The OpenAI base URL is
`http://<server>:9000/v1`; adjust paths, GPU order and endpoint for your host.
**Always keep strict dispatch enabled during development.** Guarded operations
fail with tensor metadata when coverage is missing; declared Volta cuBLAS,
Torch CUDA and Marlin implementations are valid primary backends. The guard
does not instrument every upstream operation or guarantee the fastest kernel.

## Qwen launch references

**NVFP4, TP4, thinking + two-step MTP:** choose a single NVLink quad.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run --no-project .venv/bin/python -m sglang_v100_plus "${sglang_args[@]}" \
  --model-path "$HOME/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4" \
  --served-model-name qwen3.8-flash-next --quantization modelopt_fp4 --fp4-gemm-backend marlin \
  --tensor-parallel-size 4 --mem-fraction-static 0.88 --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --speculative-use-rejection-sampling \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

**Official FP8, TP4×PP2 + EP4, ordinary decode:** all eight GPUs. EP keeps whole
640-wide experts together, preserving checkpoint 128×128 scale blocks; ordinary
TP4/TP8 expert shards would cross those blocks. PLE stays in pinned host memory.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
export SGLANG_PP_LAYER_PARTITION=24,24
export SGLANG_OPT_SM70_HC_PREFILL_SP=1 SGLANG_ENABLE_METADATA_GLUE_GRAPH=1
uv run --no-project .venv/bin/python -m sglang_v100_plus "${sglang_args[@]}" \
  --model-path "$HOME/models/sglang/Qwen-Qwen3.8-Flash-Next-FP8" \
  --served-model-name qwen3.8-flash-next --quantization fp8 \
  --json-model-override-args '{"language_model_only":true}' \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 --expert-parallel-size 4 \
  --mem-fraction-static 0.88 --chunked-prefill-size 4096 --attention-backend triton \
  --linear-attn-prefill-backend tilelang_v100 --linear-attn-decode-backend triton \
  --page-size 64 --kv-cache-dtype fp8_e5m2 --qsa-indexer-dtype float16 \
  --ple-offload-embedding --mamba-radix-cache-strategy extra_buffer \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}' \
  --preferred-sampling-params '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"frequency_penalty":0.0,"repetition_penalty":1.0}'
```

For FP8 TP8, use TP8/EP8, omit PP and unset the partition/HC/metadata flags.
For FP8 PP2 MTP, keep HC/metadata flags and 4096-token chunks, set
`SGLANG_ENABLE_PP_SPEC=1` and add Qwen's two-step speculative arguments above.
The metadata flag also enables the tested first-stage commit graph, with live
request/acceptance buffers refreshed each round; tracking and broader batching
keep eager commits. Both TP4 groups use their quad's NVLink peers and CPU socket.
Two-step MTP improves sampled generation in these measurements; prefill remains
slower because the draft also processes the prompt. Ordinary/MTP table rows
from different revisions are not a controlled speedup comparison.
The JSON language-only override skips vision loading; Qwen's CLI `--language-only`
selects a separate encoder workflow.

## GLM launch reference

**NVFP4, TP8, ordinary decode:** the current single-request default.

```bash
export CUDA_VISIBLE_DEVICES=1,0,2,3,4,5,6,7
unset SGLANG_PP_LAYER_PARTITION SGLANG_ENABLE_PP_SPEC
unset SGLANG_OPT_SM70_HC_PREFILL_SP SGLANG_ENABLE_METADATA_GLUE_GRAPH
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

For TP4×PP2, use TP4/PP2 and `SGLANG_PP_LAYER_PARTITION=24,21`. For MTP, add
`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4 --speculative-use-rejection-sampling`; PP2 also
needs `SGLANG_ENABLE_PP_SPEC=1`. Keep serialized scheduling. TP8 MTP has no
established aggregate advantage here; PP2 improves prefill and trades away
ordinary single-request generation speed.

Experimental [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) works
with TP8 and Triton draft attention. Add `--speculative-algorithm DFLASH`,
`--speculative-draft-model-path <drafter>`, `--speculative-draft-model-quantization unquant`,
`--speculative-draft-attention-backend triton` and `--speculative-num-draft-tokens 8`.
Its BF16-trained output convolution overflows FP16; the adapter keeps convolution
outputs/residual normalization in FP32 until normalized FP16 projections.
Original and locally converted W8A16 drafts both produced about **19 generation
tokens/s**, versus **40** without speculation, on the three sampled inputs.
FP8 dense numerical checks passed, but compression brought no speed advantage.
An FP8 GLM target cannot fit fully resident: routed experts alone need 283.5 GiB.

## Preliminary performance

Eight V100 SXM2 32 GB GPUs, fan control corrected, concurrency one, coherent
near-8K prompts and 512 sampled output tokens. Rates divide total native tokens
by total execution time: prefill uses the prefill interval; generation excludes
the first token. Excluded warmups, no cached input or retractions. **A**, **D**,
**E** and **F** are nine-request confirmations; **C** is a three-request smoke check.
Rows from different revisions are not a controlled attribution of an upstream gain.

| Checkpoint / layout | MTP steps | Prefill tokens/s | Generation tokens/s | Evidence |
| --- | ---: | ---: | ---: | :---: |
| Qwen NVFP4 / TP4 | Off | 3,712 | 70.1 | A |
| Qwen NVFP4 / TP4 | 2 | 3,387 | 99.1 | A |
| GLM NVFP4 / TP4×PP2 | Off | 1,719 | 24.8 | A |
| GLM NVFP4 / TP4×PP2 | 3 | 1,593 | 36.1 | F |
| Qwen FP8 / TP4×PP2 + EP4 | Off | 6,989 | 60.8 | D |
| Qwen FP8 / TP4×PP2 + EP4 | 2 | 6,657 | 69.3 | F |
| GLM NVFP4 / TP8 | Off | 1,163 | 39.6 | C |
| GLM NVFP4 / TP8 | 3 | 1,084 | 37.5 | C |

Runtime revisions: **A** [`cf7f7e9f`](https://github.com/heislera763/sglang-v100-plus/commit/cf7f7e9fa42baf4fe683ab0444183b7ca8e2e409),
**C** [`c7442393`](https://github.com/heislera763/sglang-v100-plus/commit/c7442393),
**D** [`5c019a9a`](https://github.com/heislera763/sglang-v100-plus/commit/5c019a9a),
**E** [`6f211dc6`](https://github.com/heislera763/sglang-v100-plus/commit/6f211dc61251952a88d0476a13565c5c05dcf95a),
**F** [`9212de6d`](https://github.com/heislera763/sglang-v100-plus/commit/9212de6ddcac40b7a4617782d9e98a395d618076),
2026-10-07/08, Torch `2.13.0+cu126`. Table rows use eager prefill/full batch-one
decode, strict dispatch, no overlap/radix cache, context/cache 12288 and
2048-token chunks; Qwen FP8 D/E/F use 4096-token chunks, HC partitioning and metadata graphs.
FP8 expert checks include independent FP16 references and exact batched-versus-row
comparisons. MTP's retained proposal probabilities passed native bit comparisons;
packed results preserve typed field bits and CUDA event ordering. Native protocol
coverage includes EOS, logprobs, cancellation and reuse. One/three-step FP8 screens
gave 51.6/59.9 generation tokens/s; two steps remain the better measured choice.
E enables the existing masked Tensor Core attention kernel for MTP prefill:
matched before/after runs improve 6,165→6,585 prefill tokens/s (+6.8%), with
generation unchanged at 66.1. All 468 native target/draft attention comparisons
pass the established FP16 tolerances (rtol=.005, atol=.003); selected keys stay
unchanged. These checks establish bounded numerical agreement, not bit equality.

F improves sampled PP2 generation: Qwen **66.0→69.3 (+5.1%)**, GLM
**31.7→36.1 (+13.9%)**, with unchanged prefill. Qwen alternates nine eager/nine
commit-graph requests on one warmed model; GLM uses nine requests per transport.
All 80 native Qwen state comparisons and 712 GLM proposal comparisons pass
bitwise, alongside packed-field/event and EOS/logprob/cancel/reuse checks.
GLM's nested CUDA logprobs now travel as tensors instead of restoring pickled
storage on the sender's GPU. PP2 prefill stages already overlap; its scheduler
overlap restriction remains, and prefill graphs gave no advantage in our screen.

Inputs: [NVIDIA SPEED-Bench `throughput_8k`](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1),
revision `454f88454792dfa3ccfd7ef15fff248efde44cd1`. First turns of
`91d6ca2afe114d3c99312e8758b6f964` (code), `658dbd96a19e4b138e0aafe43eea1101`
(book continuation), `9a936eebf8794621a11f963a56c92120` (reference excerpt).
The third instruction was adapted into a roughly 300-word themed briefing of
relationships, concrete examples and the author's claims. Qwen input counts:
8517/8320/8346; GLM: 7898/8116/8141. Native metrics come from
[`return_meta_info`](python/sglang/srt/entrypoints/openai/serving_chat.py);
llama.cpp's HTTP client requires adapting its engine-specific `timings` extraction.

Weights: [Qwen NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(transferred snapshot has no retained Hub revision),
[GLM NVFP4 `f46cf340`](https://huggingface.co/RadixArk/GLM-5.3-Flash-NVFP4/tree/f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35),
[Qwen FP8 `236dfdf2`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13).
Sampling follows the [Qwen thinking recipe](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#api-usage)
and [GLM task recipes](https://huggingface.co/zai-org/GLM-5.3-Flash#footnotes):
T=1, top-p=.95, Qwen top-k=20/thinking-xhigh, GLM unrestricted top-k/max effort,
zero additive penalties/repetition penalty=1. Seed 531 is not a determinism
guarantee with ordinary sampling. MTP uses branch width one and classical
rejection sampling; draft steps D imply D+1 verification positions. Nonzero
history penalties currently lack ordinary decoding's per-token speculative semantics.

## Long context and multimodal inputs

**Qwen FP8 1M YaRN:** use the ordinary PP2 reference, disable HC partitioning
and metadata graphs for the validated capacity profile, set
`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, and replace its override with:

```bash
--context-length 1000000 --max-total-tokens 1000000 --max-mamba-cache-size 1 \
--json-model-override-args '{"language_model_only":true,"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
```

These [official settings](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8#best-practices)
completed 977233 input + 512 sampled output tokens with 1M cache slots and the
original checkpoint configuration unchanged. This validates execution capacity;
million-token retrieval quality remains unmeasured. Keep short-context profiles
on the original RoPE configuration.

**Images/video:** Qwen FP8 uses `{"language_model_only":false}`; GLM omits
`--language-only`. Keep `--mm-attention-backend sdpa`, and disable Qwen's HC
partition/metadata options for the tested vision profile. Both models processed
swapped-color images and a four-second H264 clip through OpenAI `image_url` /
`video_url` content parts, including base64 data URLs. `video_config` can set
`fps: 2, max_frames: 8`; Qwen also accepts `cap_pixels_per_frame: true`. Frames
carry timestamps/frame indices into the processors and temporal RoPE. GLM's
512-token video response exhausted its thinking cap before a final answer.

Qwen's optional long-video size budget is
`{"longest_edge":469762048,"shortest_edge":4096}` (about 224K video tokens),
set through `--mm-process-config '{"video":{"size":...}}'` rather than changing
checkpoint files. Frame caps, sampling, decoded host frames and encoder/KV memory
also need budgeting; hour-scale execution and quality remain untested. GLM uses
its own processor budgets. The locked environment includes CPU video decoding.

Before/after upstream updates, run
`uv run --no-project .venv/bin/python v100_plus/check-core-diff.py` and the
affected numerical/native checks. Prefer upstream fixes when available; retain
the small plugin/operator boundary rather than duplicating model and scheduler code.

[Apache-2.0](LICENSE). Credit to SGLang, the original V100 fork, Marlin/vLLM and
the model authors; exact source lineage is recorded in provenance.
