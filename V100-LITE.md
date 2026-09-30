# SGLang V100 lite

Mainline base: `fc9bdc8a3e60a0896e58d84ac261cc470ab4a24c`.
Original V100 reference: `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`.

## Scope and installation

This opt-in profile serves the existing RadixArk Qwen3.8 Flash Next NVFP4 checkpoint
on **9001, GPUs 0–3, NUMA node 0**, with TP4, MTP and one request slot. Port 9000,
GPUs 4–7, the original fork, its environment and its launchers stay untouched.
No Docker, model copies, downloads or compatibility links are required.

Run `bash v100_lite/setup.sh` once. It creates the uv environment and builds the
SM70 native extensions. Run `./sglang-server.sh` to serve; the launcher installs
nothing and only accepts port 9001. The installed service is
`sglang-openai-9001.service`. The launcher uses FP16, page-64 E5M2 KV, pinned CPU
PLE, SDPA vision, TP4/MTP and one-slot settings. Model storage remains:
`/home/alexander/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4`.

## Small maintenance surface

The reduction removes the copied native source/header/Python-SDK tree. The AOT
profile compiles 18 existing mainline/dependency translation units plus a small
operator registry, and packages mainline's existing `sgl_kernel` Python SDK.
Unused CPU, HIP, FP8, GPTQ and QServe copies are gone. The native build has 19
compilation/link steps rather than 48. Its package version is `0.4.8+v100`; the
uv override reconciles this with mainline's older 0.4.7 dependency pin.

Runtime adapters and 22 required hook registrations live in one `runtime.py`.
GEMV and small GEMM share `kernels/gemm.py`. Uncalled projection-fusion wrappers
and their CUDA headers were removed. Quantization, QSA, PLE and the substantial
GDN kernels remain separate because they own distinct implementations.
Imported-source hashes and merged-source origins remain in `v100_lite/provenance.json`.

Five existing mainline Python files have small changes:

- Three preserve the opt-in Pi API contract: server timings and timing-only SSE chunks.
- EAGLE defers optional DeepSeek imports when the known backend already matches.
- GDN adds a default-off constexpr and a state reload: **three added lines**.
  Only the SM70 plugin enables it for sequential low-precision verification.
  This preserves the specialized FP16 MTP rounding fix while removing its
  568-line copied kernel. Ordinary mainline execution retains the default-off path.

Mainline already supports xhigh and passes it to the checkpoint's template;
there is no xhigh patch. Required hooks must all install before serving begins.
Default plugin registration imports no runtime or kernels.

## Collected performance

Same checkpoint and local PCIe hardware, TP4/MTP, FP16, E5M2, pinned CPU PLE,
one slot, greedy sampling, uncached 1K/8K/25K input and 1024 output tokens.
A 256-output-token warmup is excluded at each prompt length. PP is exact prompt
count divided by server-measured dispatch-to-first-token time; TG uses server
output counts and decode elapsed time. These are server measurements, not
GPU-kernel-only timings or client clock estimates.

| Prompt tokens | Fork PP | Lite before PP | Reduced PP | Fork TG | Lite before TG | Reduced TG | Reduced repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1776.2 | 1740.5 | 1724.1 | 93.99 | 81.93 | 81.87 | 3 |
| 8,192 | 2130.9 | 2011.1 | 1986.9 | 92.51 | 84.76 | 84.24 | 3 |
| 25,000 | 2008.7 | 1978.1 | 1960.3 | 90.34 | 82.78 | 80.96 | 1 |

Rates are tokens/s. Fork and lite-before have three repeats at every length.
**Benchmarking was stopped at the user's request.** Reduced 1K/8K have three
repeats; the 25K reduced result is one completed measurement, not a median of three.
No performance probes were run after the stop request.

At 1K/8K, reduction changes TG by less than 1% and PP by about 1% compared with
lite-before. It does not recover the existing fork-to-mainline performance gap.
Reduced verification cycles are 29.29/29.51 ms at 1K/8K, versus fork
26.97/27.19 ms. Different draft acceptance adds a separate source of TG variation.

The comparison has material differences: lite uses page 64/compressed page 16
and static memory fraction 0.88, versus fork page 16/compressed page 4 and 0.80.
Lite respects the checkpoint's sparse-selection budget above 2048 tokens;
the original fork uses dense first-chunk attention through 8192. Thus 8K/25K
are deployment comparisons, not identical attention work or identical outputs.
Native reductions can change FP16 accumulation order slightly; reference checks
pass, but this is not a broad model-accuracy evaluation.

## Why throughput differs

Rank-0 CPU/GPU traces were collected for the fork and lite-before, separately
for prefill and decode. Prefill captures include first-token speculative work.
Profiles establish different execution paths; their overhead and collective
waiting mean raw GPU durations cannot simply be added to explain request latency.

| Decode kernel/path | Fork average GPU duration | Lite average GPU duration | Evidence |
| --- | ---: | ---: | --- |
| Top-10 router | 7.65 us | 29.69 us | Fork native radix router versus mainline Triton router |
| FP16 GDN verification | 14.56 us | 35.19 us | Fork chooses BV=8 for the TP4 MTP shape; lite uses mainline BV=32 |
| GDN projection | One fused QKVZ/BA kernel | Separate small GEMMs and layout work | Fork model dispatch calls its SM70 fusion; lite's copied wrappers were never called |
| NCCL all-reduce | 117.57 us | 55.69 us | Collective waits and profiling differ; this does not demonstrate a PCIe regression |

The original BV=8 tuning launches 192 CTAs rather than 48 for this shape,
reducing register pressure and exposing more parallel work. Restoring this tuning
and the router dispatch are small, concrete follow-up candidates. Restoring the
projection fusion requires adapting the current model boundary; retaining unused
copies alone provides no speed benefit. These are observed path differences and
plausible contributors, not isolated end-to-end speedup measurements.

| Overlap opportunity | Confidence | Next discriminating measurement |
| --- | --- | --- |
| Collective/compute overlap | Unproven from these individual traces | Same-iteration multi-rank trace without stack-capture overhead |

| Fusion candidate | Current evidence | Maintenance implication |
| --- | --- | --- |
| QKVZ/BA projection plus output layout | Original fork uses its fused kernel; mainline separates projections | Requires a hook at the current projection boundary, not an unused file copy |
| Expert routing | Mainline router is slower per profiled call | Existing SM70 top-10 kernel can be considered through an opt-in hook |

PCIe communication remains a substantial part of prefill: NCCL all-reduce is
about 55–59% of profiled GPU kernel time for both runtimes. That is collective
execution/wait time, not a direct PCIe bandwidth measurement. Both baselines use
this host's non-NVLink configuration. Kernel choice, launch overhead, software
version and speculative acceptance must be separated from hardware limits.

## Validation and context

The reduced native package imports and passes exact E5M2 byte comparisons, MoE
padding guard-word checks, NVFP4 MoE decode/prefill dequantized references,
dense/sparse attention references with permuted physical pages, and four-step
FP16 MTP versus sequential decode. Twenty mainline hook-registry tests pass.
Both consolidation commits reproduce byte-for-byte from their parents.

The server loads the target and MTP models and completes generation at 1K, 8K
and 25K. Earlier full API replay established xhigh, images, tools, tokenization
and live server timings; those API integration edits remain unchanged. That full
API replay was not repeated during this reduction pass. Current allocated KV
capacity is 465,792 tokens and the advertised total context limit is 262,144.
Two earlier 200K-input requests passed; 200K and the full 262K boundary were not
retested in this pass.

Retained validation entry points are `v100_lite/tests/gpu_checks.py`,
`smoke.py` and `benchmark.py`. `smoke.py --disabled-plugin` verifies the default-off
contract without server inference. `smoke.py --tag NAME` exercises the 9001 API.
The opt-in streaming field `return_timing_metrics: true` emits server counters,
TTFT and live decode rates under `sglext.timing_metrics` for the existing Pi extension.
Raw results, sanitized settings, traces, proofs and logs are consolidated under
ignored `artifacts/reduction/`; historical evidence remains in `artifacts/`.

## CUDA / Torch support findings

Do not treat the working Torch 2.9.1 + CUDA 12.8 installation as the final
V100-capable PyTorch release. CUDA toolkit support, PyTorch wheel architectures,
and inference-kernel support are separate constraints.

| Candidate | Volta support evidence | Local execution |
| --- | --- | --- |
| Torch 2.9.1 + cu128 | Existing wheel includes sm_70 | CUDA arithmetic, FP16 GEMM and Triton 3.5.1 pass |
| Torch 2.10 + cu128/cu129 | Last release before 2.11 removes Volta from these wheel variants | Not tested |
| Torch 2.13 + cu126 | Official wheel includes sm_70; selected to match mainline's Torch pin | CUDA arithmetic, FP16 GEMM and Triton 3.7.1 pass |
| Torch 2.14 + cu126 | Officially the last release with a prebuilt Volta wheel | Not tested |
| Torch 2.15+ standard wheels | cu126 removed; standard wheels drop Volta | Requires investigating a source build |
| CUDA toolkit 12.9 | Local nvcc lists compute_70 and sm_70 | SM70 AOT library and NVFP4 Marlin build pass; native import, E5M2, dense, MoE and QSA checks pass |
| CUDA toolkit 13.x | Offline compilation and library support before Turing removed | Unsuitable for the SM70 extension build |

PyTorch's official [2.11 packaging notice](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
removes Volta from CUDA 12.8 and 12.9 wheels while retaining it in cu126.
Its [2.15 packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
identifies 2.14 as the last release providing a prebuilt wheel for Volta.
The [CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
record the removal of older architecture compilation and library support.

The profile uses the cu126 **runtime packaged with Torch** and the locally
installed CUDA 12.9 **compiler**. Those need not be the same minor release,
but actual compilation and GPU execution must pass. Triton 3.7.1 also passed
with its bundled CUDA 12.8.93 assembler; replacing it is not required for SM70.
The serving profile selects CUDA 12.9's ptxas to use the same local toolchain. The driver
version reported by nvidia-smi is not the CUDA toolkit or Torch runtime version.

Excluded audio and CUDA 13 acceleration packages are outside this Qwen
text/image serving profile. The profile is not a promise of support for every
upstream model or optional backend. Re-evaluate the dependency exclusions and
native build with each upstream update.
