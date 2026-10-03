# SGLang V100 lite

Mainline base: `bd66ce343e4f6e2f2b75d7e820fe4d0718a8d824`.
Original V100 reference: `dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3`.

For the 2026-10-03 move to the main server, see [V100-HANDOFF.md](V100-HANDOFF.md).
The source testbed now has no GPUs; deployment and performance statements below
describe the earlier runs, not a currently running source-host service.

## Use

This opt-in profile serves the existing RadixArk Qwen3.8 Flash Next NVFP4
checkpoint on **9001, GPUs 0–3**, using TP4, MTP and one request slot.
CPU and memory affinity follow the first selected GPU's NUMA node, with a
`NUMA_NODE` override. Production port 9000 is outside this project; GPU numbering
can change after hardware moves, so verify ownership before launching.
There are no Docker requirements, model downloads, copies or compatibility links.
Model storage remains:
`/home/alexander/.llama-server/models/sglang/RadixArk-Qwen3.8-Flash-Next-NVFP4`.

Run `bash v100_lite/setup.sh` once to create the uv environment and build native
extensions. Run `./sglang-server.sh` to serve. The launcher installs nothing and
accepts only port 9001. The installed service is `sglang-openai-9001.service`.
The profile uses FP16, page-64 E5M2 KV, pinned CPU PLE, SDPA vision,
a 262,144-token total context limit, and an 8,192-token prefill chunk.
The NVLink launcher now allows mainline's default custom all-reduce; pass
`--disable-custom-all-reduce` to restore the earlier NCCL-only configuration.

Setup defaults to a project-local CUTLASS v4.2.1 checkout; `CUTLASS_DIR` can
select an existing header tree, and `CUDA_HOME` selects the compiler toolkit.
The launcher accepts `MODEL_PATH`, `LLAMA_API_KEY`, `ENV_FILE`, and `NUMA_NODE` overrides.
Its existing model and credential defaults are relative to `$HOME`.
Port 9001 and GPUs 0–3 remain deliberate constraints of this host's test
launcher. The deployment unit lives outside Git; the unused copy containing
absolute host paths was removed. This improves setup portability without claiming
a generic hardware configuration or support for additional model families.

## Maintenance surface

The tracked diff against the base is **49 files, 9,397 added lines and six
removed lines**, versus 171 files / 42,743 added lines before reduction and
56 files / 11,740 at the start of this pass. About 5,000 lines are runtime/kernel
implementation and 2,694 are the reproducibility lock. Combining files alone
does not remove complexity; unreachable implementations and experimental
branches were actually deleted.

Seven existing mainline Python files change, totaling **61 added and six removed
lines**. Three preserve exact server timing counters and optional timing-only
SSE chunks for Pi; EAGLE avoids importing unused optional DeepSeek backends;
GDN adds a default-off constexpr and state reload, totaling three added lines.
Two files expose FP16 compressed QSA index-cache storage; the launcher selects
`--qsa-indexer-dtype float16` explicitly, leaving the stock default unchanged.
The SM70 plugin enables this specialized fix for low-precision sequential MTP
verification. Mainline already supports xhigh; there is no xhigh patch.

The AOT profile compiles 17 existing upstream/dependency translation units and
a small operator registry, then packages the unchanged mainline `sgl_kernel`
SDK. The copied native tree is gone. There are 19 compilation/link steps versus
48 before the reduction. The package is `sglang-kernel==0.4.8+v100`.
Marlin's three adjustments are consolidated into one patch at a pinned revision.

The opt-in integration and 23 required hooks live in `runtime.py`.
Quantization, QSA dispatch and PLE each have one adapter module. GPU code lives
under `kernels/`; GDN prefill and its interface share one module, as do the
TileLang attention helpers. Imported-source revisions/hashes remain in
`v100_lite/provenance.json`. Default registration imports no runtime or kernels;
missing required hooks stop startup. Custom GDN dispatch requires the explicit
`tilelang_v100` choice. Non-mutating GDN scoring is rejected rather than silently
modifying state.

Removed alternatives include packed GDN prefill/decode that mainline never
called, experimental GDN schedules, duplicate MQA decode, unused QPN8 code,
ordinary paged CUDA attention beside QSA, and separate cache-registration glue.
GDN prefill shrank from 2,258 lines in two files to approximately 675 lines in
one module. Runtime scheduling, QSA metadata, model definitions, decode/verify
kernels and checkpoint loading remain mainline implementations.

This is substantially smaller and more compartmentalized, but it is a scoped
Qwen TP4 serving profile, not general Volta support for every mainline model.
Most remaining custom complexity is GPU math that Volta needs or benefits from.
Its ownership is explicit; GPU reference coverage improves confidence without
proving broad model accuracy or eliminating every inherited assumption.

## Performance and comparison

The first table was measured on the previous mainline base `fc9bdc8a`.
The NVLink runs below use `bd66ce34`; the original fork has not been rerun on
the newly installed NVLink quad.

The before/after runs use the same local checkpoint, GPUs, one request slot,
TP4/MTP, greedy sampling, uncached 1K/8K/25K inputs and 1,024 output tokens.
Each length has a 256-output-token warmup followed by three measurements.
PP is exact prompt count divided by server dispatch-to-first-token elapsed time;
TG uses server output counters and decode elapsed time. These include server
overhead and are not pure GPU compute rates.

| Input tokens | Lite before PP | Lite now PP | Fork PP | Lite before TG | Lite now TG | Fork TG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1730.0 | 1719.0 | 1776.5 | 82.08 | 92.41 | 93.16 |
| 8,192 | 1996.3 | 1979.3 | 1754.9 | 82.99 | 90.74 | 90.85 |
| 25,000 | 1961.5 | 1944.2 | 1885.3 | 84.64 | 90.06 | 88.42 |

Rates are tokens/s; each cell is the median of three completed measurements.

TG improves **12.6%, 9.3% and 6.4%** at 1K/8K/25K versus the fresh
lite-before run. PP changes by -0.64%, -0.85% and -0.88%; these small observed
differences do not establish a statistically significant regression. Relative
to the matched fork, TG is -0.8%, -0.1% and +1.9%; PP is -3.2%, +12.8% and
+3.1%. Draft acceptance changes too, so these are useful deployment comparisons,
not proof of identical logits or an isolated speedup for either new hook.

The fresh fork uses a uv venv with read-only access to the original package set;
Torch binary hashes and all recorded package versions match the original Conda
environment. This checks invocation/package parity without another dependency
download. uv alone does not change GPU math. The baseline uses the same page
size, sparse cutoff, prefill chunk, sampling backend, CPU affinity and preferred
NUMA node. It needs static memory fraction **0.80**, versus lite's **0.88**:
the first attempt at 0.88 ran out of CUDA memory at 25K after passing 1K/8K.
Its failed results are retained and excluded from the completed comparison.
Thus serving memory layouts and native implementations still differ.

Earlier old-fork defaults used dense no-prefix prefill through 8K. Both current
comparison launches cap that shortcut at the checkpoint's 2,048-token selection
budget. Above that, dense and sparse attention perform different computation;
changing the cutoff for speed is not an innocuous tuning choice.

The original [README](https://github.com/haohervchb/sglang-V100/blob/dca488908ee4e3f1bc676c3bf5dcd26ff049cfc3/README.md)
reports **measured NVLink** results, not theoretical limits: MTP-3/4 gives
117.55/119.84 TG and 3,269/4,693 client-measured PP at 1K/25K. The fresh local
fork gives 93.15/88.42 client TG and 1,762.79/1,883.35 client PP; lite gives
92.40/90.05 and 1,704.47/1,942.31. Those PCIe runs did not reproduce the
README's NVLink numbers. Hardware, custom-all-reduce choice, chunk size, dense cutoff,
software paths and acceptance differ; the entire gap cannot be assigned to
NVLink alone. The local fork's 1K result is within 1% of its earlier local run.

### NVLink quad baseline, 2026-09-30

The `bd66ce34` profile completed the same three-repeat workload on the newly
installed quad, with NUMA-node-1 affinity and NV2 connectivity between every
pair. API checks passed for xhigh, tools, images and exact streaming timings.
This first NVLink launch used NCCL, with custom all-reduce disabled.

| Input tokens | Server PP | Server TG |
| --- | ---: | ---: |
| 1,000 | 3212.6 | 88.24 |
| 8,192 | 4589.5 | 87.44 |
| 25,000 | 3399.8 | 85.02 |

These are deployment observations before a physical connection check, not an
isolated NVLink comparison: the mainline base and GPU placement also changed.
PCIe generations remained 3/1/2/3 at x8 during inference; a direct host-transfer
test was deferred. GPU 1 (`84:00.0`) is also the card that dropped off PCIe on
the preceding boot. Systemd recorded 7 GB peak swap during this launch; whether
swap affected timed requests was not measured. Link negotiation, host transfers,
benchmark-time swap and custom all-reduce remain follow-up checks.
Raw measurements and telemetry are in `artifacts/nvlink-quad/` (untracked).
The test endpoint was stopped for the user's physical connection inspection.
Longer-prompt differences also reflect the corrected sparse cutoff.

### NVLink with mainline custom all-reduce, 2026-09-30

After reboot, removing `--disable-custom-all-reduce` enabled mainline's v2
custom all-reduce. Startup confirmed symmetric-memory initialization with pull
enabled and multicast disabled. No SGLang runtime/kernel edits were needed.
The same TP4/MTP-3/4, single-slot, uncached workload completed all nine measured
requests, with 1,024 output tokens each. These use greedy non-thinking prompts;
xhigh was checked separately through the API smoke test.

| Input tokens | Server PP | Server TG | TG vs earlier NVLink/NCCL |
| --- | ---: | ---: | ---: |
| 1,000 | 3107.7 | 121.90 | +38.1% |
| 8,192 | 4590.8 | 112.47 | +28.6% |
| 25,000 | 4052.8 | 117.72 | +38.5% |

These are medians of three measurements using server timing fields. Compared
with the earlier PCIe lite baseline, PP is +80.8%/+131.9%/+108.5% and TG is
+31.9%/+23.9%/+30.7%. Hardware placement, link negotiation and draft acceptance
also changed, so these differences do not isolate the collective implementation.
For the original README's client-timing comparison, current 1K/25K PP is
3,064.36/4,043.31 (-6.3%/-13.8%) and TG is 121.88/117.70 (+3.7%/-1.8%).
This approximately reproduces its decode throughput; prompt throughput is lower.

All four GPUs remained available with no service restarts or logged Xid errors.
PCIe remained Gen3/Gen3/Gen1/Gen1 x8, with NV2 between every pair and NUMA-1
affinity. Host swap was about 4.75 GiB in use during inference; sampled free swap
varied by only about 10 MiB, which does not establish whether swap I/O occurred.
The configured context limit is 262,144 tokens; this run exercised up to 25K
input plus 1K output, not the full limit. API sampling, xhigh, tokenization,
images, tool round trips and exact streaming timing checks all passed.
Raw measurements and telemetry are in `artifacts/nvlink-car/` (untracked).

On 2026-10-01, a quick check after forcing Gen2 retained Gen2/Gen2/Gen1/Gen2
x8 and completed one measured request per length, following warmups:
1K input gave 3,162.5 server PP / 122.45 server TG; 25K gave 4,379.3 / 118.88.
Each generated 1,024 tokens, and an OpenAI chat check returned `Test passed`.
These single measurements do not replace the three-repeat baseline above.
All GPUs stayed available with no service restarts or logged Xid errors.
Replay counts increased by 0/26/0/16 from loading through the final chat check;
GPU 3 also recorded seven replay rollovers, so Gen2 did not eliminate link errors.
Sampled peak temperatures were 63/64/74/72 C. Evidence is in
`artifacts/nvlink-gen2-quick/` (untracked); no runtime or launcher edits were made.

## Why retain custom kernels?

Mainline Triton GDN prefill runs correctly on SM70. Independent FP32 sequential
recurrence agrees for non-aligned variable-length sequences, nonzero indexed
FP16 state, outputs and checkpoints. However, CUDA graph microbenchmarks of
the TP4 shape measured **10.35 versus 1.25 ms at 1K**, and **77.44 versus
3.23 ms at 8K**, for mainline versus the custom TileLang path. These isolated
kernel-path measurements are not whole-model speedups. Keeping the active
custom schedule is justified by measured cost, rather than an assumption that
mainline cannot run on V100.

Earlier decode profiles identified mainline's larger GDN verification tile and
slower unified router. The profile now selects **BV=8** for the exact single-slot
TP4 FP16 verification shape and routes eligible top-10/512-expert calls through
the already-retained native kernel. Other contracts use mainline. This adds
two bounded dispatch changes and no further mainline edits or copied kernels.

Previous profiles also found a fused QKVZ/BA projection in the old fork versus
separate projections/layout work in lite. Uncalled copies of that fusion were
removed; adapting an active model boundary is future work. Raw profiler kernel
durations cannot be summed to explain end-to-end latency because collectives
wait and profiling adds overhead.

| Decode evidence | Fork | Lite before tuning | Interpretation |
| --- | ---: | ---: | --- |
| Router average | 7.65 us | 29.69 us | Different native/Triton implementations |
| GDN verify average | 14.56 us | 35.19 us | Different value tiles and launch grids |
| NCCL all-reduce average | 117.57 us | 55.69 us | Includes waiting; not proof of a PCIe bandwidth change |

| Overlap opportunity | Evidence needed |
| --- | --- |
| Collective/compute overlap | Same-iteration multi-rank capture with minimal profiling overhead |

| Fusion candidate | Maintenance cost |
| --- | --- |
| QKVZ/BA projection plus layout | Adapt the current projection boundary; avoid copying another model class |

## Torch, CUDA and NUMA

Torch **2.13.0/cu126**, Triton **3.7.1**, and the CUDA **12.9 compiler** execute
this profile. The old package set is Torch **2.9.1/cu128**, Triton **3.5.1**, and
CUDA **12.8**. Runtime libraries packaged with Torch and the extension compiler
are separate choices; actual SM70 compilation and execution pass.

Identical-source, warmed GPU microbenchmarks compared the old and new stacks
with the same CUDA 12.9 assembler. FP16 GEMM samples at M=1/4/1024 were
25.92/32.27/163.27 us on the old stack and 26.22/32.25/162.61 us on the new one.
Both use Volta cuBLAS kernels. The same BV=8 GDN verify source measured
10.29 versus 8.74 us. Clocks were observed, not locked; these bounded samples
show no large GEMM penalty from cu126 but do not isolate CUDA, Torch, Triton
or NCCL individually. In particular NCCL is 2.27.5 versus 2.29.3. They do not
justify another multi-gigabyte wheel download or a stack change without a
controlled whole-model comparison.

Both launches use `numactl --cpunodebind=0 --preferred=0`, with worker CPU masks
`0-21,44-65`. During measured operation, lite workers had 0.15-0.16% resident
memory on node 1, versus 2.58-3.96% for the fork. This is a placement observation,
not a controlled NUMA speedup. Prefer-node policy is retained: strict binding
was not benchmarked, and would remove fallback under node-0 memory pressure.
Library/file pages can already reside on another node despite the process policy.

Torch 2.9.1/cu128 is not the last usable V100 combination. PyTorch's official
[2.11 packaging notice](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290)
retains Volta in cu126 while removing it from newer wheel variants. Its
[2.15 packaging notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)
identifies **2.14/cu126** as the final prebuilt Volta option; it has not been
tested here. [CUDA 13](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
removes offline compilation support for older architectures. A later Torch
source build against CUDA 12.x is a separate investigation.

## Validation and next refinements

GPU references pass exact E5M2 byte conversion, MoE padding guard words,
NVFP4 decode/prefill against dequantized weights, router ids/ties/weights against
Torch and mainline, dense SDPA and sparse selected-row attention with permuted
pages, compressed page sizes 4/16, FP16 MTP versus sequential decode, and GDN
outputs/state/checkpoints versus independent FP32 recurrence. Twenty mainline
hook-registry tests pass. Module and patch consolidation reproduce byte-for-byte.
The default-off plugin check also passes.

The final API replay passes text, Qwen sampling, xhigh, tokenization, an image,
a tool call/result roundtrip and live server timing counters. The server remains
on 9001 with 465,792 allocated KV tokens and a 262,144-token total context limit.
This pass measures up to 25K input; earlier 200K-input requests passed before
this pass, and the full context boundary was not retested. Production PID
416531 remains unchanged; the temporary fork service and checkout are removed.

Three retained validation entry points are `v100_lite/tests/gpu_checks.py`,
`smoke.py`, and `benchmark.py`. For Pi, `return_timing_metrics: true` emits server
counts, TTFT and live decode rates under `sglext.timing_metrics`.
Authenticated API checks use exported `API_KEY` or `LLAMA_API_KEY`; they do not
read private credential files. Smoke replay needs no artifact directory unless
`--tag` is supplied. Benchmark tokenizer discovery uses the server configuration,
with `--model-path` or `MODEL_PATH` as an explicit override.
One overlapping plain-text request and duplicate GPU-test setup/count assertions
were removed. All numerical reference cases remain: the dtype, shape, page-size
and checkpoint variants exercise different supported paths. The empty benchmark
state update remains because it resets speculative acceptance counters.
Raw measurements, sanitized settings, numerical checks, environment manifests,
NUMA snapshots, proofs and logs are consolidated in ignored
`artifacts/lite-audit/`; older evidence stays in `artifacts/reduction/`.

Future work should be small and discriminating:

- Benchmark an additional Torch wheel against identical source and a rebuilt
  native package, separating Triton/cuBLAS/NCCL changes from Python packaging.
- Profile multi-rank collectives and CPU launch gaps across the main server's
  two NVLink quads, including the PCIe connection between the quads.
- Evaluate QSA/indexer kernel efficiency while preserving the selection budget;
  verify cached-prefix and context-boundary behavior before expanding support.
- Expand actual-checkpoint/logprob and scale-distribution accuracy checks for
  inherited quantization and GDN math; synthetic references are not a quality eval.
- Consider projection fusion only if its measured gain warrants another hook.
- Upstream the small generic import/timing/state-precision changes where appropriate.

## Applying to stock mainline

The profile applied cleanly to upstream `bd66ce34` (42 commits after the
previous base), with no textual merge conflicts. The only required integration
adjustment was FP16 support for the new independent QSA index-cache dtype
selector. GDN prefill hooks and QSA metadata changes remain upstream code.
The entire adaptation is a single commit directly on that stock parent.

On this stock base, 38 unit tests, FP16 cache allocation/aliasing, all GPU
reference checks, and the API replay pass. Two 5,522-token prompts also pass
through the sparse path. Port 9001 runs this version; production 9000 is unchanged.
The existing uv environment and unchanged native extension were reused; this
was not a fresh installation or a new performance benchmark.
