# SGLang V100 lite

Upstream starting revision: `84523d67851171fa20f7c68d3d6dc6cbf20c4423`. Published V100 reference: `dca488908ee4`.

## Purpose and boundaries

Serve the existing RadixArk Qwen3.8 Flash Next NVFP4 checkpoint on port 9001, GPUs 0–3, NUMA node 0, with TP4 and MTP. Port 9000 is a live deployment and is never used for project requests or modified. The old fork, environment, launchers, and caches stay intact for rollback.

Use uv, explicit CUDA wheel indexes, and project-local build caches. Prefer separate compatibility modules and existing registration APIs, activated only by `SGLANG_V100_LITE=1`. Preserve xhigh, images, tools, exact tokenization, and server-derived Pi timing fields. Avoid importing obsolete model and scheduler code already replaced upstream.

## Acceptance

Capture the old 9001 baseline, investigate published PyTorch/CUDA combinations, then validate a small FP16 model and the RadixArk target/MTP service. Compare 1K/8K/25K prompts with 1024 output tokens and three repetitions, plus a 200K-context request. Aim within 10% of existing performance. Promote only after required functionality passes, otherwise restore 9001. No Docker or compatibility symlinks.

## Implementation status

The compatibility plugin, standalone SM70 build profile, API timing integration,
9001 launcher, and validation scripts are implemented. The full RadixArk TP4/MTP
candidate passed text, nonzero-temperature sampling, xhigh reasoning, images,
tokenization, tool calls and round trips, and live server timing checks. The 1K/8K/25K
benchmark is complete. Two uncached 200K-input requests passed, followed by another
complete API replay. The candidate is now installed under the original 9001 unit
name, `sglang-openai-9001.service`. Its full API replay passed after restart,
and VM localhost port 19001 returned the lite model through the existing tunnel.

- Baseline 9001 passed text, xhigh, image, tools, tokenization and live timing checks.
- Median server PP rates (prompt tokens / dispatch-to-first-token time): 1K 1772.55,
  8K 2137.14, 25K 2018.46 tokens/s.
- Median server TG rates: 1K 94.42, 8K 91.58, 25K 90.79 tokens/s.
- Existing Torch 2.9.1+cu128 passed CUDA arithmetic, FP16 matrix multiplication,
  and Triton 3.5.1 execution on GPU 0.
- A fresh uv-managed Torch 2.13.0+cu126 runtime passed CUDA arithmetic, FP16
  GEMM and Triton 3.7.1 on GPU 0. CUDA 12.9 compiled and ran the E5M2 cache
  writer (exact byte equality) and FP16 dense kernels (M=1 and M=4 reference
  comparisons). The official wheel's download-r2 host returned HTTP 403;
  the equivalent download.pytorch.org URL works and is pinned in the uv profile.
- Stock mainline's official sglang-kernel 0.4.7 wheel cannot load: it requires
  CUDA 13 libraries. Its embedded cubins cover SM80/89/90/100/120, not SM70.
  The unmodified tiny-model launch fails during kernel-library import.
- Dense D256 attention passed an SDPA reference check; four-step FP16 MTP
  recurrence matched sequential decode. NVFP4 MoE decode M=1/M=4 and prefill
  M=17 passed against independently dequantized synthetic checkpoint tensors.
  Sparse QSA prefill/decode passed selected-row reference checks with permuted
  physical cache slots. The optional NVFP4-only
  Marlin build compiled successfully. Full model API validation passed.
- The uv environment now contains the locally built `sglang-kernel 0.4.7+v100`.
  Small-model GPU generation matched a CPU reference for 16 tokens. QSA index
  scoring passed with randomized physical pages for compressed page sizes 4 and 16.
- Original 9001 unit and sanitized settings are saved in ignored artifacts.
  Only the 9001 unit is replaced. Its existing disabled-at-boot setting is preserved,
  and its VM tunnel is active. 9000's PID and configuration remain unchanged.

## Layout and update strategy

`python/sglang` follows upstream. Three Python files contain small API integration
edits: opt-in timing fields, live server counters, and timing-only SSE chunks.
A fourth makes EAGLE's backend checks import optional DeepSeek modules only
when the already-known backend types do not match. Mainline already accepts xhigh
and forwards it to the existing checkpoint template, which supplies its xhigh
reasoning instructions; no xhigh-specific SGLang patch is needed.
The plugin routes speculative top-k/top-p filtering through its SM70 AOT kernels;
mainline EAGLE otherwise calls SM75+ FlashInfer even with the PyTorch sampler.
Those filters passed independent Torch references.

`v100_lite/sglang_v100_lite` is an opt-in plugin and contains alternative kernels,
QSA execution, linear-attention dispatch, FP16 hyperconnections, PLE host gathers,
and NVFP4 layout handling. It retains current model loading, QSA metadata,
scheduling, speculation, and OpenAI request processing. Required plugin hooks
must all install before the compatibility entrypoint starts the server.
Mainline already replaces ordinary attention with QSA for this model, including
MTP. The launcher therefore selects the existing Triton backend and carries only
the alternative QSA kernels, without the old ordinary-attention backend. Mainline's
64-token pages map to 16-key compressed pages. QSA index states use FP16 on SM70.
Dense QSA prefill is bounded by the checkpoint's 2048-token selection budget;
longer prefixes use selected sparse rows. The old fork defaults to a dense first
chunk through 8192 tokens, so its 8K/25K baseline includes a different attention
policy. Performance comparisons should state that difference.

`v100_lite/aot` is the isolated SM70 native build. It uses 23 unchanged upstream
native sources directly, retains older SM70-compatible alternatives where needed,
and packages the current Python kernel API. Marlin builds from a pinned external
revision with two tuning patches and an optional NVFP4-only build profile in
`v100_lite/patches`. The serving profile builds repacking and NVFP4 MoE kernels;
other Marlin quantization families require its full build. Original source hashes and
revisions are recorded in `v100_lite/provenance.json`.

`artifacts/` stores downloads, build outputs, original-unit backup, logs, and raw
measurements; `.cache/` stores project-only runtime compilation caches. Neither
is tracked. Temporary Torch probe environments, the stock CUDA 13 wheel, the
manual native staging tree, and porting scripts were removed after validation;
approximately 7 GB of temporary directory contents were removed (apparent size;
uv's shared package cache retains deduplicated dependencies). Models remain in the
existing `models/sglang/` directory, with no
compatibility links or duplicate checkpoints.

Rebase `v100-lite` on upstream main and review the small timing edits plus plugin
hook targets. Rebuild native extensions whenever Torch or the CUDA toolkit
changes. Run the kernel checks and API/benchmark acceptance before replacing the
9001 service. Changes to mainline attention metadata should be reviewed rather
than hidden behind aliases to old namespaces.

## Measured comparison

Medians of three uncached requests, 1024 output tokens, greedy sampling; one
256-token warmup per prompt size is excluded. PP uses exact prompt-token counts
and server-measured dispatch-to-first-token time. TG uses server-observed output
counts and elapsed decode time, excluding the first output batch.

| Prompt tokens | Old fork PP | Lite PP | Old fork TG | Lite TG |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 1772.55 | 1730.60 | 94.42 | 82.18 |
| 8,192 | 2137.14 | 2001.24 | 91.58 | 85.46 |
| 25,000 | 2018.46 | 1964.75 | 90.79 | 85.37 |

All rates are tokens/s. PP is 2.4–6.4% below the baseline. TG is 6.0–6.7% below
at 8K/25K, and 13.0% below at 1K; the 1K case misses the aspirational 10% target.
MTP accept length ranged from 2.40 to 2.75 tokens per verification. This is a
functional and performance baseline, not a general accuracy evaluation.

Both runs use TP4/MTP, FP16, E5M2 KV, pinned CPU PLE, one slot, and the same existing
RadixArk checkpoint. Mainline uses page size 64 rather than the fork's 16 and a
0.88 static memory fraction rather than 0.80. The attention-policy difference
above also applies to 8K/25K. Raw request results and sanitized server settings are
in `artifacts/fork-baseline_measurements.jsonl` and `artifacts/lite_measurements.jsonl`.

The allocated KV pool holds 465,792 tokens, while the model/API context limit is
262,144 total tokens. Two requests with exactly 200,000 uncached input tokens
completed (256 and 16 output tokens). The measured 16-output-token run achieved
1,715.68 PP tokens/s; its short decode is not a representative TG benchmark.
The full 262K boundary has not been tested. The Mamba pool has 21 state slots.

The opt-in chat API field `return_timing_metrics: true`, together with server
`--enable-metrics`, emits timing-only SSE chunks under `sglext.timing_metrics`.
These contain prompt/cached/completion counts, `server_ttft`, and live
`stream_decode_throughput` when available. The existing Pi extension consumes
this contract unchanged. Ordinary chat requests retain their existing stream.

## Installation and validation

Run `bash v100_lite/setup.sh` once to create the uv environment and build native
extensions. `sglang-server.sh` is the serving launcher; it installs nothing and
only accepts port 9001 and GPUs 0–3. The supplied unit is installed only after runtime
acceptance succeeds. The launcher supplies explicit FP16, page-64, pinned PLE,
SDPA vision, TP4/MTP and one-slot settings.
The initial 0.80 memory fraction allocated only 138,688 cache tokens on mainline;
the target launcher now uses 0.88 to make room for the long-context acceptance check.
Marlin uses the existing `/home/alexander/cutlass` headers (revision recorded in
the provenance file); the SM70 AOT build downloads its own pinned header dependencies.

`v100_lite/tests/torch_probe.py`, `kernel_checks.py`, `attention_checks.py`, `noop.py`, `make_tiny.py`, `smoke.py`
and `benchmark.py` provide the staged checks. The API tests and benchmarks have
port 9001 hard-coded. Benchmark client rates are labeled separately; comparisons
use server-measured rates. PP remains tokens divided by server TTFT, including
server queue and request processing; it is not a GPU-kernel-only timing.

## Running service

The service is active on `http://127.0.0.1:9001/v1`, served model
`qwen3.8-flash-next-radixark-nvfp4`. On the VM, its existing SSH tunnel exposes
`http://127.0.0.1:19001/v1`. Authentication and Pi configuration remain unchanged.
The existing disabled-at-boot setting is preserved; starting/restarting the unit
also starts its enabled VM tunnel dependency.

Use `systemctl --user restart sglang-openai-9001.service` to restart and
`journalctl --user -u sglang-openai-9001.service -f` for logs. To use the light
`./sglang-server.sh` launcher directly, stop the 9001 unit first. The launcher
performs no installation and runs only the NUMA-0 quad. Pinned CPU PLE tables,
both loaded model runners, and the production endpoint together consumed about
226 GiB of host RAM during testing. The test unit has a high OOM score so host
memory pressure favors reclaiming it over the live deployment.

Three local commits separate the timing API, the generic EAGLE import fix, and
the opt-in compatibility runtime. The four modified upstream Python files total
52 added and 2 removed lines. Alternatives and build/runtime tooling stay under
this project; no changes were made to the original fork or llama-server files.

## Rollback

The original unit is `artifacts/9001-original.service`. No edits have been made to
the original fork, Conda environment, shared launcher, model checkpoints, or
llama-server configuration. Restore the original 9001 unit, reload user systemd,
and start that service if any required runtime acceptance check fails. Preserve
the service name `sglang-openai-9001.service` so its existing VM tunnel relationship
continues to work. Do not touch the 9000 service.
