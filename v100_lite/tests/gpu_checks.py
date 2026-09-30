"""GPU references for the SM70 operators and FP16 MTP recurrence."""

import os
from pathlib import Path
import torch
from sglang_v100_lite.quantization import prepare_nvfp4_moe
from sglang_v100_lite.kernels.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode
from sglang_v100_lite.kernels.sm70_fp8_kv import write_fp8_e5m2_cache_sm70

assert torch.cuda.get_device_capability() == (7, 0)
torch.manual_seed(12)
device = "cuda"
# Integer E5M2 cache conversion must match Torch's software conversion, including
# ties and negative values. Values stay in finite FP16's ordinary model range.
x = torch.randn(17, 1, 256, device=device, dtype=torch.float16) * 4
loc = torch.randperm(64, device=device)[:17].long()
k = torch.zeros((64, 1, 256), device=device, dtype=torch.uint8)
v = torch.zeros_like(k)
assert write_fp8_e5m2_cache_sm70(x, x, k, v, loc)
expected = x.to(torch.float8_e5m2).view(torch.uint8)
assert torch.equal(k[loc], expected) and torch.equal(v[loc], expected)
print("E5M2 paged cache: exact bytes match Torch", flush=True)

# Regression for upstream #41759: int4 padding must not overwrite a view's
# backing guard words when the output length is not divisible by four.
import sgl_kernel

for experts, tokens in ((33, 8), (129, 640)):
    ids = torch.randint(experts - 1, (tokens, 8), device=device, dtype=torch.int32)
    needed = ids.numel() + experts * 63
    for remainder in range(4):
        length = needed + (remainder - needed) % 4
        backing = torch.full((length + 4,), -1, device=device, dtype=torch.int32)
        expert_ids = torch.zeros((length + 63) // 64, device=device, dtype=torch.int32)
        count = torch.empty(1, device=device, dtype=torch.int32)
        cumsum = torch.empty(experts + 1, device=device, dtype=torch.int32)
        torch.ops.sgl_kernel.moe_align_block_size.default(
            ids, experts, 64, backing[:length], expert_ids, count, cumsum, True, False
        )
        torch.cuda.synchronize()
        assert torch.equal(backing[length:], torch.full_like(backing[length:], -1))
        tail = backing[count.item():length]
        assert torch.equal(tail, torch.full_like(tail, ids.numel()))
print("MoE padding: scalar tails filled, backing guard words unchanged", flush=True)

# Qwen TP4 MoE shape. A synthetic checkpoint and a dequantized Torch reference
# test packing, the unusual S0E5M3 scale layout, gating, and weighted summation.
e, h, i = 512, 2560, 160


class Layer(torch.nn.Module):
    params_dtype = torch.float16


layer = Layer()
raw = {}
scales = {}
globals_ = {}
for prefix, n, kk in [("w13", 2 * i, h), ("w2", h, i)]:
    raw[prefix] = torch.randint(
        0, 256, (e, n, kk // 2), device=device, dtype=torch.uint8
    )
    scales[prefix] = torch.full(
        (e, n, kk // 16), 1.0, device=device, dtype=torch.float8_e4m3fn
    )
    globals_[prefix] = torch.full((e,), 0.01, device=device, dtype=torch.float32)
    for suffix, t in [
        ("_weight", raw[prefix]),
        ("_weight_scale", scales[prefix]),
        ("_weight_scale_2", globals_[prefix]),
    ]:
        # Loading postprocessing may update an existing Parameter in place.
        # Keep the checkpoint tensors independent for the reference decoder.
        layer.register_parameter(
            prefix + suffix, torch.nn.Parameter(t.clone(), requires_grad=False)
        )
prepare_nvfp4_moe(None, layer)
codes = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device
)


def dequant(prefix, expert):
    b = raw[prefix][expert]
    unpacked = torch.stack((b & 15, b >> 4), dim=-1).reshape(b.shape[0], -1).long()
    return (
        codes[unpacked]
        * scales[prefix][expert].float().repeat_interleave(16, -1)
        * globals_[prefix][expert]
    )


from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_v100_lite.quantization import marlin_gemm

HookRegistry.register(
    "sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm",
    marlin_gemm,
    HookType.AROUND,
)
HookRegistry.apply_hooks()
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

for m in (1, 4, 17):
    a = torch.randn(m, h, device=device, dtype=torch.float16) * 0.1
    ids = torch.arange(10, device=device, dtype=torch.int32).repeat(m, 1)
    weights = torch.full((m, 10), 0.1, device=device, dtype=torch.float32)
    if m <= 4:
        actual = sm70_nvfp4_moe_decode(
            a,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_weight_scale_2,
            layer.w2_weight_scale_2,
            ids.flatten(),
            weights.flatten(),
        )
    else:
        actual = fused_marlin_moe(
            a,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            torch.zeros(m, e, device=device),
            weights,
            ids,
            global_num_experts=e,
            w1_global_scale=layer.w13_weight_scale_2,
            w2_global_scale=layer.w2_weight_scale_2,
            workspace=layer.workspace,
            num_bits=4,
        )
    ref = torch.zeros_like(actual, dtype=torch.float32)
    for expert in range(10):
        first = (a.float() @ dequant("w13", expert).T).half().float()
        activated = (
            (torch.nn.functional.silu(first[:, :i]) * first[:, i:]).half().float()
        )
        ref += (activated @ dequant("w2", expert).T) * 0.1
    torch.testing.assert_close(actual.float(), ref, rtol=0.03, atol=0.001)
    print(
        f"NVFP4 MoE {'decode' if m <= 4 else 'prefill'} M={m}: dequantized reference agrees",
        flush=True,
    )

import torch
from sglang_v100_lite.tilelang_attention._kernels_dense_d256 import (
    get_dense_prefix_d256_kernel,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as recurrent,
)

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_v100_lite.runtime import round_verify_state

HookRegistry.register(
    "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_kernel.run",
    round_verify_state,
    HookType.AROUND,
)
HookRegistry.apply_hooks()

assert torch.cuda.get_device_capability() == (7, 0)
torch.manual_seed(51)
q = torch.randn(64, 6, 256, device="cuda", dtype=torch.float16) * 0.1
k = torch.randn(64, 1, 256, device="cuda", dtype=torch.float16) * 0.1
v = torch.randn_like(k)
actual = get_dense_prefix_d256_kernel(6, 1)(q, k, v, 0, 256**-0.5)
expected = (
    torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0).repeat(1, 6, 1, 1),
        v.transpose(0, 1).unsqueeze(0).repeat(1, 6, 1, 1),
        is_causal=True,
    )
    .squeeze(0)
    .transpose(0, 1)
)
torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
print("D256 causal dense attention: SDPA reference agrees", flush=True)

# Current mainline uses compressed page 16; retain the fork's page 4 too.
# Random physical page order catches addressing errors hidden by identity maps.
from sglang_v100_lite.tilelang_attention._decode_cuda import (
    sm70_cuda_qsa_indexer_decode,
)

for page in (4, 16):
    queries = torch.randn(4, 4, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(80, page, 1, 128, device="cuda", dtype=torch.float16)
    pages = (
        torch.randperm(80, device="cuda")[: 4 * (64 // page)]
        .reshape(4, 64 // page)
        .int()
    )
    lengths = torch.tensor([1, 13, 31, 64], device="cuda", dtype=torch.int32)
    scores = sm70_cuda_qsa_indexer_decode(queries, keys, pages, lengths, 64, 128**0.5)
    for row, length in enumerate(lengths.tolist()):
        logical = keys[pages[row].long()].reshape(-1, 128)[:length].float()
        ref = (queries[row].float() @ logical.T).relu().sum(0) / (128**0.5)
        torch.testing.assert_close(scores[row, :length], ref, rtol=0.001, atol=0.0001)
    print(
        f"QSA indexer compressed page {page}: permuted-page reference agrees",
        flush=True,
    )

from sglang_v100_lite.tilelang_attention._decode_cuda import (
    sm70_cuda_qsa_prefill,
    sm70_cuda_qsa_decode,
)

# Sparse attention must respect selected logical rows, causal masks and physical
# addressing. Use a permuted cache and padded selections, not contiguous pages.
keys = torch.randn(129, 1, 256, device="cuda", dtype=torch.float16) * 0.1
values = torch.randn_like(keys)
keys = keys.to(torch.float8_e5m2)
values = values.to(torch.float8_e5m2)
req_table = torch.randperm(128, device="cuda").add(1).reshape(1, 128).int()


def selected_reference(queries, selections, lengths):
    outputs = []
    for row, length in enumerate(lengths):
        selected = selections[row].long()
        selected = selected[(selected >= 0) & (selected < length)]
        slots = req_table[0, selected].long()
        kk = keys[slots, 0].float()
        vv = values[slots, 0].float()
        probs = (queries[row].float() @ kk.T / (256**0.5)).softmax(-1)
        outputs.append(probs @ vv)
    return torch.stack(outputs).half()


for mode, rows in [("prefill", 11), ("decode", 4)]:
    queries = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    lengths = list(range(64 - rows + 1, 65))
    selections = torch.randint(0, 48, (rows, 17), device="cuda", dtype=torch.int32)
    selections[:, -1] = -1
    if mode == "prefill":
        actual = sm70_cuda_qsa_prefill(
            queries,
            keys,
            values,
            req_table,
            torch.tensor([0], device="cuda", dtype=torch.int32),
            selections,
            torch.tensor([64], device="cuda", dtype=torch.int32),
            256**-0.5,
        )
    else:
        actual = sm70_cuda_qsa_decode(
            queries,
            keys,
            values,
            req_table,
            torch.zeros(rows, device="cuda", dtype=torch.int32),
            selections,
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
            256**-0.5,
        )
    expected = selected_reference(queries, selections, lengths)
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
    print(f"QSA sparse {mode}: selected-row reference agrees", flush=True)

q = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(1, 4, 8, 128, device="cuda", dtype=torch.float16)
a = torch.randn(4, 8, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
alog = torch.zeros(8, device="cuda", dtype=torch.float32)
bias = torch.zeros(8, device="cuda", dtype=torch.float16)
state = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float16) * 0.01
idx = torch.zeros(1, device="cuda", dtype=torch.int32)
snapshots = torch.empty(1, 4, 8, 128, 128, device="cuda", dtype=torch.float16)
common = dict(
    A_log=alog,
    dt_bias=bias,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    initial_state_indices=idx,
    use_qk_l2norm_in_kernel=True,
)
actual = recurrent(
    a=a,
    b=b,
    q=q,
    k=k,
    v=v,
    initial_state_source=state,
    disable_state_update=True,
    intermediate_states_buffer=snapshots,
    intermediate_state_indices=idx,
    cache_steps=4,
    **common,
)
sequential = state.clone()
outputs = []
for t in range(4):
    outputs.append(
        recurrent(
            a=a[t : t + 1],
            b=b[t : t + 1],
            q=q[:, t : t + 1].contiguous(),
            k=k[:, t : t + 1].contiguous(),
            v=v[:, t : t + 1].contiguous(),
            initial_state_source=sequential,
            **common,
        )
    )
    torch.testing.assert_close(snapshots[:, t], sequential, rtol=0.001, atol=0.0001)
torch.testing.assert_close(actual, torch.cat(outputs, dim=1), rtol=0.001, atol=0.0001)
print("Four-step FP16 MTP recurrence: sequential decode agrees", flush=True)
