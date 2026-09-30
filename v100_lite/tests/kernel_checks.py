"""GPU correctness checks before loading the target model. Run on GPU 0 only."""

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
from sglang_v100_lite.native_api import moe_align_block_size, moe_sum_reduce

HookRegistry.register(
    "sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm",
    marlin_gemm,
    HookType.AROUND,
)
HookRegistry.register(
    "sgl_kernel.moe.moe_align_block_size", moe_align_block_size, HookType.AROUND
)
HookRegistry.register("sgl_kernel.moe.moe_sum_reduce", moe_sum_reduce, HookType.AROUND)
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
