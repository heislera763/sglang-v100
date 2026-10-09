"""Small-row NVFP4 experts using the FP32-accumulating Volta GEMV operator."""

import torch

from sglang.kernels.ops.gemm.sm70_nvfp4_gemv import sm70_nvfp4_gemv


def sm70_nvfp4_moe(
    a, ids, router, w13, w2, s13, s2, g13, g2, clamp_limit, routed_scaling_factor
):
    """FP16 [1..4,4096], gated width256/512, eight routes per row.

    Weights/scales use the encoded Marlin layout, not checkpoint-native FP8
    scales. Route IDs are valid, int32 and distinct within each input row;
    different rows may select the same expert. Router rows are normalized.
    Per-call scratch has no global
    mutable owners and remains private during graph capture/replay.
    Preserve FP16 FC1 output, SiLU output and product rounding, FC2 routed
    output and the ordinary FP16 expert-sum boundary.
    """
    if a.ndim != 2 or w2.ndim != 3 or w13.ndim != 3:
        raise ValueError("SM70 NVFP4 MoE requires matrix input and expert banks")
    rows, hidden = a.shape
    experts, groups, _ = w2.shape
    intermediate = groups * 16
    if not (
        a.is_cuda
        and a.dtype == torch.float16
        and 1 <= rows <= 4
        and hidden == 4096
        and intermediate in (256, 512)
        and ids.shape == router.shape == (rows, 8)
        and w13.shape == (experts, hidden // 16, intermediate * 4)
        and w2.shape == (experts, intermediate // 16, hidden * 2)
        and ids.dtype == torch.int32
        and router.dtype == torch.float32
        and clamp_limit == 10.0
        and routed_scaling_factor == 2.5
    ):
        raise ValueError("Unsupported SM70 NVFP4 expert geometry or activation")
    tensors = (a, ids, router, w13, w2, s13, s2, g13, g2)
    if any(t.device != a.device or not t.is_contiguous() for t in tensors):
        raise ValueError("SM70 NVFP4 experts require contiguous colocated tensors")
    routes = rows * 8
    order = torch.arange(routes, device=a.device, dtype=torch.int32)
    ids, router = ids.reshape(-1), router.reshape(-1)
    up = torch.empty((routes, intermediate * 2), device=a.device, dtype=a.dtype)
    sm70_nvfp4_gemv(a, up, w13, s13, g13, order, ids, router, 1, 8, False)
    gate, linear = up.chunk(2, dim=-1)
    # Deliberately retain both Half rounding boundaries in upstream's SM70
    # swiglu_limit_func. Fusing SiLU*up in FP32 would compute another function.
    activated = torch.nn.functional.silu(gate.clamp(max=clamp_limit)) * linear.clamp(
        min=-clamp_limit, max=clamp_limit
    )
    down = torch.empty((routes, hidden), device=a.device, dtype=a.dtype)
    sm70_nvfp4_gemv(activated, down, w2, s2, g2, order, ids, router, 1, 1, True)
    from sgl_kernel import moe_sum_reduce

    output = torch.empty_like(a)
    moe_sum_reduce(down.reshape(rows, 8, hidden), output, routed_scaling_factor)
    return output
