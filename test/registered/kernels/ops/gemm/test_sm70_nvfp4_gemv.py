"""Guard packed-weight/scales decoding and dynamic routing during graph replay.

The repacker comes from the independent installed SM70 Marlin extension;
the reference multiplies checkpoint-native FP4 values and E4M3 scales on CPU.
Changing nibble ordering, exponent compensation, or cached routing breaks parity.
"""

import os

import pytest
import torch

from sglang.kernels.ops.gemm.sm70_nvfp4_gemv import sm70_nvfp4_gemv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 Marlin encoded-layout coverage",
)


@pytest.mark.parametrize(
    "m,topk,n,k",
    [
        (1, 1, 64, 256),
        (1, 1, 128, 256),
        (1, 1, 512, 4096),
        (1, 1, 3072, 4096),
        (1, 1, 6144, 4096),
        (1, 1, 4096, 1024),
        (1, 1, 4096, 256),
        (1, 1, 4096, 512),
        (1, 8, 512, 4096),
        (1, 8, 1024, 4096),
        (8, 1, 4096, 256),
        (8, 1, 4096, 512),
    ],
)
def test_checkpoint_reference_and_graph_replay(m, topk, n, k):
    from sglang_v100_lite.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    assert os.environ.get("SGLANG_V100_MARLIN_DIR"), "Install the V100 Marlin extension"
    torch.manual_seed(531)
    routes = m * topk
    experts = 1 if routes == 1 else 9
    packed = torch.randint(
        0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda"
    )
    scales = torch.rand(experts, n, k // 16, device="cuda").to(torch.float8_e4m3fn)
    # Include zero block scales and non-power-of-two global scales.
    scales.view(torch.uint8)[:, :, ::7] = 0
    global_native = torch.linspace(0.0625, 0.123456, experts, device="cuda")
    encoded, factor = sm70_nvfp4_marlin_process_scales(
        scales.transpose(1, 2).contiguous(), torch.float16
    )
    global_scale = (
        sm70_nvfp4_marlin_process_global_scale(global_native, torch.float16) / factor
    )
    weight = _dense_repack(packed)
    codes = torch.stack((packed.cpu().int() & 15, packed.cpu().int() >> 4), -1).reshape(
        experts, n, k
    )
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    reference_weight = (
        lookup[codes]
        * scales.cpu().float().repeat_interleave(16, -1)
        * global_native.cpu()[:, None, None]
    )
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    out = torch.full((routes, n), float("nan"), dtype=torch.float16, device="cuda")
    partials = torch.empty((routes, k // 128, n), dtype=torch.float32, device="cuda")
    block = 8
    sorted_ids = torch.full((routes, block), routes, dtype=torch.int32, device="cuda")
    sorted_ids[:, 0] = torch.arange(routes, device="cuda", dtype=torch.int32)
    sorted_ids = sorted_ids.flatten()
    expert_ids = torch.arange(routes, device="cuda", dtype=torch.int32)
    router = torch.linspace(0.125, 1, routes, device="cuda")
    mul_router = m > 1

    def run():
        return sm70_nvfp4_gemv(
            a,
            out,
            weight,
            encoded,
            global_scale,
            sorted_ids,
            expert_ids,
            router,
            block,
            topk,
            mul_router,
            partials,
            block_k=128,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        selected = torch.randperm(experts)[:routes]
        row_ids = torch.randperm(routes)
        expert_ids.copy_(selected)
        sorted_ids.view(routes, block)[:, 0].copy_(row_ids)
        a.copy_(torch.randn_like(a))
        router.copy_(torch.rand_like(router))
        graph.replay()
        torch.cuda.synchronize()
        expected = torch.empty(routes, n)
        cpu_a = a.cpu().float()
        cpu_router = router.cpu()
        for index, row in enumerate(row_ids.tolist()):
            expected[row] = cpu_a[row // topk] @ reference_weight[selected[index]].T
        if mul_router:
            expected *= cpu_router[:, None]
        torch.testing.assert_close(out.cpu().float(), expected, rtol=0.01, atol=0.015)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
