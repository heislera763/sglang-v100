"""Checkpoint-native MoE reference, clamping and live graph routing on SM70."""

import os

import pytest
import torch

from sglang.kernels.ops.moe.sm70_nvfp4 import sm70_nvfp4_moe
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta encoded NVFP4 experts",
)


@pytest.mark.parametrize("rows,width", [(1, 256), (2, 512), (3, 256), (4, 512)])
def test_checkpoint_reference_and_fresh_graph_routes(rows, width):
    from sglang_v100_plus.quantization import (
        _dense_repack,
        sm70_nvfp4_marlin_process_global_scale,
        sm70_nvfp4_marlin_process_scales,
    )

    assert os.environ.get("SGLANG_V100_MARLIN_DIR"), (
        "Install the Volta Marlin extension"
    )
    torch.manual_seed(831)
    experts, hidden = 9, 4096
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )

    def bank(n, k):
        packed = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        )
        scales = (torch.rand(experts, n, k // 16, device="cuda") * 0.04).to(
            torch.float8_e4m3fn
        )
        scales.view(torch.uint8)[:, :, ::7] = 0
        globals_native = torch.linspace(0.25, 1.123456, experts, device="cuda")
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        globals_encoded = (
            sm70_nvfp4_marlin_process_global_scale(globals_native, torch.float16)
            / factor
        )
        codes = torch.stack(
            (packed.cpu().int() & 15, packed.cpu().int() >> 4), -1
        ).reshape(experts, n, k)
        # Independent of the encoded layout's word/nibble addressing.
        reference = (
            lookup[codes].double()
            * scales.cpu().double().repeat_interleave(16, -1)
            * globals_native.cpu().double()[:, None, None]
        )
        return _dense_repack(packed), encoded, globals_encoded, reference

    w13, s13, g13, reference13 = bank(2 * width, hidden)
    w2, s2, g2, reference2 = bank(hidden, width)
    a = torch.randn(rows, hidden, device="cuda", dtype=torch.float16) * 4
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
    ).int()
    router = torch.rand(rows, 8, device="cuda")
    router.div_(router.sum(-1, keepdim=True))

    def run():
        return sm70_nvfp4_moe(a, ids, router, w13, w2, s13, s2, g13, g2, 10.0, 2.5)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        a.normal_(std=4)
        ids.copy_(
            torch.stack(
                [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
            )
        )
        router.copy_(torch.rand_like(router))
        router.div_(router.sum(-1, keepdim=True))
        before = (a.clone(), ids.clone(), router.clone())
        graph.replay()
        eager = run()
        expected = torch.empty(rows, hidden, dtype=torch.float64)
        clamped = False
        cpu_a, cpu_ids, cpu_router = a.cpu().double(), ids.cpu(), router.cpu().double()
        for row in range(rows):
            parts = []
            for route in range(8):
                expert = cpu_ids[row, route]
                up = (cpu_a[row] @ reference13[expert].T).half()
                clamped |= bool((up.abs() > 10).any())
                gate, linear = up.chunk(2)
                # Keep both independent Half rounding boundaries.
                silu = torch.nn.functional.silu(gate.clamp(max=10).float()).half()
                activation = (silu.float() * linear.clamp(-10, 10).float()).half()
                down = (
                    activation.double() @ reference2[expert].T * cpu_router[row, route]
                ).half()
                parts.append(down.double())
            expected[row] = torch.stack(parts).sum(0) * 2.5
        assert clamped
        for result in (actual, eager):
            torch.testing.assert_close(
                result.cpu().double(), expected, rtol=0.01, atol=0.015
            )
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))
        for live, saved in zip((a, ids, router), before):
            assert torch.equal(live, saved)
