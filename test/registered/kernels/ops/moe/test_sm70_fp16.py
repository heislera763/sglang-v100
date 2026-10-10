"""SM70 expert GEMM routing ownership, precision boundaries and graph refresh."""

import pytest
import torch

from sglang.kernels.ops.moe.sm70_fp16 import sm70_fp16_moe_gemm
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Volta FP16 routed expert MMA",
)


@pytest.mark.parametrize(
    "rows,block_size,n,k,top_k,mul",
    [
        (1, 4, 1024, 4096, 8, False),
        (4, 16, 1024, 4096, 8, False),
        (3, 16, 512, 4096, 8, False),
        (4, 16, 4096, 512, 1, True),
        (2, 8, 4096, 256, 1, True),
        (129, 64, 1024, 4096, 8, False),
        (128, 64, 4096, 512, 1, True),
        (33, 64, 144, 80, 8, True),
        (129, 128, 144, 80, 1, False),
    ],
)
def test_routed_gemm_independent_reference_and_live_graph(
    rows, block_size, n, k, top_k, mul
):
    """Distinct route positions must stay distinct even when they share input."""
    torch.manual_seed(853)
    experts = 9
    routes = rows * 8
    a = torch.randn(routes // top_k, k, device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn(experts, n, k, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.stack(
        [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
    ).int()
    weights = torch.softmax(torch.randn(rows, 8, device="cuda"), dim=-1)
    sorted_ids, expert_ids, count = moe_align_block_size(ids, block_size, experts)
    output = torch.full((rows, 8, n), float("nan"), device="cuda", dtype=torch.float16)

    def run():
        sm70_fp16_moe_gemm(
            a,
            b,
            output,
            sorted_ids,
            expert_ids,
            count,
            weights,
            block_size,
            top_k,
            mul,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for update in range(3):
        if update == 1:
            ids.fill_(0)  # Collapse many routes into one expert/padded block.
        else:
            ids.copy_(
                torch.stack(
                    [torch.randperm(experts, device="cuda")[:8] for _ in range(rows)]
                ).int()
            )
        a.normal_().mul_(0.1)
        b.normal_().mul_(0.1)
        weights.copy_(torch.softmax(torch.randn_like(weights), dim=-1))
        aligned = moe_align_block_size(ids, block_size, experts)
        sorted_ids.copy_(aligned[0])
        expert_ids.copy_(aligned[1])
        count.copy_(aligned[2])
        # Establish the scatter ownership premise independently of the GEMM.
        live = sorted_ids[: int(count.item())]
        actual_routes = live[live < routes].sort().values
        assert torch.equal(
            actual_routes, torch.arange(routes, device="cuda", dtype=torch.int32)
        )
        output.fill_(float("nan"))
        graph.replay()
        expected = torch.zeros(routes, n, device="cuda", dtype=torch.float64)
        flat_ids = ids.reshape(-1)
        for expert in range(experts):
            selected = (flat_ids == expert).nonzero().flatten()
            if selected.numel():
                expected[selected] = (
                    a[selected // top_k].double() @ b[expert].double().T
                )
        if mul:
            expected *= weights.reshape(-1, 1).double()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(
            output.reshape(routes, n).double(), expected, rtol=0.002, atol=0.0005
        )


@pytest.mark.parametrize("mul", [False, True])
def test_routed_gemm_filtered_experts_write_zero(mul):
    """Filtered routes must overwrite prior nonzero outputs, not leave stale rows."""
    torch.manual_seed(854)
    a = torch.randn(5, 80, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 144, 80, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0], [-1], [1], [-1], [0]], device="cuda", dtype=torch.int32)
    weights = torch.ones(5, 1, device="cuda")
    weights[1] = float("nan")
    weights[3] = float("inf")
    aligned = moe_align_block_size(ids, 16, 2)
    output = torch.full((5, 144), 11.0, device="cuda", dtype=torch.float16)
    sm70_fp16_moe_gemm(a, b, output, *aligned, weights, 16, 1, mul)
    expected = torch.zeros(5, 144, device="cuda", dtype=torch.float64)
    for row, expert in enumerate(ids.flatten().tolist()):
        if expert >= 0:
            expected[row] = b[expert].double() @ a[row].double()
    torch.testing.assert_close(output.double(), expected, rtol=0.002, atol=0.002)
    assert torch.count_nonzero(output[[1, 3]]) == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
