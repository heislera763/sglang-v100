"""FP16 sparse MLA parity, including GLM's zero-dimensional RoPE tail."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import triton_sparse_mla_fwd
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    _get_splitk_bufs,
    triton_sparse_mla_decode_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA FP16 kernel coverage",
)


@pytest.mark.parametrize("tail_dim", [0, 64])
@pytest.mark.parametrize(
    "mode,seq,heads,topk",
    [
        ("prefill", 3, 16, 63),
        ("prefill", 3, 16, 2048),
        ("prefill", 129, 8, 63),
        ("prefill", 129, 8, 2051),
        ("decode-fused", 3, 16, 193),
        ("decode-split", 3, 16, 193),
    ],
)
def test_fp16_sparse_mla_matches_reference(mode, seq, heads, topk, tail_dim):
    # FP32 CPU math independently checks the indexed softmax/value contract.
    generator = torch.Generator().manual_seed(731)
    value_dim = 512
    q = torch.randn(seq, heads, value_dim + tail_dim, generator=generator).half()
    kv = torch.randn(4099, value_dim + tail_dim, generator=generator).half()
    indices = torch.stack(
        [torch.randperm(kv.shape[0], generator=generator)[:topk] for _ in range(seq)]
    ).int()
    indices[:, ::7] = -1
    indices[-1] = -1
    scale = (value_dim + tail_dim) ** -0.5
    expected = torch.zeros(seq, heads, value_dim)
    for row in range(seq - 1):
        selected = kv[indices[row][indices[row] >= 0].long()].float()
        probabilities = (q[row].float() @ selected.T * scale).softmax(-1)
        expected[row] = probabilities @ selected[:, :value_dim]

    q_gpu = q.cuda()
    args = (
        q_gpu[:, :, :value_dim],
        q_gpu[:, :, value_dim:],
        kv.cuda().unsqueeze(1),
        indices.cuda().unsqueeze(1),
        scale,
        value_dim,
    )
    if mode == "prefill":
        actual = triton_sparse_mla_fwd(*args)
    else:
        actual = triton_sparse_mla_decode_splitk(
            *args, kv_splits=1 if mode == "decode-fused" else 3
        )
    assert actual.dtype == torch.float16
    assert actual.shape == (1, seq, heads, value_dim)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0].float().cpu(), expected, rtol=0.01, atol=0.001)
    assert torch.count_nonzero(actual[0, -1]) == 0
    print(
        mode,
        seq,
        heads,
        topk,
        tail_dim,
        "max_abs_error",
        (actual[0].float().cpu() - expected).abs().max().item(),
    )


@pytest.mark.parametrize("kv_splits", [None, 1, 3])
def test_glm_decode_graph_replays_sparse_and_empty_indices(kv_splits):
    """Tile changes must preserve padded/strided queries and live graph indices."""
    generator = torch.Generator().manual_seed(735)
    heads, dim, topk = 8, 512, 2051
    # A larger packed row leaves noncontiguous head strides, as model views do.
    packed = torch.randn(1, heads, dim + 64, generator=generator).half()
    kv = torch.randn(4099, dim, generator=generator).half()
    q = packed.cuda()
    cache = kv.cuda().unsqueeze(1)
    indices = torch.full((1, 1, topk), -1, dtype=torch.int32, device="cuda")
    scale = dim**-0.5
    workspace = []

    def run():
        return triton_sparse_mla_decode_splitk(
            q[:, :, :dim],
            q[:, :, dim:dim],
            cache,
            indices,
            scale,
            dim,
            kv_splits=kv_splits,
            workspace=workspace,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for count in (0, 1, 65, 128, 2048, 2051, 0):
            ids = torch.full((topk,), -1, dtype=torch.int32)
            positions = torch.randperm(topk, generator=generator)[:count]
            ids[positions] = torch.randperm(kv.shape[0], generator=generator)[
                :count
            ].int()
            indices.copy_(ids.reshape(1, 1, topk))
            graph.replay()
            stream.synchronize()
            expected = torch.zeros(heads, dim)
            if count:
                selected = kv[ids[ids >= 0].long()].float()
                probabilities = (
                    packed[0, :, :dim].float() @ selected.T * scale
                ).softmax(-1)
                expected = probabilities @ selected
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(
                actual[0, 0].float().cpu(), expected, rtol=0.01, atol=0.001
            )
            if count == 0:
                assert torch.count_nonzero(actual) == 0


def test_splitk_workspace_preserves_dtype_and_old_allocations():
    workspace = []
    args = (3, 3, 16, 512, torch.device("cuda:0"), workspace)
    _, bf16 = _get_splitk_bufs(*args)
    _, fp16 = _get_splitk_bufs(*args, dtype=torch.float16)
    assert bf16.dtype == torch.bfloat16
    assert fp16.dtype == torch.float16
    assert bf16.data_ptr() != fp16.data_ptr()
    assert len(workspace) == 2
    _, reused = _get_splitk_bufs(*args, dtype=torch.float16)
    assert reused.data_ptr() == fp16.data_ptr()
    assert len(workspace) == 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
