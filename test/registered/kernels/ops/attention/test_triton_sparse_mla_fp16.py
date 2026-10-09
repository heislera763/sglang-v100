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


def test_empty_tile_prefill_replays_without_changing_valid_tile_math():
    """Skipping padding must retain holes, pool4 tails and graph input refresh."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta prefill padding dispatch")
    torch.manual_seed(738)
    packed = torch.randn(129, 8, 576, device="cuda", dtype=torch.float16)
    q, rope = packed[:, :, :512], packed[:, :, 512:512]
    kv = torch.randn(4099, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.full((129, 1, 2051), -1, device="cuda", dtype=torch.int32)
    args = (q, rope, kv, indices, 256**-0.5, 512)
    for _ in range(3):
        triton_sparse_mla_fwd(*args, skip_empty_tiles=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = triton_sparse_mla_fwd(*args, skip_empty_tiles=True)
    for count in (0, 1, 33, 128, 2048, 2051, 0):
        indices.fill_(-1)
        if count:
            indices[:, :, :count] = torch.arange(count, device="cuda")
            if count < 2048:
                indices[:, :, 16:32] = -1  # Empty interior tile/partial masks.
                indices[:, :, 2048:] = torch.tensor([7, -1, 31], device="cuda")
        packed.normal_()
        kv.normal_()
        reference = triton_sparse_mla_fwd(*args)
        graph.replay()
        # The independent softmax fixtures above establish the arithmetic;
        # this pins its reduction/rounding order across the padding branch.
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("heads", [8, 16])
def test_sm70_query_union_preserves_support_ragged_tail_and_graph_replay(
    heads, monkeypatch
):
    """Disjoint/empty neighbours must not receive each other's KV values."""
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("Volta query union")
    from sglang_v100_plus import glm_dsa

    from sglang.kernels.ops.attention.dsa import sm70_sparse_prefill
    from sglang.kernels.ops.attention.dsa.sm70_sparse_prefill import (
        sparse_mla_prefill_sm70,
    )

    # Installed hooks also wrap the ordinary API used by an odd last query.
    # Exercise its optional d_v argument rather than bypassing that boundary.
    monkeypatch.setattr(glm_dsa, "get_attn_backend", lambda: None)
    monkeypatch.setattr(
        sm70_sparse_prefill,
        "triton_sparse_mla_fwd",
        lambda *args, **kwargs: glm_dsa.sparse_prefill(
            triton_sparse_mla_fwd, *args, **kwargs
        ),
    )

    torch.manual_seed(740)
    # Head-strided input, odd query count and non-power-of-two pool4 width.
    packed = torch.randn(33, heads, 576, device="cuda", dtype=torch.float16)
    q = packed[:, :, :512]
    kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.float16)
    indices = torch.full((33, 1, 67), -1, device="cuda", dtype=torch.int32)
    scale = 256**-0.5
    for _ in range(3):
        sparse_mla_prefill_sm70(q, kv, indices, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sparse_mla_prefill_sm70(q, kv, indices, scale)
    for update in range(3):
        packed.normal_()
        kv.normal_()
        indices.fill_(-1)
        for row in range(33):
            if (row + update) % 5 == 0:
                continue
            valid = torch.randperm(257, device="cuda")[: min(67, row * 3 + 1)]
            columns = torch.randperm(67, device="cuda")[: valid.numel()]
            indices[row, 0, columns] = valid.int()
        # A populated first query with an entirely empty second query guards
        # the all-masked initial tile: a finite sentinel would leak KV values.
        indices[0].fill_(-1)
        indices[0, 0, :3] = torch.tensor([5, 17, 31], device="cuda")
        indices[1].fill_(-1)
        before = indices.clone()
        graph.replay()
        eager = sparse_mla_prefill_sm70(q, kv, indices, scale)
        expected = torch.zeros_like(q, dtype=torch.float64)
        for row in range(33):
            ids = indices[row, 0]
            selected = kv[ids[ids >= 0].long(), 0].double()
            if selected.numel():
                expected[row] = (q[row].double() @ selected.T * scale).softmax(
                    -1
                ) @ selected
        for result in (actual, eager):
            assert torch.isfinite(result).all()
            torch.testing.assert_close(
                result[0].double(), expected, rtol=0.01, atol=0.001
            )
            assert torch.count_nonzero(result[0, 1]) == 0
        assert torch.equal(indices, before)
        assert torch.equal(actual.view(torch.uint8), eager.view(torch.uint8))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
