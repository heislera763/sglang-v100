"""Software E4M3 boundaries and pool4 rotation parity without FP8 instructions."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa.sm70_indexer import (
    fp8_quantize_sm70,
    kpool_compress_sm70,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="jit-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA software FP8 kernels",
)


def reference_quantize(x, round_scale):
    x = x.float()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("round_scale", [False, True])
def test_quantize_rounding_boundaries_and_strides(round_scale):
    # Independent Torch conversion covers every finite FP8 midpoint, ties to
    # even, subnormals, signed zero, and the 448 endpoint. Fixed maxima make
    # the unrounded scale exactly one; near-midpoint values guard off-by-one
    # encodings that random numerical tolerance tests would miss.
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mids = (values[:-1] + values[1:]) / 2
    delta = torch.tensor([-1e-5, 0, 1e-5])
    boundary = (mids[:, None] + delta).flatten()
    boundary = torch.cat((boundary, -boundary, torch.tensor([0.0, -0.0])))
    rows = torch.zeros((boundary.numel(), 256), dtype=torch.float32)
    rows[:, 0] = boundary
    rows[:, 2] = 448
    x = rows.cuda()[:, ::2]
    expected, scale = reference_quantize(rows[:, ::2], round_scale)
    actual, actual_scale = fp8_quantize_sm70(x, round_scale)
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected.view(torch.uint8))

    # Real query layout and data-dependent (usually non-power-of-two) scales.
    query = torch.randn(3, 32, 128, generator=torch.Generator().manual_seed(736)).half()
    query[0, 0] = 0
    query = query.cuda()
    # Use the prior CUDA Torch arithmetic for data-dependent scales: CPU and
    # GPU division can differ at an exact FP8 midpoint even before this change.
    expected, scale = reference_quantize(query, round_scale)
    actual, actual_scale = fp8_quantize_sm70(query, round_scale)
    torch.testing.assert_close(actual_scale, scale, rtol=1e-6, atol=1e-8)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("round_scale", [False, True])
@pytest.mark.parametrize("rows", [1, 65])
def test_pool4_matches_dense_walsh_reference(rows, round_scale):
    # A dense +/-1 Walsh matrix checks the butterfly orientation independently.
    # Noncontiguous score/key/APE inputs defend runtime slice semantics.
    generator = torch.Generator().manual_seed(735)
    key_storage = torch.randn(rows, 4, 256, generator=generator).bfloat16()
    score_storage = torch.randn(rows, 4, 256, generator=generator)
    ape_storage = torch.randn(4, 256, generator=generator)
    keys, scores, ape = (
        key_storage[:, :, ::2],
        score_storage[:, :, ::2],
        ape_storage[:, ::2],
    )
    walsh = torch.tensor(
        [[(-1) ** ((r & c).bit_count() % 2) for c in range(128)] for r in range(128)],
        dtype=torch.float32,
    )
    pooled = ((scores + ape).softmax(1) * keys.float()).sum(1).bfloat16().float()
    rotated = ((pooled @ walsh) * 128**-0.5).bfloat16().float()
    expected, scale = reference_quantize(rotated, round_scale)
    actual, actual_scale = kpool_compress_sm70(
        key_storage.cuda()[:, :, ::2],
        score_storage.cuda()[:, :, ::2],
        ape_storage.cuda()[:, ::2],
        round_scale,
    )
    torch.testing.assert_close(actual_scale.cpu(), scale, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(actual.float().cpu(), expected.float(), rtol=0, atol=0)


def test_quantization_and_pool_graph_replay_use_new_inputs():
    # A captured primitive must read updated input storage, not freeze results.
    q = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(1, 4, 128, device="cuda").bfloat16()
    scores = torch.randn(1, 4, 128, device="cuda")
    ape = torch.randn(4, 128, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fp8_quantize_sm70(q)
            kpool_compress_sm70(keys, scores, ape)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q8, qs = fp8_quantize_sm70(q)
        k8, ks = kpool_compress_sm70(keys, scores, ape)
    for _ in range(2):
        q.copy_(torch.randn_like(q))
        keys.copy_(torch.randn_like(keys))
        scores.copy_(torch.randn_like(scores))
        graph.replay()
        expected_q, expected_qs = fp8_quantize_sm70(q)
        expected_k, expected_ks = kpool_compress_sm70(keys, scores, ape)
        torch.testing.assert_close(qs, expected_qs, rtol=0, atol=0)
        torch.testing.assert_close(ks, expected_ks, rtol=0, atol=0)
        assert torch.equal(q8.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(k8.view(torch.uint8), expected_k.view(torch.uint8))


@pytest.mark.parametrize("use_graph", [False, True])
def test_device_pool4_replays_lengths_requests_and_page_boundaries(use_graph):
    from sglang.kernels.ops.attention.dsa.sm70_pool4_decode import pool4_decode_sm70

    torch.manual_seed(737)
    cache = torch.zeros(40, 8448, device="cuda", dtype=torch.uint8)
    finite_codes = (
        torch.cat((torch.arange(127), torch.arange(128, 255))).to(torch.uint8).cuda()
    )
    cache[:, :8192] = finite_codes.repeat((40 * 8192 + 253) // 254)[
        : 40 * 8192
    ].reshape(40, 8192)
    cache[:, 8192:].view(torch.float32).fill_(0.01)
    tail_k = torch.randn(3, 4, 128, device="cuda").bfloat16()
    tail_g = torch.randn_like(tail_k)
    table = torch.zeros(3, 512, device="cuda", dtype=torch.int32)
    for req, start in ((1, 1), (2, 17)):
        pages = torch.randperm(8, device="cuda") + start
        table[req] = (pages[:, None] * 64 + torch.arange(64, device="cuda")).flatten()
    reference_cache, reference_k, reference_g = (
        cache.clone(),
        tail_k.clone(),
        tail_g.clone(),
    )
    keys = torch.zeros(3, 128, device="cuda", dtype=torch.float16)
    gates = torch.zeros_like(keys)
    ape = torch.randn(4, 128, device="cuda")
    reqs = torch.tensor([1, 2, -1], device="cuda", dtype=torch.int64)
    lengths = torch.zeros(3, device="cuda", dtype=torch.int32)

    def run():
        return pool4_decode_sm70(
            keys, gates, ape, tail_k, tail_g, cache, table, reqs, lengths
        )

    for _ in range(3):
        run()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_keys, actual_scales, actual_pools = run()

    for step in range(12):
        order = [1, 2, -1] if step % 2 == 0 else [2, 1, -1]
        seqs = [
            step + 1 if req == 1 else step + 255 if req == 2 else 0 for req in order
        ]
        keys.normal_()
        gates.normal_()
        reqs.copy_(torch.tensor(order, device="cuda"))
        lengths.copy_(torch.tensor(seqs, device="cuda", dtype=torch.int32))
        if use_graph:
            graph.replay()
        else:
            actual_keys, actual_scales, actual_pools = run()
        for row, (req, length) in enumerate(zip(order, seqs)):
            if req < 0:
                assert actual_pools[row] == 0
                assert torch.count_nonzero(actual_keys[row]) == 0
                assert torch.count_nonzero(actual_scales[row]) == 0
                continue
            prefix = length - 1
            old_tail = prefix % 4
            pos = torch.arange(prefix - old_tail, prefix, device="cuda") % 4
            if length % 4 == 0:
                assembled_k = torch.cat(
                    (reference_k[req, pos], keys[row : row + 1].bfloat16())
                )
                assembled_g = torch.cat(
                    (reference_g[req, pos].float(), gates[row : row + 1].float())
                )
                q, scale = kpool_compress_sm70(
                    assembled_k[None], assembled_g[None], ape
                )
                pool_id = prefix // 4
                page = int(table[req, pool_id // 64 * 256].item()) // 64
                offset = pool_id % 64
                reference_cache[page, offset * 128 : (offset + 1) * 128] = q.view(
                    torch.uint8
                )[0]
                reference_cache[page, 8192 + offset * 4 : 8192 + (offset + 1) * 4] = (
                    scale.view(torch.uint8).flatten()
                )
            else:
                reference_k[req, prefix % 4] = keys[row].bfloat16()
                reference_g[req, prefix % 4] = gates[row].bfloat16()
            count = length // 4
            ids = torch.arange(count, device="cuda")
            pages = table[req, ids // 64 * 256].long() // 64
            columns = (
                ids[:, None] % 64 * 128 + torch.arange(128, device="cuda")[None, :]
            )
            expected_keys = (
                reference_cache[pages[:, None], columns]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .half()
            )
            expected_scales = reference_cache[:, 8192:].view(torch.float32)[
                pages, ids % 64
            ]
            assert actual_pools[row] == count
            torch.testing.assert_close(
                actual_keys[row, :count], expected_keys, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_scales[row, :count], expected_scales, rtol=0, atol=0
            )
            assert torch.count_nonzero(actual_keys[row, count:]) == 0
            assert torch.count_nonzero(actual_scales[row, count:]) == 0
        assert torch.equal(cache, reference_cache)
        assert torch.equal(tail_k, reference_k)
        assert torch.equal(tail_g, reference_g)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
