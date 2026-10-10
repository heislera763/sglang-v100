"""Volta Tensor Core split-K sparse MLA for FP16 decode and verification.

Preserve selected physical KV rows, FP16 probabilities/normalized partials and
FP32 softmax/accumulation. Each CTA owns one query and a contiguous index range;
the existing log-space reduction combines bounded partials. MMA changes rounding.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _decode_kernel(heads, topk, splits, stride_t, stride_h, use_lengths):
    bm, bn, dim = 16, 64, 512
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")
    nl = T.dynamic("nl")
    tiles = (topk + splits * bn - 1) // (splits * bn)

    @T.prim_func
    def sparse_mla_split_sm70(
        Q: T.StridedTensor([nt, heads, dim], [stride_t, stride_h, 1], T.float16),
        KV: T.Tensor([nk, dim], T.float16),
        Ids: T.Tensor([nt, topk], T.int32),
        Lengths: T.Tensor([nl], T.int32),
        LSE: T.Tensor([nt, splits, bm], T.float32),
        Partial: T.Tensor([nt, splits, bm, dim], T.float16),
        scale: T.float32,
    ):
        with T.Kernel(nt, splits, threads=128) as (row, part):
            qs = T.alloc_shared([bm, dim], T.float16)
            kv = T.alloc_shared([bn, dim], T.float16)
            ps = T.alloc_shared([bm, bn], T.float16)
            ids = T.alloc_shared([bn], T.int32)
            scores = T.alloc_fragment([bm, bn], T.float32)
            o = T.alloc_fragment([bm, dim], T.float32)
            m = T.alloc_fragment([bm], T.float32)
            old = T.alloc_fragment([bm], T.float32)
            l = T.alloc_fragment([bm], T.float32)
            z = T.alloc_fragment([bm], T.float32)
            alpha = T.alloc_fragment([bm], T.float32)
            for h, d in T.Parallel(bm, dim):
                qs[h, d] = T.if_then_else(h < heads, Q[row, h, d], 0)
            T.clear(o)
            T.fill(m, -1.0e30)
            T.clear(l)
            limit = T.min(topk, T.max(0, Lengths[row])) if use_lengths else topk
            for tile in T.serial(tiles):
                start = (part * tiles + tile) * bn
                if start < limit:
                    for n in T.Parallel(bn):
                        ids[n] = T.if_then_else(
                            start + n < limit, Ids[row, start + n], -1
                        )
                    for n, d in T.Parallel(bn, dim):
                        kv[n, d] = T.if_then_else(
                            (ids[n] >= 0) & (ids[n] < nk),
                            KV[T.cast(ids[n], T.int64), d],
                            0,
                        )
                    T.clear(scores)
                    T.gemm(
                        qs,
                        kv,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for h, n in T.Parallel(bm, bn):
                        scores[h, n] = T.if_then_else(
                            (ids[n] >= 0) & (ids[n] < nk),
                            scores[h, n] * scale * 1.4426950408889634,
                            -1.0e30,
                        )
                    T.copy(m, old)
                    T.reduce_max(scores, m, dim=1, clear=False)
                    for h in T.Parallel(bm):
                        alpha[h] = T.exp2(old[h] - m[h])
                        l[h] *= alpha[h]
                    for h, d in T.Parallel(bm, dim):
                        o[h, d] *= alpha[h]
                    for h, n in T.Parallel(bm, bn):
                        scores[h, n] = T.if_then_else(
                            (ids[n] >= 0) & (ids[n] < nk),
                            T.exp2(scores[h, n] - m[h]),
                            0,
                        )
                    T.reduce_sum(scores, z, dim=1)
                    for h in T.Parallel(bm):
                        l[h] += z[h]
                    T.copy(scores, ps)
                    T.gemm(ps, kv, o, policy=T.GemmWarpPolicy.FullCol)
            for h in T.Parallel(bm):
                LSE[row, part, h] = T.if_then_else(
                    l[h] > 0, T.log2(T.max(l[h], 1.0e-30)) + m[h], -1073741824.0
                )
            for h, d in T.Parallel(bm, dim):
                Partial[row, part, h, d] = T.cast(
                    o[h, d] / T.max(l[h], 1.0e-30), T.float16
                )

    return sparse_mla_split_sm70


def sparse_mla_decode_sm70(
    q, kv, indices, sm_scale, kv_splits=None, workspace=None, topk_length=None
):
    """Return [1, rows1..4, heads8/16, 512] from unquantized FP16 latent KV.

    Queries may be head/row-strided; Q/KV base pointers require 16-byte alignment
    for vectorized loads. Negative/out-of-pool indices are padding;
    optional lengths bound the last index column, without a host scan. Explicit
    splits retain contiguous tile groups; inactive groups need no allocation.
    Workspace follows the backend-owned grow-only LSE/FP16 partial contract.
    With no workspace, scratch is call-owned (including CUDA graph captures).
    """
    if not (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and q.ndim == 3
        and 1 <= q.shape[0] <= 4
        and q.shape[1] in (8, 16)
        and q.shape[2] == 512
        and q.dtype == kv.dtype == torch.float16
        and q.stride(-1) == 1
        and q.data_ptr() % 16 == kv.data_ptr() % 16 == 0
        and kv.device == indices.device == q.device
        and kv.is_contiguous()
        and kv.shape[-1] == 512
        and (kv.ndim == 2 or (kv.ndim == 3 and kv.shape[1] == 1))
        and indices.dtype == torch.int32
        and indices.shape[0] == q.shape[0]
        and (indices.ndim == 2 or (indices.ndim == 3 and indices.shape[1] == 1))
        and indices.shape[-1] > 0
        and (kv_splits is None or kv_splits > 0)
    ):
        raise ValueError(
            "SM70 sparse decode requires FP16 rows1..4, heads8/16, latent512 and CUDA int32 indices"
        )
    indices = indices.reshape(q.shape[0], indices.shape[-1]).contiguous()
    if topk_length is not None and not (
        topk_length.device == q.device
        and topk_length.dtype == torch.int32
        and topk_length.numel() == q.shape[0]
        and topk_length.is_contiguous()
    ):
        raise ValueError("SM70 sparse decode lengths must be one CUDA int32 per query")
    rows, heads, dim = q.shape
    topk = indices.shape[-1]
    tiles = (topk + 63) // 64
    if kv_splits is None:
        # One CTA's ~83KiB shared storage allows one resident CTA per V100 SM.
        # Fill a wave when support permits, without rounding away half the CTAs.
        kv_splits = max(
            1, torch.cuda.get_device_properties(q.device).multi_processor_count // rows
        )
    tiles_per_split = (tiles + kv_splits - 1) // kv_splits
    splits = (tiles + tiles_per_split - 1) // tiles_per_split
    if workspace is None:
        lse = torch.empty(rows, splits, 16, device=q.device, dtype=torch.float32)
        partial = torch.empty(
            rows, splits, 16, dim, device=q.device, dtype=torch.float16
        )
    else:
        from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
            _get_splitk_bufs,
        )

        lse, partial = _get_splitk_bufs(
            rows, splits, 16, dim, q.device, workspace, dtype=torch.float16
        )
    lengths = indices.reshape(-1) if topk_length is None else topk_length.reshape(-1)
    _decode_kernel(
        heads, topk, splits, q.stride(0), q.stride(1), topk_length is not None
    )(q, kv.reshape(-1, dim), indices, lengths, lse, partial, float(sm_scale))
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
        _sparse_mla_reduce_kernel,
    )

    out = torch.empty(rows, heads, dim, device=q.device, dtype=torch.float16)
    _sparse_mla_reduce_kernel[(rows, heads, dim // 64)](
        lse,
        partial,
        out,
        H=heads,
        D_V=dim,
        KV_SPLITS=splits,
        ACTIVE_SPLITS=splits,
        ACTIVE_SPLITS_POW2=1 << (splits - 1).bit_length(),
        D_CHUNK=64,
        BLOCK_K=64,
        num_warps=4,
    )
    return out.unsqueeze(0)
