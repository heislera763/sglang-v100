"""Volta Tensor Core sparse MLA prefill over selected FP16 latent KV rows.

One CTA owns a query's local heads and reuses each gathered tile for QK and PV.
Support/masks and FP16 probability boundaries are unchanged; MMA and online
softmax change reduction order, so this is not bit-exact to scalar Triton.
"""

import tilelang
import tilelang.language as T
import torch

from sglang.kernels.ops.kvcache.cache_ops import q8kv8_topk_length_from_indices

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(out_idx=[5], pass_configs=_PASS_CONFIGS)
def _prefill_kernel(heads):
    bn, threads = 64, 128
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")
    topk = T.dynamic("topk")
    bm = 16
    dim = 512

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        KV: T.Tensor([nk, dim], T.float16),
        Ids: T.Tensor([nt, topk], T.int32),
        Lengths: T.Tensor([nt], T.int32),
        scale: T.float32,
        Out: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(nt, threads=threads) as row:
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
            limit = T.min(topk, T.max(0, Lengths[row]))
            for tile in T.serial(T.ceildiv(limit, bn)):
                for n in T.Parallel(bn):
                    ids[n] = T.if_then_else(
                        tile * bn + n < limit, Ids[row, tile * bn + n], -1
                    )
                for n, d in T.Parallel(bn, dim):
                    kv[n, d] = T.if_then_else(
                        (ids[n] >= 0) & (ids[n] < nk), KV[ids[n], d], 0
                    )
                T.clear(scores)
                T.gemm(
                    qs, kv, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullCol
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
                        (ids[n] >= 0) & (ids[n] < nk), T.exp2(scores[h, n] - m[h]), 0
                    )
                T.reduce_sum(scores, z, dim=1)
                for h in T.Parallel(bm):
                    l[h] += z[h]
                T.copy(scores, ps)
                T.gemm(ps, kv, o, policy=T.GemmWarpPolicy.FullCol)
            for h, d in T.Parallel(bm, dim):
                if h < heads:
                    Out[row, h, d] = T.cast(o[h, d] / T.max(l[h], 1.0e-30), T.float16)

    return main


def sparse_mla_prefill_sm70(q, kv, indices, sm_scale, topk_length=None):
    """Return [1, tokens, heads8/16, 512] using FP16 inputs and FP32 reductions.

    Index rows address physical KV slots, with negative padding. Optional lengths
    bound the last valid column; otherwise a device scan derives that bound.
    Both gathered matrices remain FP16; the KV cache is never quantized/copied.
    """
    if not (
        q.is_cuda
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and q.ndim == 3
        and q.shape[1] in (8, 16)
        and q.shape[2] == 512
        and q.dtype == kv.dtype == torch.float16
        and kv.device == indices.device == q.device
        and kv.shape[-1] == 512
        and indices.dtype == torch.int32
        and indices.shape[0] == q.shape[0]
        and indices.shape[-1] > 0
        and (indices.ndim == 2 or (indices.ndim == 3 and indices.shape[1] == 1))
        and (kv.ndim == 2 or (kv.ndim == 3 and kv.shape[1] == 1))
    ):
        raise ValueError(
            "SM70 sparse prefill requires FP16 heads8/16, latent512 and CUDA int32 indices"
        )
    indices = indices.reshape(q.shape[0], indices.shape[-1]).contiguous()
    if topk_length is None:
        topk_length = q8kv8_topk_length_from_indices(indices)
    elif not (
        topk_length.device == q.device
        and topk_length.dtype == torch.int32
        and topk_length.numel() == q.shape[0]
    ):
        raise ValueError("SM70 sparse prefill lengths must be one CUDA int32 per query")
    return _prefill_kernel(q.shape[1])(
        q.contiguous(),
        kv.reshape(-1, 512).contiguous(),
        indices,
        topk_length.reshape(-1).contiguous(),
        float(sm_scale),
    ).unsqueeze(0)
