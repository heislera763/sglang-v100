"""Shared TileLang settings for the SM70 QSA kernels."""

import tilelang

tilelang.set_log_level("WARNING")

# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
# `out_idx` list in place. Patch once on import (idempotent). Mirrors
# sglang/srt/layers/attention/dsa/tilelang_kernel.py.
from tilelang.jit.adapter.base import (  # noqa: E402
    BaseKernelAdapter as _BaseKernelAdapter,
)

if not getattr(_BaseKernelAdapter, "_legalize_result_idx_patched", False):
    _orig_legalize = _BaseKernelAdapter._legalize_result_idx

    def _legalize_result_idx_safe(self, result_idx):
        if isinstance(result_idx, list):
            result_idx = list(result_idx)
        return _orig_legalize(self, result_idx)

    _BaseKernelAdapter._legalize_result_idx = _legalize_result_idx_safe
    _BaseKernelAdapter._legalize_result_idx_patched = True

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

"""TileLang fallback combining QSA split-KV attention."""

import tilelang.language as T


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def _decode_combine_kernel(
    batch: int,
    heads: int,
    dim: int,
    max_splits: int,
    threads: int,
    min_tokens_per_split: int,
    selected_tokens: int = 0,
):
    @T.prim_func
    def main(
        PartialO: T.Tensor([batch, max_splits, heads, dim], T.float16),
        PartialLSE: T.Tensor([batch, max_splits, heads], T.float32),
        SeqLens: T.Tensor([batch], T.int32),
        Output: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (head, batch_id):
            lse = T.alloc_shared([max_splits], T.float32)
            max_lse = T.alloc_fragment([1], T.float32)
            sum_lse = T.alloc_fragment([1], T.float32)
            output = T.alloc_fragment([dim], T.float32)
            context = (
                T.min(SeqLens[batch_id], selected_tokens)
                if selected_tokens > 0
                else SeqLens[batch_id]
            )
            active_splits = T.min(
                max_splits,
                T.max(
                    1,
                    T.ceildiv(context, min_tokens_per_split),
                ),
            )
            for split in T.Parallel(max_splits):
                lse[split] = T.if_then_else(
                    split < active_splits,
                    PartialLSE[batch_id, split, head],
                    -(2**30),
                )
            T.fill(max_lse, -(2**30))
            for split in T.serial(max_splits):
                max_lse[0] = T.max(max_lse[0], lse[split])
            T.fill(sum_lse, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    sum_lse[0] += T.exp2(lse[split] - max_lse[0])
            T.fill(output, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    weight = T.exp2(lse[split] - max_lse[0]) / sum_lse[0]
                    for d in T.Parallel(dim):
                        output[d] += weight * T.cast(
                            PartialO[batch_id, split, head, d],
                            T.float32,
                        )
            for d in T.Parallel(dim):
                Output[batch_id, head, d] = T.cast(output[d], T.float16)

    return main

"""Exact dense-prefix D256 prefill kernel for SM70.

Long paged prefixes are gathered into logical order before this kernel is
called.  The attention body keeps the existing TileLang N32 reduction order
while removing page-table lookup, integer divide, and scattered-page address
resolution from every K/V element load.
"""



_LOG2_E = 1.4426950408889634
_D256_PASS_CONFIGS = dict(pass_configs)
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _D256_PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = False
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _D256_PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = True

_BLOCK_M = 64
_BLOCK_N = 32
_THREADS = 256


@tilelang.jit(out_idx=[5], pass_configs=_D256_PASS_CONFIGS)
def _dense_prefix_d256_kernel(
    heads,
    heads_kv,
    qk_policy=2,
    pv_policy=1,
    block_m=_BLOCK_M,
    block_n=_BLOCK_N,
    threads=_THREADS,
    kv_union=False,
    gemm_version=2,
    num_stages=0,
):
    dim = 256
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")
    qk_warp_policy = (
        T.GemmWarpPolicy.FullRow
        if qk_policy == 1
        else T.GemmWarpPolicy.FullCol
        if qk_policy == 2
        else T.GemmWarpPolicy.Square
    )
    pv_warp_policy = (
        T.GemmWarpPolicy.FullRow
        if pv_policy == 1
        else T.GemmWarpPolicy.FullCol
        if pv_policy == 2
        else T.GemmWarpPolicy.Square
    )
    gemm = getattr(T, "gemm_v1" if gemm_version == 1 else "gemm_v2", T.gemm)

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        K: T.Tensor([nk, heads_kv, dim], T.float16),
        V: T.Tensor([nk, heads_kv, dim], T.float16),
        prefix_kv_len: T.int32,
        sm_scale: T.float32,
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(nt, block_m), heads, threads=threads) as (
            q_tile,
            q_head,
        ):
            Q_shared = T.alloc_shared([block_m, dim], T.float16)
            if kv_union:
                KV_shared = T.alloc_shared([block_n, dim], T.float16)
                K_shared = KV_shared
                V_shared = KV_shared
            else:
                K_shared = T.alloc_shared([block_n, dim], T.float16)
                V_shared = T.alloc_shared([block_n, dim], T.float16)
            P_shared = T.alloc_shared([block_m, block_n], T.float16)

            scores = T.alloc_fragment([block_m, block_n], T.float32)
            probabilities = T.alloc_fragment([block_m, block_n], T.float16)
            output = T.alloc_fragment([block_m, dim], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_max = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            tile_sum = T.alloc_fragment([block_m], T.float32)
            rescale = T.alloc_fragment([block_m], T.float32)

            kv_head = q_head // (heads // heads_kv)
            query_start = q_tile * block_m

            T.clear(Q_shared)
            for row, d in T.Parallel(block_m, dim):
                if query_start + row < nt:
                    Q_shared[row, d] = Q[query_start + row, q_head, d]

            T.fill(output, 0)
            T.fill(row_max, -T.infinity(T.float32))
            T.fill(row_sum, 0)

            loop_end = T.min(
                T.ceildiv(nk, block_n),
                T.ceildiv(
                    prefix_kv_len + query_start + block_m,
                    block_n,
                ),
            )
            for kv_tile in T.Pipelined(loop_end, num_stages=num_stages):
                tile_start = kv_tile * block_n
                T.clear(K_shared)
                for n, d in T.Parallel(block_n, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        K_shared[n, d] = K[kv_index, kv_head, d]

                for row, n in T.Parallel(block_m, block_n):
                    kv_index = tile_start + n
                    scores[row, n] = T.if_then_else(
                        (query_start + row < nt)
                        & (kv_index < nk)
                        & (kv_index <= prefix_kv_len + query_start + row),
                        0,
                        -T.infinity(T.float32),
                    )

                gemm(
                    Q_shared,
                    K_shared,
                    scores,
                    transpose_B=True,
                    policy=qk_warp_policy,
                )
                T.copy(row_max, previous_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for row in T.Parallel(block_m):
                    row_max[row] = T.if_then_else(
                        row_max[row] == -T.infinity(T.float32),
                        0,
                        row_max[row],
                    )
                    row_max[row] = T.max(row_max[row], previous_max[row])
                    rescale[row] = T.exp2(
                        (previous_max[row] - row_max[row]) * sm_scale * _LOG2_E
                    )
                    row_sum[row] *= rescale[row]
                for row, d in T.Parallel(block_m, dim):
                    output[row, d] *= rescale[row]
                for row, n in T.Parallel(block_m, block_n):
                    scores[row, n] = T.exp2(
                        (scores[row, n] - row_max[row]) * sm_scale * _LOG2_E
                    )
                T.reduce_sum(scores, tile_sum, dim=1)
                for row in T.Parallel(block_m):
                    row_sum[row] += tile_sum[row]

                T.clear(V_shared)
                for n, d in T.Parallel(block_n, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        V_shared[n, d] = V[kv_index, kv_head, d]

                for row, n in T.Parallel(block_m, block_n):
                    P_shared[row, n] = T.cast(scores[row, n], T.float16)
                T.copy(P_shared, probabilities)
                gemm(
                    probabilities,
                    V_shared,
                    output,
                    policy=pv_warp_policy,
                )

            for row, d in T.Parallel(block_m, dim):
                if query_start + row < nt:
                    Output[query_start + row, q_head, d] = T.cast(
                        output[row, d] / row_sum[row], T.float16
                    )

    return main


_KERNEL_CACHE = {}


def get_dense_prefix_d256_kernel(
    heads,
    heads_kv,
    qk_policy=2,
    pv_policy=1,
    block_m=_BLOCK_M,
    block_n=_BLOCK_N,
    threads=_THREADS,
    kv_union=False,
    gemm_version=2,
    num_stages=0,
):
    key = (
        heads,
        heads_kv,
        qk_policy,
        pv_policy,
        block_m,
        block_n,
        threads,
        kv_union,
        gemm_version,
        num_stages,
    )
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _dense_prefix_d256_kernel(
            heads=heads,
            heads_kv=heads_kv,
            qk_policy=qk_policy,
            pv_policy=pv_policy,
            block_m=block_m,
            block_n=block_n,
            threads=threads,
            kv_union=kv_union,
            gemm_version=gemm_version,
            num_stages=num_stages,
        )
    return _KERNEL_CACHE[key]
