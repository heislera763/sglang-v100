"""FP16 routed expert GEMM with explicit Volta MMA and FP32 accumulation.

Reuse upstream's padded routing permutation and scatter to route-major output.
Activation, per-stage FP16 rounding and expert reduction belong to the caller.
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
def _routed_gemm(n, k, block_m, topk, mul):
    stages = 2
    bm = max(16, min(64, block_m))
    bn = 64
    bk = 64
    subtiles = max(1, block_m // bm)
    nt = T.dynamic("nt")
    nr = T.dynamic("nr")
    ns = T.dynamic("ns")
    ne = T.dynamic("ne")
    nw = T.dynamic("nw")
    nb = T.dynamic("nb")

    @T.prim_func
    def routed_fp16_gemm(
        A: T.Tensor([nt, k], T.float16),
        B: T.Tensor([ne, n, k], T.float16),
        C: T.Tensor([nr, n], T.float16),
        Ids: T.Tensor([ns], T.int32),
        Experts: T.Tensor([nb], T.int32),
        Count: T.Tensor([1], T.int32),
        Weights: T.Tensor([nw], T.float32),
    ):
        with T.Kernel(
            T.ceildiv(ns, block_m) * subtiles, T.ceildiv(n, bn), threads=128
        ) as (row, col):
            ash = T.alloc_shared([bm, bk], T.float16)
            bsh = T.alloc_shared([bn, bk], T.float16)
            out = T.alloc_fragment([bm, bn], T.float32)
            block = row // subtiles
            offset = (row % subtiles) * bm
            expert = Experts[block]
            T.clear(out)
            if block * block_m < Count[0]:
                if expert >= 0:
                    for kk in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                        for m, d in T.Parallel(bm, bk):
                            route = T.if_then_else(
                                offset + m < block_m,
                                Ids[block * block_m + offset + m],
                                nr,
                            )
                            ash[m, d] = T.if_then_else(
                                (route >= 0) & (route < nr) & (kk * bk + d < k),
                                A[route // topk, kk * bk + d],
                                0,
                            )
                        for j, d in T.Parallel(bn, bk):
                            bsh[j, d] = T.if_then_else(
                                (col * bn + j < n) & (kk * bk + d < k),
                                B[expert, col * bn + j, kk * bk + d],
                                0,
                            )
                        T.gemm(
                            ash,
                            bsh,
                            out,
                            transpose_B=True,
                            policy=(
                                T.GemmWarpPolicy.Square
                                if bm >= 32
                                else T.GemmWarpPolicy.FullCol
                            ),
                        )
                # Valid route ids occur once in the padded permutation. Thus
                # scatter rows/column tiles have unique writers, although a
                # static race checker cannot infer uniqueness of indirect ids.
                for m, j in T.Parallel(bm, bn):
                    if offset + m < block_m:
                        route = Ids[block * block_m + offset + m]
                        if (route >= 0) & (route < nr) & (col * bn + j < n):
                            if mul:
                                C[route, col * bn + j] = T.cast(
                                    T.if_then_else(
                                        expert >= 0, out[m, j] * Weights[route], 0
                                    ),
                                    T.float16,
                                )
                            else:
                                C[route, col * bn + j] = T.cast(out[m, j], T.float16)

    return routed_fp16_gemm


def sm70_fp16_moe_gemm(
    a,
    b,
    output,
    sorted_ids,
    expert_ids,
    padded_count,
    weights,
    block_size,
    top_k,
    mul_routed_weight,
):
    """Write FP16 [routes,N] from FP16 A and expert-major [E,N,K] weights.

    Routing is the standard moe_align_block_size contract: each valid route id
    appears once; padding is >=routes; expert -1 writes zero. The device scalar
    padded_count gates uninitialized tail metadata. No CPU reads or new scratch.
    """
    if not (
        a.is_cuda
        and torch.cuda.get_device_capability(a.device) == (7, 0)
        and a.ndim == 2
        and b.ndim == 3
        and output.ndim in (2, 3)
        and a.dtype == b.dtype == output.dtype == torch.float16
        and a.shape[1] == b.shape[2]
        and output.shape[-1] == b.shape[1]
        and all(
            t.device == a.device and t.is_contiguous()
            for t in (a, b, output, sorted_ids, expert_ids, padded_count, weights)
        )
        and sorted_ids.ndim == expert_ids.ndim == 1
        and sorted_ids.dtype == expert_ids.dtype == padded_count.dtype == torch.int32
        and padded_count.numel() == 1
        and weights.dtype == torch.float32
        and weights.numel() == output.numel() // b.shape[1]
        and top_k > 0
        and weights.numel() == a.shape[0] * top_k
        and block_size in (4, 8, 16, 32, 64, 128, 256)
        and expert_ids.numel() >= (sorted_ids.numel() + block_size - 1) // block_size
    ):
        raise ValueError("Unsupported SM70 FP16 routed GEMM layout")
    _routed_gemm(b.shape[1], b.shape[2], block_size, top_k, mul_routed_weight)(
        a,
        b,
        output.view(-1, b.shape[1]),
        sorted_ids,
        expert_ids,
        padded_count,
        weights.reshape(-1),
    )
