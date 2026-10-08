"""SM70 QSA prefill: trade masked arithmetic for Tensor Core/KV reuse.

Several query/head rows share a K/V tile. Selection remains per query; scanning
logical keys changes accumulation order but does not enlarge its attention set.
"""

import tilelang
import tilelang.language as T
import torch
import triton
import triton.language as tl
from sglang_v100_plus.kernels.attention import _D256_PASS_CONFIGS, _LOG2_E


@triton.jit
def _selection_mask(
    Ids,
    Mask,
    NK: tl.constexpr,
    NT: tl.constexpr,
    TOPK: tl.constexpr,
    NW: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    visible = NK - NT + row + 1
    logical = tl.load(Ids + row * TOPK + col, col < TOPK, -1)
    valid = (col < TOPK) & (col < visible) & (logical >= 0) & (logical < visible)
    tl.atomic_or(
        Mask + row * NW + logical // 32, (1 << (logical % 32)).to(tl.int32), valid
    )


def _make_mask(ids, nk):
    # QSA selects distinct micro-blocks, expanded to distinct token positions.
    # A bitmap preserves that selection, including each row's native prefix cutoff.
    nt, topk = ids.shape
    mask = torch.zeros((nt, triton.cdiv(nk, 32)), dtype=torch.int32, device=ids.device)
    _selection_mask[(nt, triton.cdiv(topk, 256))](
        ids, mask, nk, nt, topk, mask.shape[1], 256
    )
    return mask


@tilelang.jit(out_idx=[6], pass_configs=_D256_PASS_CONFIGS)
def _masked_prefill_kernel(heads):
    block_m, block_n, threads = 64, 32, 256
    heads_kv = 1
    dim = 256
    nt = T.dynamic("nt")
    nk = T.dynamic("nk")
    nw = T.dynamic("nw")
    qk_warp_policy = T.GemmWarpPolicy.FullCol
    pv_warp_policy = T.GemmWarpPolicy.FullRow
    gemm = getattr(T, "gemm_v2", T.gemm)

    @T.prim_func
    def main(
        Q: T.Tensor([nt, heads, dim], T.float16),
        K: T.Tensor([nk, heads_kv, dim], T.float16),
        V: T.Tensor([nk, heads_kv, dim], T.float16),
        Mask: T.Tensor([nt, nw], T.int32),
        prefix_kv_len: T.int32,
        sm_scale: T.float32,
        Output: T.Tensor([nt, heads, dim], T.float16),
    ):
        with T.Kernel(T.ceildiv(nt * heads, block_m), threads=threads) as q_tile:
            Q_shared = T.alloc_shared([block_m, dim], T.float16)
            # Reuse the K allocation for V after QK finishes. Keep the tile
            # within Volta's 96 KiB shared-memory limit.
            KV_shared = T.alloc_shared([block_n, dim], T.float16)
            K_shared, V_shared = KV_shared, KV_shared
            P_shared = T.alloc_shared([block_m, block_n], T.float16)

            scores = T.alloc_fragment([block_m, block_n], T.float32)
            probabilities = T.alloc_fragment([block_m, block_n], T.float16)
            output = T.alloc_fragment([block_m, dim], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_max = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            tile_sum = T.alloc_fragment([block_m], T.float32)
            rescale = T.alloc_fragment([block_m], T.float32)

            query_start = q_tile * block_m

            T.clear(Q_shared)
            for row, d in T.Parallel(block_m, dim):
                if (query_start + row) // heads < nt:
                    Q_shared[row, d] = Q[
                        (query_start + row) // heads, (query_start + row) % heads, d
                    ]

            T.fill(output, 0)
            T.fill(row_max, -1.0e30)
            T.fill(row_sum, 0)

            loop_end = T.min(
                T.ceildiv(nk, block_n),
                T.ceildiv(
                    prefix_kv_len + (query_start + block_m - 1) // heads + 1,
                    block_n,
                ),
            )
            for kv_tile in T.serial(loop_end):
                tile_start = kv_tile * block_n
                T.clear(K_shared)
                for n, d in T.Parallel(block_n, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        K_shared[n, d] = K[kv_index, 0, d]

                T.clear(scores)

                gemm(
                    Q_shared,
                    K_shared,
                    scores,
                    transpose_B=True,
                    policy=qk_warp_policy,
                )
                for row, n in T.Parallel(block_m, block_n):
                    query = (query_start + row) // heads
                    logical = tile_start + n
                    selected = T.if_then_else(
                        (query < nt) & (logical < nk),
                        (Mask[query, logical // 32] >> (logical % 32)) & 1,
                        0,
                    )
                    scores[row, n] = T.if_then_else(
                        selected != 0, scores[row, n], -1.0e30
                    )
                T.copy(row_max, previous_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for row in T.Parallel(block_m):
                    row_max[row] = T.max(row_max[row], previous_max[row])
                    rescale[row] = T.exp2(
                        (previous_max[row] - row_max[row]) * sm_scale * _LOG2_E
                    )
                    row_sum[row] *= rescale[row]
                for row, d in T.Parallel(block_m, dim):
                    output[row, d] *= rescale[row]
                for row, n in T.Parallel(block_m, block_n):
                    query = (query_start + row) // heads
                    logical = tile_start + n
                    selected = T.if_then_else(
                        (query < nt) & (logical < nk),
                        (Mask[query, logical // 32] >> (logical % 32)) & 1,
                        0,
                    )
                    scores[row, n] = T.if_then_else(
                        selected != 0,
                        T.exp2((scores[row, n] - row_max[row]) * sm_scale * _LOG2_E),
                        0,
                    )
                T.reduce_sum(scores, tile_sum, dim=1)
                for row in T.Parallel(block_m):
                    row_sum[row] += tile_sum[row]

                T.clear(V_shared)
                for n, d in T.Parallel(block_n, dim):
                    kv_index = tile_start + n
                    if kv_index < nk:
                        V_shared[n, d] = V[kv_index, 0, d]

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
                if (query_start + row) // heads < nt:
                    Output[
                        (query_start + row) // heads, (query_start + row) % heads, d
                    ] = T.cast(output[row, d] / T.max(row_sum[row], 1.0e-30), T.float16)

    return main


def qsa_masked_prefill(q, k_cache, v_cache, table, requests, indices, seq_len, scale):
    """Exact QSA membership with TC arithmetic over shared logical K/V tiles.

    QSA's block selector supplies unique token positions. Inputs and AV
    probabilities retain FP16 boundaries; reductions and normalization use FP32.
    The selector confines this dense masked execution to moderate contexts.
    """
    if (
        q.ndim != 3
        or q.shape[1:] not in ((3, 256), (6, 256))
        or q.dtype != torch.float16
        or q.device.type != "cuda"
        or k_cache.ndim != 3
        or k_cache.shape[1:] != (1, 256)
        or v_cache.shape != k_cache.shape
        or k_cache.dtype != torch.float8_e5m2
        or v_cache.dtype != k_cache.dtype
        or requests.numel() != 1
        or indices.ndim != 2
        or indices.shape[0] != q.shape[0]
        or indices.dtype != torch.int32
        or not q.shape[0] <= seq_len <= table.shape[1]
    ):
        raise ValueError("Unsupported SM70 masked QSA prefill layout")
    slots = table.index_select(0, requests.long())[0, :seq_len].long()
    k = k_cache.index_select(0, slots).half().contiguous()
    v = v_cache.index_select(0, slots).half().contiguous()
    mask = _make_mask(indices.contiguous(), seq_len)
    return _masked_prefill_kernel(q.shape[1])(
        q.contiguous(), k, v, mask, seq_len - q.shape[0], scale
    )
