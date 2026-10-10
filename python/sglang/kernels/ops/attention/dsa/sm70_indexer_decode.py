"""Volta MMA index scoring with live-history bounds and fused head reduction.

Keep the graph's reserved output width while skipping unused score tiles on the
device. Q/K are FP16, dot products and both weighted reductions remain FP32.
No full [query, head, history] score temporary or host length scan is needed.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    compile_flags=["--fmad=false"],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _index_score_kernel(capacity):
    rows = T.dynamic("rows")
    heads, dim, block = 32, 128, 64

    @T.prim_func
    def mqa_logits_decode_sm70_kernel(
        Q: T.Tensor([rows, heads, dim], T.float16),
        K: T.Tensor([rows, capacity, dim], T.float16),
        S: T.Tensor([rows, capacity], T.float32),
        W: T.Tensor([rows, heads], T.float32),
        L: T.Tensor([rows], T.int32),
        O: T.Tensor([rows, capacity], T.float32),
    ):
        with T.Kernel(T.ceildiv(capacity, block), rows, threads=128) as (tile, row):
            qs = T.alloc_shared([heads, dim], T.float16)
            ks = T.alloc_shared([block, dim], T.float16)
            dots = T.alloc_fragment([heads, block], T.float32)
            result = T.alloc_fragment([block], T.float32)
            length = T.min(capacity, T.max(0, L[row]))
            start = tile * block
            if start < length:
                T.copy(Q[row, :, :], qs)
                for n, d in T.Parallel(block, dim):
                    ks[n, d] = T.if_then_else(
                        start + n < length, K[row, start + n, d], 0
                    )
                T.clear(dots)
                T.gemm(qs, ks, dots, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for h, n in T.Parallel(heads, block):
                    # Separate FP32 multiplication and summation, as in Torch.
                    # The conditional preserves ReLU's NaN and signed-zero rules.
                    dots[h, n] = (
                        T.if_then_else(dots[h, n] < 0, 0, dots[h, n]) * W[row, h]
                    )
                T.reduce_sum(dots, result, dim=0)
                for n in T.Parallel(block):
                    if start + n < capacity:
                        O[row, start + n] = T.if_then_else(
                            start + n < length,
                            result[n] * S[row, start + n],
                            -T.infinity(T.float32),
                        )
            else:
                for n in T.Parallel(block):
                    if start + n < capacity:
                        O[row, start + n] = -T.infinity(T.float32)

    return mqa_logits_decode_sm70_kernel


def mqa_logits_decode_sm70(query, keys, key_scales, weights, lengths):
    """Score batched FP16 index keys for 1..8 queries with 32 heads of width128.

    Software-FP8 queries are converted losslessly to FP16. The caller owns
    learned-index key scaling and request/causal metadata; latent KV is untouched.
    Lengths can change on each graph replay, including zero and the full width.
    All future columns are -inf. Scratch is call-owned; reduction rounding can
    differ from Torch's head sum, with the same FP32 arithmetic boundaries.
    """
    if not (
        query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (7, 0)
        and query.ndim == 3
        and 1 <= query.shape[0] <= 8
        and query.shape[1:] == (32, 128)
        and query.dtype in (torch.float16, torch.float8_e4m3fn)
        and keys.ndim == 3
        and keys.shape[0] == query.shape[0]
        and keys.shape[-1] == 128
        and keys.dtype == torch.float16
        and keys.is_contiguous()
        and keys.data_ptr() % 16 == 0
        and key_scales.shape == keys.shape[:2]
        and weights.shape == query.shape[:2]
        and key_scales.dtype == weights.dtype == torch.float32
        and lengths.numel() == query.shape[0]
        and lengths.dtype == torch.int32
        and all(t.device == query.device for t in (keys, key_scales, weights, lengths))
    ):
        raise ValueError(
            "SM70 decode scoring requires rows1..8/heads32/dim128, batched aligned FP16 keys, "
            "FP32 scales/weights and CUDA int32 lengths"
        )
    output = torch.empty(keys.shape[:2], device=query.device, dtype=torch.float32)
    if keys.shape[1]:
        query = query.half().contiguous()
        if query.data_ptr() % 16:
            raise ValueError("SM70 decode scoring queries require 16-byte alignment")
        _index_score_kernel(keys.shape[1])(
            query,
            keys,
            key_scales.contiguous(),
            weights.contiguous(),
            lengths.reshape(-1).contiguous(),
            output,
        )
    return output
