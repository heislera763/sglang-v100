# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""SM70 TileLang kernels for the chunked gated-delta-rule forward pass.

The implementation follows the public FlashQLA block algorithm (MIT, Qwen
team) at the level of equations: solve the 64-token lower-triangular delta
system, then carry the recurrent state between chunks while tensor cores do
the dense intra-chunk work.  The kernels and integration here are native
SGLang code and have no FlashQLA or 1Cat runtime/build dependency.

Unlike the Hopper schedule that motivated the algorithm, this implementation
uses ordinary CTA synchronization, explicit shared-memory transposes and SM70
MMA shapes.  Persistent state is read and written directly in SGLang's indexed
``[slot, Hv, V, K]`` cache layout.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T
import torch
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

CHUNK_SIZE = 64
KKT_SHARED_ROWS = 96
KEY_DIM = 128
VALUE_DIM = 128
THREADS = 128
KKT_THREADS = 128
_LOG2_E = 1.4426950408889634
# The Cython adapter builds a normal CUDA shared object and calls it with raw
# tensor pointers.  On Volta this avoids roughly 0.2 ms of TVM-FFI launch
# overhead per helper, which is material for the multi-kernel chunked path.
_EXECUTION_BACKEND = "cython"

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    _PASS_CONFIGS[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

@tilelang.jit(
    out_idx=[4],
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _kkt_inverse_kernel(q_heads: int, value_heads: int, num_sequences: int):
    """Fuse KKT construction and SM70-friendly 16/32/64 inversion."""

    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads

    @T.macro
    def merge_diagonal_pair(
        base,
        lower_shared,
        inverse_shared,
        left_shared,
        right_shared,
        offdiag_shared,
        middle_shared,
        first,
        second,
    ):
        T.clear(left_shared)
        T.clear(right_shared)
        T.clear(offdiag_shared)
        for i, j in T.Parallel(16, 16):
            left_shared[i, j] = T.cast(inverse_shared[base + i, base + j], T.float16)
            right_shared[i, j] = T.cast(
                inverse_shared[base + 16 + i, base + 16 + j],
                T.float16,
            )
            offdiag_shared[i, j] = T.cast(
                lower_shared[base + 16 + i, base + j], T.float16
            )
        T.sync_threads()
        T.clear(first)
        T.gemm(
            right_shared,
            offdiag_shared,
            first,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for i, j in T.Parallel(32, 32):
            middle_shared[i, j] = T.cast(first[i, j], T.float16)
        T.sync_threads()
        T.clear(second)
        T.gemm(
            middle_shared,
            left_shared,
            second,
            policy=T.GemmWarpPolicy.FullRow,
        )
        for i, j in T.Parallel(16, 16):
            inverse_shared[base + 16 + i, base + j] = -second[i, j]
        T.sync_threads()

    @T.prim_func
    def main(
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkIndices: T.Tensor([chunks, 2], T.int32),
        Inverse: T.Tensor([1, tokens, value_heads, CHUNK_SIZE], T.float16),
    ):
        with T.Kernel(chunks, value_heads, threads=KKT_THREADS) as (
            chunk_id,
            value_head,
        ):
            k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            # TileLang's SM70 linear-layout lowering vectorizes some 32x32
            # fragment transfers beyond the logical 64-row tile. Keep an
            # explicit physical tail so those accesses cannot alias the next
            # shared allocation. Only the first CHUNK_SIZE rows are logical.
            lower_shared = T.alloc_shared([KKT_SHARED_ROWS, CHUNK_SIZE + 1], T.float32)
            inverse_shared = T.alloc_shared(
                [KKT_SHARED_ROWS, CHUNK_SIZE + 1], T.float32
            )
            left_shared = T.alloc_shared([32, 32], T.float16)
            right_shared = T.alloc_shared([32, 32], T.float16)
            offdiag_shared = T.alloc_shared([32, 32], T.float16)
            middle_shared = T.alloc_shared([32, 32], T.float16)
            dot = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)
            first = T.alloc_fragment([32, 32], T.float32)
            second = T.alloc_fragment([32, 32], T.float32)
            T.annotate_layout(
                {
                    lower_shared: tilelang.layout.make_linear_layout(lower_shared),
                    inverse_shared: tilelang.layout.make_linear_layout(inverse_shared),
                }
            )

            sequence = ChunkIndices[chunk_id, 0]
            local_chunk = ChunkIndices[chunk_id, 1]
            begin = CuSeqLens[sequence] + local_chunk * CHUNK_SIZE
            end = CuSeqLens[sequence + 1]
            key_head = T.floordiv(value_head, heads_per_key)
            for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                k_shared[i, d] = T.if_then_else(
                    begin + i < end,
                    K[0, begin + i, key_head, d],
                    0,
                )
            T.clear(dot)
            T.gemm(
                k_shared,
                k_shared,
                dot,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                lower_shared[i, j] = T.if_then_else(
                    (j < i) & (begin + i < end),
                    T.cast(Beta[0, begin + i, value_head], T.float32) * dot[i, j],
                    0,
                )
                inverse_shared[i, j] = 0
            T.sync_threads()

            # Four independent 16x16 forward substitutions keep the scalar
            # dependency chain short. The remaining work is tensor-core block
            # multiplication, where Volta is strongest.
            for block, column in T.Parallel(4, 16):
                inverse_shared[block * 16, block * 16 + column] = T.if_then_else(
                    column == 0, 1, 0
                )
            T.sync_threads()
            for row in T.serial(1, 16):
                for block, column in T.Parallel(4, 16):
                    inverse_shared[block * 16 + row, block * 16 + column] = 0
                T.sync_threads()
                for inner in T.serial(row):
                    for block, column in T.Parallel(4, 16):
                        inverse_shared[block * 16 + row, block * 16 + column] -= (
                            lower_shared[block * 16 + row, block * 16 + inner]
                            * inverse_shared[block * 16 + inner, block * 16 + column]
                        )
                    T.sync_threads()
                for block, column in T.Parallel(4, 16):
                    inverse_shared[block * 16 + row, block * 16 + column] = (
                        T.if_then_else(
                            column < row,
                            inverse_shared[block * 16 + row, block * 16 + column],
                            T.if_then_else(column == row, 1, 0),
                        )
                    )
                T.sync_threads()

            # Merge 16->32 independently for the upper and lower halves.
            merge_diagonal_pair(
                0,
                lower_shared,
                inverse_shared,
                left_shared,
                right_shared,
                offdiag_shared,
                middle_shared,
                first,
                second,
            )
            merge_diagonal_pair(
                32,
                lower_shared,
                inverse_shared,
                left_shared,
                right_shared,
                offdiag_shared,
                middle_shared,
                first,
                second,
            )

            # Merge the two complete 32x32 halves. For a block-lower matrix,
            # the off-diagonal inverse is -R^-1 * A_rl * L^-1.
            for i, j in T.Parallel(32, 32):
                left_shared[i, j] = T.cast(inverse_shared[i, j], T.float16)
                right_shared[i, j] = T.cast(inverse_shared[32 + i, 32 + j], T.float16)
                offdiag_shared[i, j] = T.cast(lower_shared[32 + i, j], T.float16)
            T.sync_threads()
            T.clear(first)
            T.gemm(
                right_shared,
                offdiag_shared,
                first,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(32, 32):
                middle_shared[i, j] = T.cast(first[i, j], T.float16)
            T.sync_threads()
            T.clear(second)
            T.gemm(
                middle_shared,
                left_shared,
                second,
                policy=T.GemmWarpPolicy.FullRow,
            )
            for i, j in T.Parallel(32, 32):
                inverse_shared[32 + i, j] = -second[i, j]
            T.sync_threads()

            for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if begin + i < end:
                    Inverse[0, begin + i, value_head, j] = T.cast(
                        inverse_shared[i, j], T.float16
                    )

    return main


@tilelang.jit(
    execution_backend=_EXECUTION_BACKEND,
    pass_configs=_PASS_CONFIGS,
)
def _chunk_forward_kernel(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = 32,
):
    if value_block not in (16, 32, 64) or VALUE_DIM % value_block != 0:
        raise ValueError("value_block must be 16, 32, or 64 and divide VALUE_DIM")
    tokens = T.dynamic("tokens")
    sequences = num_sequences
    chunks = T.dynamic("chunks")
    heads_per_key = value_heads // q_heads
    state_dtype = T.float32 if state_fp32 else T.float16
    v_shape = [1, tokens, value_heads, VALUE_DIM]

    @T.prim_func
    def main(
        Q: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        K: T.Tensor([1, tokens, q_heads, KEY_DIM], T.float16),
        V: T.Tensor(v_shape, T.float16),
        Inverse: T.Tensor([1, tokens, value_heads, CHUNK_SIZE], T.float16),
        GateCumsum: T.Tensor([1, tokens, value_heads], T.float32),
        Beta: T.Tensor([1, tokens, value_heads], T.float32),
        State: T.Tensor([state_slots, value_heads, VALUE_DIM, KEY_DIM], state_dtype),
        StateIndices: T.Tensor([sequences], T.int32),
        CuSeqLens: T.Tensor([sequences + 1], T.int32),
        ChunkOffsets: T.Tensor([sequences + 1], T.int32),
        Scale: T.float32,
        Output: T.Tensor([1, tokens, value_heads, VALUE_DIM], T.float16),
        Checkpoints: T.Tensor([1, chunks, value_heads, VALUE_DIM, KEY_DIM], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(VALUE_DIM, value_block),
            value_heads,
            sequences,
            threads=THREADS,
        ) as (value_tile, value_head, sequence):
            q_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            k_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            k_dot_shared = T.alloc_shared([CHUNK_SIZE, KEY_DIM], T.float16)
            k_update_shared = k_dot_shared
            value_shared = T.alloc_shared([CHUNK_SIZE, value_block], T.float16)
            inverse_shared = T.alloc_shared([CHUNK_SIZE, CHUNK_SIZE], T.float16)
            state_shared = T.alloc_shared([KEY_DIM, value_block], T.float16)
            delta_shared = T.alloc_shared([CHUNK_SIZE, value_block], T.float16)
            delta_t_shared = T.alloc_shared([value_block, CHUNK_SIZE], T.float16)
            scores_shared = T.alloc_shared([CHUNK_SIZE, CHUNK_SIZE], T.float16)
            gate_shared = T.alloc_shared([CHUNK_SIZE], T.float32)
            beta_shared = T.alloc_shared([CHUNK_SIZE], T.float32)

            state = T.alloc_fragment([value_block, KEY_DIM], T.float32)
            prediction = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            corrected = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            output = T.alloc_fragment([CHUNK_SIZE, value_block], T.float32)
            scores = T.alloc_fragment([CHUNK_SIZE, CHUNK_SIZE], T.float32)

            slot = StateIndices[sequence]
            begin = CuSeqLens[sequence]
            end = CuSeqLens[sequence + 1]
            num_chunks = T.ceildiv(end - begin, CHUNK_SIZE)
            key_head = T.floordiv(value_head, heads_per_key)
            value_start = value_tile * value_block

            for dv, d in T.Parallel(value_block, KEY_DIM):
                state[dv, d] = T.if_then_else(
                    (slot >= 0) & (value_start + dv < VALUE_DIM),
                    State[slot, value_head, value_start + dv, d],
                    0,
                )

            if slot >= 0:
                for chunk in T.serial(num_chunks):
                    chunk_begin = begin + chunk * CHUNK_SIZE
                    valid = T.min(CHUNK_SIZE, end - chunk_begin)

                    if store_checkpoints:
                        checkpoint = ChunkOffsets[sequence] + chunk
                        for dv, d in T.Parallel(value_block, KEY_DIM):
                            if value_start + dv < VALUE_DIM:
                                Checkpoints[
                                    0,
                                    checkpoint,
                                    value_head,
                                    value_start + dv,
                                    d,
                                ] = T.cast(state[dv, d], T.float16)

                    for i, d in T.Parallel(CHUNK_SIZE, KEY_DIM):
                        if i < valid:
                            q_shared[i, d] = Q[0, chunk_begin + i, key_head, d]
                            k_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                            k_dot_shared[i, d] = K[0, chunk_begin + i, key_head, d]
                        else:
                            q_shared[i, d] = 0
                            k_shared[i, d] = 0
                            k_dot_shared[i, d] = 0
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        value_shared[i, dv] = T.if_then_else(
                            (i < valid) & (value_start + dv < VALUE_DIM),
                            V[
                                0,
                                chunk_begin + i,
                                value_head,
                                value_start + dv,
                            ],
                            0,
                        )
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        inverse_shared[i, j] = T.if_then_else(
                            i < valid,
                            Inverse[0, chunk_begin + i, value_head, j],
                            0,
                        )
                    for i in T.Parallel(CHUNK_SIZE):
                        gate_shared[i] = T.if_then_else(
                            i < valid,
                            GateCumsum[0, chunk_begin + i, value_head],
                            GateCumsum[0, chunk_begin + valid - 1, value_head],
                        )
                        beta_shared[i] = T.if_then_else(
                            i < valid,
                            T.cast(
                                Beta[0, chunk_begin + i, value_head],
                                T.float32,
                            ),
                            0,
                        )
                    for d, dv in T.Parallel(KEY_DIM, value_block):
                        state_shared[d, dv] = T.cast(state[dv, d], T.float16)
                    T.sync_threads()

                    # W = V - exp(g) K H.
                    T.clear(prediction)
                    T.gemm(
                        k_dot_shared,
                        state_shared,
                        prediction,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        prediction[i, dv] = (
                            T.cast(value_shared[i, dv], T.float32)
                            - T.exp2(gate_shared[i] * _LOG2_E) * prediction[i, dv]
                        )
                        delta_shared[i, dv] = T.cast(prediction[i, dv], T.float16)
                    T.sync_threads()

                    # Apply the solved intra-chunk delta transform.
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores_shared[i, j] = T.cast(
                            T.if_then_else(
                                (j <= i) & (i < valid),
                                T.cast(inverse_shared[i, j], T.float32)
                                * T.exp2((gate_shared[i] - gate_shared[j]) * _LOG2_E)
                                * beta_shared[j],
                                0,
                            ),
                            T.float16,
                        )
                    T.sync_threads()
                    T.clear(corrected)
                    T.gemm(
                        scores_shared,
                        delta_shared,
                        corrected,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        delta_shared[i, dv] = T.cast(corrected[i, dv], T.float16)
                    T.sync_threads()

                    # O = scale * (exp(g) QH + (G * QK^T) V_delta).
                    T.clear(output)
                    T.gemm(
                        q_shared,
                        state_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.clear(scores)
                    T.gemm(
                        q_shared,
                        k_update_shared,
                        scores,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                        scores[i, j] = T.if_then_else(
                            (j <= i) & (i < valid),
                            scores[i, j]
                            * T.exp2((gate_shared[i] - gate_shared[j]) * _LOG2_E)
                            * Scale,
                            0,
                        )
                        scores_shared[i, j] = T.cast(scores[i, j], T.float16)
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        output[i, dv] *= Scale * T.exp2(gate_shared[i] * _LOG2_E)
                    T.sync_threads()
                    T.gemm(
                        scores_shared,
                        delta_shared,
                        output,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, dv in T.Parallel(CHUNK_SIZE, value_block):
                        if (i < valid) & (value_start + dv < VALUE_DIM):
                            Output[
                                0,
                                chunk_begin + i,
                                value_head,
                                value_start + dv,
                            ] = T.cast(output[i, dv], T.float16)

                    # H' = exp(g_last) H + V_delta^T diag(exp(g_last-g)) K.
                    for dv, i in T.Parallel(value_block, CHUNK_SIZE):
                        delta_t_shared[dv, i] = T.cast(
                            T.cast(delta_shared[i, dv], T.float32)
                            * T.exp2(
                                (gate_shared[valid - 1] - gate_shared[i]) * _LOG2_E
                            ),
                            T.float16,
                        )
                    last_decay = T.exp2(gate_shared[valid - 1] * _LOG2_E)
                    for dv, d in T.Parallel(value_block, KEY_DIM):
                        state[dv, d] *= last_decay
                    T.gemm(
                        delta_t_shared,
                        k_shared,
                        state,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                for dv, d in T.Parallel(value_block, KEY_DIM):
                    if value_start + dv < VALUE_DIM:
                        State[slot, value_head, value_start + dv, d] = state[dv, d]

    return main


@lru_cache(maxsize=None)
def _get_kkt_inverse(q_heads: int, value_heads: int, num_sequences: int):
    return _kkt_inverse_kernel(q_heads, value_heads, num_sequences)


@lru_cache(maxsize=None)
def _get_chunk_forward(
    q_heads: int,
    value_heads: int,
    num_sequences: int,
    state_slots: int,
    store_checkpoints: bool,
    state_fp32: bool,
    value_block: int = 32,
):
    return _chunk_forward_kernel(
        q_heads,
        value_heads,
        num_sequences,
        state_slots,
        store_checkpoints,
        state_fp32,
        value_block,
    )


def chunked_gdn_sm70(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    store_checkpoints: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run normalized, variable-length GDN prefill on SM70."""

    if torch.cuda.get_device_capability(q.device) != (7, 0):
        raise ValueError("The TileLang chunked GDN kernel is specialized for SM70.")
    if q.dtype != torch.float16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("The SM70 TileLang chunked GDN kernel requires FP16 QKV.")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1:
        raise ValueError(
            "Variable-length QKV must be flattened with batch dimension 1."
        )
    _, tokens, q_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if key_dim != KEY_DIM or value_dim != VALUE_DIM:
        raise ValueError("The SM70 TileLang chunked GDN path supports K=V=128.")
    if state.dtype not in (torch.float16, torch.float32):
        raise ValueError(
            "The SM70 TileLang chunked GDN path requires FP16 or FP32 state."
        )

    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    cu_seqlens = cu_seqlens.to(dtype=torch.int32).contiguous()
    state_indices = state_indices.to(dtype=torch.int32).contiguous()
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_indices = chunk_indices.to(dtype=torch.int32).contiguous()
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE).to(dtype=torch.int32)
    g_cumsum = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    output = torch.empty(
        (1, tokens, value_heads, value_dim),
        dtype=torch.float16,
        device=q.device,
    )
    num_chunks = int(chunk_indices.shape[0])
    if store_checkpoints:
        checkpoints = torch.empty(
            (1, num_chunks, value_heads, value_dim, key_dim),
            dtype=torch.float16,
            device=q.device,
        )
    else:
        # A zero-sized placeholder keeps the compiled ABI stable.
        checkpoints = torch.empty(
            (1, 0, value_heads, value_dim, key_dim),
            dtype=torch.float16,
            device=q.device,
        )

    num_sequences = state_indices.numel()
    state_slots = state.shape[0]
    inverse = _get_kkt_inverse(q_heads, value_heads, num_sequences)(
        k,
        beta,
        cu_seqlens,
        chunk_indices,
    )
    _get_chunk_forward(
        q_heads,
        value_heads,
        num_sequences,
        state_slots,
        store_checkpoints,
        state.dtype == torch.float32,
        value_block=32,
    )(
        q,
        k,
        v,
        inverse,
        g_cumsum,
        beta,
        state,
        state_indices,
        cu_seqlens,
        chunk_offsets,
        float(scale),
        output,
        checkpoints,
    )
    return output, checkpoints if store_checkpoints else None
