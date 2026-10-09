"""FP16 sparse MLA with a shared key union for two adjacent Volta queries.

Each index row must contain distinct valid KV positions, as GLM pool4 does.
Ownership bits preserve each query's support, including tails and empty rows.
The union changes key traversal/reduction order; it does not approximate the
selected attention support. The 128-wide latent fragments fit Volta's 96 KiB
shared-memory budget, unlike a monolithic 512-wide MMA.
"""

import torch
import triton

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import (
    _sparse_mla_fused_kernel,
    triton_sparse_mla_fwd,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import _union_dedup


def sparse_mla_prefill_sm70(q, kv, indices, sm_scale):
    """Return [1, tokens, heads, 512] for FP16 SM70 heads8/16, no RoPE tail.

    Inputs share a CUDA device. Index rows are int32 with -1 padding and no
    duplicate valid slots. Like the established sparse MLA API, callers own
    index bounds. Ragged last queries retain the ordinary sparse implementation.
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
    q, kv = q.contiguous(), kv.reshape(-1, 512).contiguous()
    tokens, heads, _ = q.shape
    indices = indices.reshape(tokens, indices.shape[-1])
    out = torch.empty_like(q)
    main = tokens // 2 * 2
    if main:
        padded_k = triton.next_power_of_2(indices.shape[-1])
        padded = torch.full((main, padded_k), -1, device=q.device, dtype=torch.int32)
        padded[:, : indices.shape[-1]].copy_(indices[:main])
        union, bits, lengths = _union_dedup(padded, 2)
        # Two 8-head rows fill one Volta MMA tile. Two 16-head rows use eight
        # warps; split latent fragments avoid the full-width shared scratch.
        block_h = 2 * heads
        _sparse_mla_fused_kernel[(main // 2, 1)](
            q,
            q,
            kv,
            union,
            lengths,
            out,
            float(sm_scale) * 1.4426950408889634,
            448.0,
            topk=2 * padded_k,
            H=2 * heads,
            KV_DIM=512,
            D_V=512,
            D_TAIL=0,
            NUM_GROUPS=4,
            STRIDE_QN_T=q.stride(0),
            STRIDE_QN_H=q.stride(1),
            STRIDE_QR_T=q.stride(0),
            STRIDE_QR_H=q.stride(1),
            USE_FP8_DOT=False,
            USE_TOPK_LENGTH=True,
            BLOCK_H=block_h,
            BLOCK_K=32,
            USE_I64_PAGE=True,
            PIPE_STAGES=1,
            union_bits_ptr=bits,
            UNION_GROUP_SIZE=2,
            BASE_HEADS=heads,
            num_warps=4 if heads == 8 else 8,
            num_stages=1,
        )
    if main < tokens:
        out[main:].copy_(
            triton_sparse_mla_fwd(
                q[main:], q[main:, :, :0], kv[:, None], indices[main:, None], sm_scale
            )[0]
        )
    return out.unsqueeze(0)
