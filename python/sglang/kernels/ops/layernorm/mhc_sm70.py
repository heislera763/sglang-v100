"""FP16 mHC pre-mix with an FP32 projection and fused Sinkhorn finalization.

Volta has neither BF16 tensor cores nor TF32. The default learned projection
uses FP32 cuBLAS. An opt-in batch-one path fuses FP32 projection and squared
norm partials, then reduces them with the gates and Sinkhorn finalization.
Another opt-in path retains both reference reductions and fuses only the
surrounding pointwise operations.
The residual reduction shares the established mHC combine kernel.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import _hc_combine_kernel


@triton.jit
def _cast_square_kernel(X, FLOAT_X, SQUARE, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, 0).to(tl.float32)
    tl.store(FLOAT_X + i, x, i < SIZE)
    tl.store(SQUARE + i, x * x, i < SIZE)


@triton.jit
def _project_rms_kernel(
    RESIDUAL,
    FN,
    PARTIALS,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FP32 projection and squared-norm partials for a single decode row."""
    split = tl.program_id(0)
    group = tl.program_id(1)
    k = split * BLOCK_K + tl.arange(0, BLOCK_K)
    n = group * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(RESIDUAL + k, k < K, 0).to(tl.float32)
    w = tl.load(
        FN + n[:, None] * K + k[None, :], (n[:, None] < 24) & (k[None, :] < K), 0
    )
    projection = tl.sum(w * x[None, :], 1)
    tl.store(PARTIALS + split * 25 + n, projection, n < 24)
    if group == 0:
        tl.store(PARTIALS + split * 25 + 24, tl.sum(x * x, 0))


@triton.jit
def _finalize_kernel(
    MIXES,
    RMS,
    SCALE,
    BASE,
    PRE,
    POST,
    COMB,
    HC: tl.constexpr,
    PRE_EPS: tl.constexpr,
    SINK_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    ITERS: tl.constexpr,
    PROJECTION_SPLITS: tl.constexpr = 0,
    PROJECTION_K: tl.constexpr = 0,
    RMS_EPS: tl.constexpr = 0.0,
    RMS_FROM_MEAN: tl.constexpr = False,
):
    row = tl.program_id(0)
    j = tl.arange(0, HC)
    jj, kk = j[:, None], j[None, :]
    if PROJECTION_SPLITS:
        split = tl.arange(0, PROJECTION_SPLITS)
        valid = split < PROJECTION_SPLITS
        squares = tl.load(MIXES + split * 25 + 24, valid, 0)
        rms = tl.rsqrt(tl.sum(squares, 0) / PROJECTION_K + RMS_EPS)
        pre_raw = (
            tl.sum(
                tl.load(MIXES + split[:, None] * 25 + j[None, :], valid[:, None], 0), 0
            )
            * rms
        )
        post_raw = (
            tl.sum(
                tl.load(
                    MIXES + split[:, None] * 25 + HC + j[None, :], valid[:, None], 0
                ),
                0,
            )
            * rms
        )
        comb_raw = (
            tl.sum(
                tl.load(
                    MIXES
                    + split[:, None, None] * 25
                    + 2 * HC
                    + jj[None, :, :] * HC
                    + kk[None, :, :],
                    valid[:, None, None],
                    0,
                ),
                0,
            )
            * rms
        )
    else:
        mix = MIXES + row * (2 + HC) * HC
        rms = tl.load(RMS + row)
        if RMS_FROM_MEAN:
            rms = tl.rsqrt(rms + RMS_EPS)
        pre_raw = tl.load(mix + j) * rms
        post_raw = tl.load(mix + HC + j) * rms
        comb_raw = tl.load(mix + 2 * HC + jj * HC + kk) * rms
    pre = tl.sigmoid(pre_raw * tl.load(SCALE) + tl.load(BASE + j)) + PRE_EPS
    post = POST_MULT * tl.sigmoid(
        post_raw * tl.load(SCALE + 1) + tl.load(BASE + HC + j)
    )
    comb = comb_raw * tl.load(SCALE + 2) + tl.load(BASE + 2 * HC + jj * HC + kk)
    comb = tl.exp(comb - tl.max(comb, 1)[:, None])
    comb = comb / tl.sum(comb, 1)[:, None] + SINK_EPS
    comb = comb / (tl.sum(comb, 0)[None, :] + SINK_EPS)
    for _ in tl.static_range(ITERS - 1):
        comb = comb / (tl.sum(comb, 1)[:, None] + SINK_EPS)
        comb = comb / (tl.sum(comb, 0)[None, :] + SINK_EPS)
    tl.store(PRE + row * HC + j, pre)
    tl.store(POST + row * HC + j, post)
    tl.store(COMB + row * HC * HC + jj * HC + kk, comb)


def mhc_pre_sm70(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    *,
    fuse_projection=False,
    fuse_pointwise=False,
):
    """Return (FP32 post gates, FP32 residual mixing, FP16 layer input).

    HC=4 contiguous FP16 residuals and contiguous FP32 parameters are required.
    Epsilons and the post multiplier retain the torch fallback's semantics.
    ``fuse_projection`` supports only (M, HC, H) = (1, 4, 4096); its split
    FP32 summation has a different reduction order from cuBLAS/torch.
    ``fuse_pointwise`` retains cuBLAS projection and torch mean reduction,
    fusing only cast/square and epsilon/rsqrt operations for the same shape.
    """
    assert residual.ndim == 3 and residual.shape[1] == 4
    assert residual.dtype == torch.float16 and residual.is_contiguous()
    assert all(
        t.dtype == torch.float32 and t.is_contiguous() for t in (fn, hc_scale, hc_base)
    )
    m, hc, h = residual.shape
    assert fn.shape == (24, hc * h)
    assert hc_scale.numel() == 3 and hc_base.numel() == 24
    assert sinkhorn_repeat >= 1
    assert not fuse_projection or (m == 1 and h == 4096)
    assert not fuse_pointwise or (m == 1 and h == 4096)
    assert not (fuse_projection and fuse_pointwise)
    pre = torch.empty((m, hc), dtype=torch.float32, device=residual.device)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc, hc), dtype=torch.float32, device=residual.device)
    layer_input = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    if m:
        splits = 0
        if fuse_projection:
            splits = triton.cdiv(hc * h, 1024)
            mixes = torch.empty(
                (splits, 25), dtype=torch.float32, device=residual.device
            )
            rms = mixes  # The fused finalizer reads norm partials from mixes.
            _project_rms_kernel[(splits, 6)](
                residual,
                fn,
                mixes,
                K=hc * h,
                BLOCK_K=1024,
                BLOCK_N=4,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif fuse_pointwise:
            x = torch.empty((m, hc * h), dtype=torch.float32, device=residual.device)
            square = torch.empty_like(x)
            _cast_square_kernel[(triton.cdiv(m * hc * h, 1024),)](
                residual,
                x,
                square,
                SIZE=m * hc * h,
                BLOCK=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
            # Identical input layout and reduction implementation to the default.
            rms = square.mean(-1)
            mixes = F.linear(x, fn)
        else:
            x = residual.view(m, hc * h).float()
            rms = torch.rsqrt(x.square().mean(-1) + rms_eps)
            mixes = F.linear(x, fn)
        _finalize_kernel[(m,)](
            mixes,
            rms,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            HC=hc,
            PRE_EPS=hc_pre_eps,
            SINK_EPS=hc_sinkhorn_eps,
            POST_MULT=hc_post_mult_value,
            ITERS=sinkhorn_repeat,
            PROJECTION_SPLITS=splits,
            PROJECTION_K=hc * h,
            RMS_EPS=rms_eps,
            RMS_FROM_MEAN=fuse_pointwise,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _hc_combine_kernel[(m, triton.cdiv(h, 256))](
            residual,
            pre,
            layer_input,
            h,
            hc * h,
            hc,
            1,
            h,
            HC=hc,
            BLOCK_H=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return post.unsqueeze(-1), comb, layer_input
