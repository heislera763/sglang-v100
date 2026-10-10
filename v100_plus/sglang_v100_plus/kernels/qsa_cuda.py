"""SM70 CUDA kernels for QSA prefill, decode and indexer scoring."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SRC_PATH = Path(__file__).resolve().parents[1] / "kernels/csrc/sm70_longctx_decode.cu"

_EXT = None
_OPS_LOAD_ATTEMPTED = False

QSA_DECODE_TARGET_CTAS = 160
QSA_DECODE_TOKENS_PER_SPLIT = 32


def _load_sm70_cuda_decode_ops():
    """Lazy-load the standalone SM70 long-context decode extension (JIT-built)."""
    global _EXT, _OPS_LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _OPS_LOAD_ATTEMPTED:
        return None
    _OPS_LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("SM70 CUDA decode partial source not found: %s", _SRC_PATH)
        return None
    from torch.utils.cpp_extension import load_inline

    build_directory = os.environ.get(
        "SGLANG_V100_DECODE_CUDA_BUILD_DIR", "/tmp/sglang_sm70_longctx_decode"
    )
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_longctx_decode_v100",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        )
    except Exception:  # pragma: no cover - build/environment failures
        logger.exception("SM70 CUDA decode partial failed to build")
        return None
    logger.info("SM70 (V100): CUDA long-context decode partial loaded.")
    return _EXT


def _cache_storage(cache):
    """Retain FP16 elements; reinterpret only the legacy E5M2 byte storage."""
    if cache.dtype == torch.float16:
        return cache.contiguous()
    if cache.dtype == torch.float8_e5m2:
        return cache.view(torch.uint8).contiguous()
    raise ValueError("SM70 QSA cache storage must be FP16 or E5M2")


def sm70_cuda_qsa_prefill(
    q,
    k_cache,
    v_cache,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    softmax_scale,
):
    """Run exact-shape QSA chunk-prefill directly from FP16 or legacy E5M2 cache."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA prefill extension is unavailable.")
    output = torch.empty_like(q)
    ext.sm70_qsa_prefill(
        q.contiguous(),
        _cache_storage(k_cache),
        _cache_storage(v_cache),
        req_to_token.to(dtype=torch.int32).contiguous(),
        req_indices.to(dtype=torch.int32).contiguous(),
        indices.to(dtype=torch.int32).contiguous(),
        seq_lens.to(dtype=torch.int32).contiguous(),
        float(softmax_scale),
        output,
    )
    return output


def sm70_cuda_qsa_decode(
    q,
    k_cache,
    v_cache,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    softmax_scale,
):
    """Run QSA split-KV decode directly from selected FP16 or legacy E5M2 cache rows."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA decode extension is unavailable.")
    batch, heads, dim = q.shape
    max_splits = max(1, math.ceil(QSA_DECODE_TARGET_CTAS / batch))
    partial_o = torch.empty(
        (batch, max_splits, heads, dim), dtype=torch.float16, device=q.device
    )
    partial_lse = torch.empty(
        (batch, max_splits, heads), dtype=torch.float32, device=q.device
    )
    seq_lens = seq_lens.to(dtype=torch.int32).contiguous()
    indices = indices.to(dtype=torch.int32).contiguous()
    ext.sm70_qsa_decode(
        q.contiguous(),
        _cache_storage(k_cache),
        _cache_storage(v_cache),
        req_to_token.to(dtype=torch.int32).contiguous(),
        req_indices.to(dtype=torch.int32).contiguous(),
        indices,
        seq_lens,
        max_splits,
        QSA_DECODE_TOKENS_PER_SPLIT,
        float(softmax_scale),
        partial_o,
        partial_lse,
    )
    if (
        1 <= batch <= 4
        and heads == 6
        and dim == 256
        and max_splits <= 160
        and torch.cuda.get_device_capability(q.device) == (7, 0)
        and os.environ.get("SGLANG_SM70_QSA_COMBINE", "1") == "1"
    ):
        from sglang_v100_plus.kernels.fusions import combine

        return combine(
            partial_o,
            partial_lse,
            seq_lens,
            indices.shape[1],
            QSA_DECODE_TOKENS_PER_SPLIT,
        )
    from .attention import _decode_combine_kernel

    combine = _decode_combine_kernel(
        batch,
        heads,
        dim,
        max_splits,
        256,
        QSA_DECODE_TOKENS_PER_SPLIT,
        selected_tokens=indices.shape[1],
    )
    return combine(partial_o, partial_lse, seq_lens)


def sm70_cuda_qsa_indexer_decode(
    q,
    k_cache,
    page_table,
    context_lens,
    max_model_len,
    score_scale,
):
    """Score compressed QSA index keys without Volta MMA head padding."""
    ext = _load_sm70_cuda_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 CUDA QSA indexer extension is unavailable.")
    logits = torch.empty(
        (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
    )
    ext.sm70_qsa_indexer_decode(
        q.contiguous(),
        k_cache.contiguous(),
        page_table.to(dtype=torch.int32).contiguous(),
        context_lens.to(dtype=torch.int32).contiguous(),
        int(max_model_len),
        float(score_scale),
        logits,
    )
    return logits
