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
