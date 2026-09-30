"""Adapt SM70 chunked prefill to mainline's GDN interface."""

import torch
from sglang.srt.layers.attention.linear.kernels.kernel_backend import LinearAttnKernelBase


class TileLangGDNKernel(LinearAttnKernelBase):
    """Prefill adapter; decode and verify remain mainline Triton."""

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("TileLang GDN decode requires packed mixed_qkv.")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float | None = None,
        store_checkpoints: bool = True,
        **kwargs,
    ) -> tuple:
        # Mainline's non-FlashInfer contract returns the chunk-state tensor.
        # Prefix tracking can consume it without passing store_checkpoints.
        from sglang_v100_lite.gdn_chunked_tilelang import (
            chunked_gdn_sm70,
        )

        out, checkpoints = chunked_gdn_sm70(
            q,
            k,
            v,
            g,
            beta,
            scale=scale or k.shape[-1] ** -0.5,
            state=ssm_states,
            state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            store_checkpoints=store_checkpoints,
        )
        return out, None, checkpoints
