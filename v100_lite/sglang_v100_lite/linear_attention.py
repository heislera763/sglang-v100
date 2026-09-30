"""Add a TileLang GDN prefill choice through the existing dispatcher."""


def dispatcher_init(
    original, self, decode_backend, prefill_backend, verify_backend=None
):
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend

    use_tilelang = prefill_backend.is_custom()
    original(
        self,
        decode_backend,
        LinearAttnKernelBackend.TRITON if use_tilelang else prefill_backend,
        verify_backend,
    )
    if use_tilelang:
        from .gdn_tilelang import TileLangGDNKernel

        self.extend_kernel = TileLangGDNKernel()


def round_verify_state(original, *args, **kwargs):
    import torch

    states = kwargs.get("intermediate_states_buffer")
    kwargs["QUANTIZE_STATE_EACH_STEP"] = (
        states is not None
        and states.dtype != torch.float32
        and not kwargs.get("HAS_EAGLE_TREE_CUSTOM_ATTN_MASK", False)
    )
    return original(*args, **kwargs)
