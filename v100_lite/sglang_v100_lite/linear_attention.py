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
