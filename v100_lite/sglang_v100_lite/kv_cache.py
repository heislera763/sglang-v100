"""Fast SM70 E5M2 cache stores using the current pool's address contract."""


def store(
    original,
    self,
    layer,
    loc_info,
    key,
    value,
    k_scale=None,
    v_scale=None,
    layer_id_override=None,
    dcp_kv_mask=None,
):
    import torch

    if self.dtype == torch.float8_e5m2 and not self.use_hnd and dcp_kv_mask is None:
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
        from .kernels.sm70_fp8_kv import write_fp8_e5m2_cache_sm70

        loc, _, _ = unwrap_write_loc(loc_info)
        idx = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        ) - self.start_layer
        if write_fp8_e5m2_cache_sm70(
            key, value, self.k_buffer[idx], self.v_buffer[idx], loc, k_scale, v_scale
        ):
            return
    return original(
        self,
        layer,
        loc_info,
        key,
        value,
        k_scale,
        v_scale,
        layer_id_override,
        dcp_kv_mask,
    )
