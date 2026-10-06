"""Select the FP16 mHC kernels for the opt-in SM70 integration."""

import torch


def mhc_pre(original, residual, *args, **kwargs):
    if residual.dtype != torch.float16 or residual.shape[1] != 4:
        return original(residual, *args, **kwargs)
    from sglang.kernels.ops.layernorm.mhc_sm70 import mhc_pre_sm70
    from sglang.srt.environ import envs

    fuse_projection = (
        residual.shape == (1, 4, 4096) and envs.SGLANG_OPT_SM70_MHC_PROJECTION.get()
    )
    fuse_pointwise = (
        residual.shape == (1, 4, 4096) and envs.SGLANG_OPT_SM70_MHC_POINTWISE.get()
    )
    return mhc_pre_sm70(
        residual,
        *args,
        fuse_projection=fuse_projection,
        fuse_pointwise=fuse_pointwise,
        **kwargs,
    )


def mhc_post(original, x, residual, post_layer_mix, comb_res_mix):
    if x.dtype != torch.float16 or residual.shape[1] != 4:
        return original(x, residual, post_layer_mix, comb_res_mix)
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h

    return mhc_post_split_h(x, residual, post_layer_mix, comb_res_mix)
