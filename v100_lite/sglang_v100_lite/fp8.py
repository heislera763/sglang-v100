"""Block-FP8 weights with FP16 activations on Volta Marlin.

Retain mainline's checkpoint loading and expert mapping. Only finalization
and execution change; weights stay E4M3 and are unpacked inside the GEMM.
"""

import torch


def fp8_minimum_capability(original, config):
    if config.use_mxfp8:
        return original(config)
    return 70


def _validate(method):
    from sglang.srt.layers.moe import get_moe_runner_backend

    if (
        method.use_mxfp8
        or method.is_fp4_expert
        or method.weight_block_size != [128, 128]
        or not method.quant_config.is_checkpoint_fp8_serialized
        or not get_moe_runner_backend().is_marlin()
    ):
        raise ValueError(
            "SM70 FP8 experts require serialized E4M3 blocks [128,128] and Marlin"
        )


def create_fp8_moe_runner(original, method, layer, config):
    from sglang.srt.layers.moe.moe_runner import MoeRunner
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    _validate(method)
    method.moe_runner_config = config
    method._owns_moe_runner = True
    method.runner = MoeRunner(MoeRunnerBackend.MARLIN, config)


def prepare_fp8_moe(original, method, layer):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    from .quantization import _dense_repack

    _validate(method)
    if layer.params_dtype != torch.float16:
        raise ValueError("SM70 FP8 experts require FP16 activations")
    if any(
        getattr(layer, n, None) is not None
        for n in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("SM70 FP8 expert bias is unsupported")
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        block_scales = getattr(layer, prefix + "_weight_scale_inv")
        e, n, k = weight.shape
        if weight.dtype != torch.float8_e4m3fn or n % 128 or k % 128:
            raise ValueError(
                "SM70 FP8 expert dimensions must preserve complete 128x128 blocks"
            )
        if tuple(block_scales.shape) != (e, n // 128, k // 128):
            raise ValueError("Invalid block-FP8 expert scale shape")
        packed = _dense_repack(weight, num_bits=8)
        scales = torch.empty(
            (e, k // 128, n), device=weight.device, dtype=torch.float16
        )
        for i in range(e):
            logical = block_scales[i].T.repeat_interleave(128, dim=1).to(torch.float16)
            # Unlike mainline Marlin, the SM70 iterator consumes logical
            # [K/group,N] scale order, without the SM80 tile permutation.
            scales[i] = fp8_fused_exponent_bias_into_scales(logical)
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("SM70 FP8 scales overflow FP16")
        copy_or_rebind_param(layer, prefix + "_weight", packed)
        name = prefix + "_weight_scale_inv"
        copy_or_rebind_param(layer, name, scales)
        # The shared Marlin runner otherwise interprets eight-bit storage as
        # integer quantization. Preserve the floating-point format explicitly.
        getattr(layer, name)._sm70_fp8_scale = True


def fp8_marlin_scalar_type(original, num_bits, has_zp, scales=None, global_scale=None):
    if getattr(scales, "_sm70_fp8_scale", False):
        from sglang.srt.layers.quantization.utils import get_scalar_types

        if num_bits != 8 or has_zp or global_scale is not None:
            raise ValueError("Invalid SM70 FP8 Marlin metadata")
        return get_scalar_types()[1].float8_e4m3fn
    return original(num_bits, has_zp, scales, global_scale)


def apply_fp8_moe(original, method, layer, dispatch_output):
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

    mapping = layer.dispatcher.local_expert_mapping
    quant = MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale_inv,
        w2_scales=layer.w2_weight_scale_inv,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=8,
        expert_map=mapping,
        # StandardDispatcher has already translated global IDs to local IDs
        # (or -1). Alignment needs only the stored experts, not empty global
        # expert slots on every rank.
        global_num_experts=layer.w13_weight.shape[0] if mapping is not None else -1,
    )
    return method.runner.run(dispatch_output, quant)
