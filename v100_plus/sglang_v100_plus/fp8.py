"""Block-FP8 weights with FP16 activations on Volta Marlin.

Retain mainline's checkpoint loading and expert mapping. Only finalization
and execution change; weights stay E4M3 and are unpacked inside the GEMM.
"""

import torch


def fp8_minimum_capability(original, config):
    if config.use_mxfp8:
        return original(config)
    return 70


def prepare_fp8_dense(original, method, layer):
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        fp8_fused_exponent_bias_into_scales,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    from .quantization import _dense_repack

    if (
        method.use_mxfp8
        or method.weight_block_size != [128, 128]
        or not method.quant_config.is_checkpoint_fp8_serialized
        or layer.orig_dtype != torch.float16
        or getattr(layer, "bias", None) is not None
    ):
        raise ValueError(
            "SM70 dense FP8 requires serialized 128x128 blocks, FP16 and no bias"
        )
    weight, block_scales = layer.weight, layer.weight_scale_inv
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if (
        weight.dtype != torch.float8_e4m3fn
        or tuple(weight.shape) != (n, k)
        or n % 128
        or k % 128
        or tuple(block_scales.shape) != (n // 128, k // 128)
    ):
        raise ValueError(
            "SM70 dense FP8 dimensions must preserve complete 128x128 blocks"
        )
    # The SM70 iterator consumes logical [K/group,N] scales, unlike the
    # SM80 dense Marlin wrapper's permuted scale layout.
    scales = fp8_fused_exponent_bias_into_scales(
        block_scales.T.repeat_interleave(128, dim=1).to(torch.float16)
    )
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("SM70 dense FP8 scales overflow FP16")
    copy_or_rebind_param(layer, "weight", _dense_repack(weight[None], num_bits=8)[0])
    copy_or_rebind_param(layer, "weight_scale_inv", scales)
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(weight.device).multi_processor_count * 4,
        device=weight.device,
        dtype=torch.int32,
    )
    layer._sm70_fp8_dense_ready = True


def apply_fp8_dense(original, method, layer, x, bias=None):
    from sglang.srt.layers.quantization.utils import get_scalar_types

    from .quantization import dense_marlin_gemm

    if not getattr(layer, "_sm70_fp8_dense_ready", False) or (
        x.dtype != torch.float16 or not x.is_cuda or bias is not None
    ):
        raise ValueError(
            "SM70 dense FP8 requires prepared weights and FP16 activations"
        )
    a = x.reshape(-1, x.shape[-1]).contiguous()
    n, k = layer.output_size_per_partition, layer.input_size_per_partition
    if a.shape[1] != k:
        raise ValueError(
            "SM70 dense FP8 activation width does not match the checkpoint"
        )
    output = dense_marlin_gemm(
        None,
        a,
        None,
        layer.weight,
        layer.weight_scale_inv,
        None,
        None,
        None,
        None,
        layer.workspace,
        get_scalar_types()[1].float8_e4m3fn,
        a.shape[0],
        n,
        k,
    )
    return output.reshape(*x.shape[:-1], n)


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


def fp8_route_block_size(original, hidden, ids, w1, w2, scales):
    """Align Qwen prefill routes to the native kernel's 32-row CTA."""
    from sglang.srt.runtime_context import get_parallel, get_schedule

    from .kernels.moe_marlin import _sm70_marlin_user_tuning

    if (
        not _sm70_marlin_user_tuning
        and get_schedule().disable_overlap_schedule
        and get_schedule().max_running_requests == 1
        and get_parallel().tp_size == get_parallel().ep_size == 4
        and get_parallel().pp_size == 2
        and hidden.dtype == scales.dtype == torch.float16
        and hidden.is_cuda
        and hidden.ndim == 2
        and hidden.shape[0] >= 128
        and hidden.shape[1] == 2560
        and ids.shape == (hidden.shape[0], 10)
        and ids.dtype == torch.int32
        and tuple(w1.shape) == (128, 160, 5120)
        and tuple(w2.shape) == (128, 40, 10240)
        and getattr(scales, "_sm70_fp8_scale", False)
    ):
        return 32
    return original(hidden, ids, w1, w2, scales)


def apply_fp8_moe(original, method, layer, dispatch_output):
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.runtime_context import get_parallel, get_schedule, get_spec

    config = method.moe_runner_config
    hidden = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    if (
        get_schedule().disable_overlap_schedule
        and get_parallel().tp_size == 4
        and (
            (
                get_spec().speculative_algorithm is None
                and get_parallel().pp_size == 2
                and hidden.shape == (1, 2560)
            )
            or (
                get_spec().speculative_algorithm == "EAGLE"
                # The one-layer draft temporarily uses a local PP group.
                and get_parallel().pp_size in (1, 2)
                and get_schedule().max_running_requests == 1
                and hidden.ndim == 2
                and 1 <= hidden.shape[0] <= 4
                and hidden.shape[1] == 2560
            )
        )
        and hidden.dtype == torch.float16
        and TopKOutputChecker.format_is_standard(topk)
        and topk.topk_ids.shape == (hidden.shape[0], 10)
        and layer.w13_weight.shape[1:] == (160, 5120)
        and layer.w2_weight.shape[1:] == (40, 10240)
        and config.is_gated
        and config.activation == "silu"
        and not config.apply_router_weight_on_input
        and config.routed_scaling_factor is None
        and config.gemm1_alpha is None
        and config.gemm1_clamp_limit is None
        and config.swiglu_limit is None
    ):
        from .kernels.sm70_fp8_moe_decode import fp8_moe_decode

        return StandardCombineInput(
            hidden_states=fp8_moe_decode(
                hidden,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                topk.topk_ids,
                topk.topk_weights,
            )
        )

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
