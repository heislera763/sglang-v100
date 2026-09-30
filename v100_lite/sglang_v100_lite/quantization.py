"""SM70 NVFP4 layout and dispatch, retaining mainline loading and MoE APIs."""

from pathlib import Path
import os
import torch


def minimum_capability(original, cls):
    return 70


def _dense_repack(weight):
    directory = Path(os.environ["SGLANG_V100_MARLIN_DIR"])
    libraries = list(directory.glob("_C*.so"))
    if not libraries:
        raise RuntimeError("Build the SM70 dense Marlin extension before loading NVFP4")
    torch.ops.load_library(str(libraries[0]))
    e, n, packed_k = weight.shape
    k = packed_k * 2
    inputs = weight.contiguous().view(torch.int32).transpose(1, 2).contiguous()
    perm = torch.empty(0, device=weight.device, dtype=torch.int32)
    result = torch.empty((e, k // 16, n * 2), device=weight.device, dtype=torch.int32)
    for i in range(e):
        result[i] = torch.ops._C.gptq_marlin_repack(inputs[i], perm, k, n, 4, False)
    return result


def prepare_nvfp4_moe(original, layer):
    from sglang.srt.layers.utils import copy_or_rebind_param

    if layer.params_dtype != torch.float16:
        raise ValueError("The V100 NVFP4 profile requires FP16 activations")
    if hasattr(layer, "moe_runner_config") and not layer.moe_runner_config.is_gated:
        raise ValueError("The V100 NVFP4 profile requires gated MoE experts")
    if hasattr(layer, "quant_config") and layer.quant_config.group_size != 16:
        raise ValueError("The V100 NVFP4 profile requires group_size=16")
    if any(getattr(layer, name, None) is not None for name in ("w13_bias", "w2_bias")):
        raise ValueError("The V100 NVFP4 profile does not support expert bias")
    # Match mainline's Marlin workspace contract: four counters per SM.
    layer.workspace = torch.zeros(
        torch.cuda.get_device_properties(layer.w13_weight.device).multi_processor_count
        * 4,
        device=layer.w13_weight.device,
        dtype=torch.int32,
    )
    for prefix in ("w13", "w2"):
        weight = getattr(layer, prefix + "_weight")
        scales = getattr(layer, prefix + "_weight_scale")
        global_scales = getattr(layer, prefix + "_weight_scale_2")
        if global_scales.ndim == 2:
            if not torch.equal(global_scales[:, 0], global_scales[:, 1]):
                raise ValueError(
                    "Unequal NVFP4 gate/up global scales are unsupported on SM70"
                )
            global_scales = global_scales[:, 0]
        encoded, factor = sm70_nvfp4_marlin_process_scales(
            scales.transpose(1, 2).contiguous(), torch.float16
        )
        global_scales = (
            sm70_nvfp4_marlin_process_global_scale(global_scales, torch.float16)
            / factor
        )
        copy_or_rebind_param(layer, prefix + "_weight", _dense_repack(weight))
        copy_or_rebind_param(layer, prefix + "_weight_scale", encoded)
        copy_or_rebind_param(
            layer, prefix + "_weight_scale_2", global_scales.contiguous()
        )


def marlin_gemm(original, *args, **kwargs):
    from .kernels.moe_marlin import moe_wna16_marlin_gemm

    return moe_wna16_marlin_gemm(*args, **kwargs)


def moe_runner(original, cls, dispatch_name, runner_name):
    fn = original(cls, dispatch_name, runner_name)
    if dispatch_name != "none" or runner_name != "marlin" or fn is None:
        return fn

    def run(dispatch, quant, config):
        h = dispatch.hidden_states
        topk = dispatch.topk_output
        supported = (
            h.dtype == torch.float16
            and h.ndim == 2
            and 1 <= h.shape[0] <= 4
            and h.shape[1] == 2560
            and tuple(topk.topk_ids.shape) == (h.shape[0], 10)
            and tuple(quant.w13_qweight.shape) == (512, 160, 640)
            and tuple(quant.w2_qweight.shape) == (512, 10, 5120)
            and quant.w13_scales.dtype == quant.w2_scales.dtype == torch.float8_e4m3fn
            and quant.weight_bits == 4
            and quant.w13_qzeros is None
            and quant.w2_qzeros is None
            and quant.w13_global_scale is not None
            and quant.w2_global_scale is not None
            and quant.expert_map is None
            and config.num_experts == config.num_local_experts == 512
            and config.activation == "silu"
            and config.is_gated
            and not config.apply_router_weight_on_input
            and config.swiglu_limit is None
            and config.gemm1_clamp_limit is None
            and config.gemm1_alpha is None
            and config.gemm1_beta is None
            and config.routed_scaling_factor in (None, 1.0)
        )
        if supported:
            from .kernels.sm70_nvfp4_moe_decode import (
                sm70_nvfp4_moe_decode,
                sm70_nvfp4_moe_decode_enabled,
            )

            if sm70_nvfp4_moe_decode_enabled():
                from sglang.srt.layers.moe.token_dispatcher.standard import (
                    StandardCombineInput,
                )

                output = sm70_nvfp4_moe_decode(
                    h,
                    quant.w13_qweight,
                    quant.w2_qweight,
                    quant.w13_scales,
                    quant.w2_scales,
                    quant.w13_global_scale,
                    quant.w2_global_scale,
                    topk.topk_ids.view(-1),
                    topk.topk_weights.view(-1),
                )
                return StandardCombineInput(hidden_states=output)
        return fn(dispatch, quant, config)

    return run


def sm70_nvfp4_marlin_process_scales(
    scales: torch.Tensor, activation_dtype: torch.dtype
) -> tuple[torch.Tensor, float]:
    """Encode logical E4M3 NVFP4 scales for the SM70 Marlin fast path.

    The V100 iterator converts four metadata bytes to FP16/BF16 with integer
    bit operations.  It therefore consumes the special non-negative S0E5M3
    representation used by marlin_v100, not checkpoint-native S1E4M3 bytes.
    ``scales`` must already be in logical ``[E, K/16, N]`` order.

    Returns the encoded metadata and a power-of-two scale factor.  The caller
    must divide the processed global scale by that factor.
    """
    if activation_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    if scales.dtype != torch.float8_e4m3fn or scales.ndim != 3:
        raise ValueError(
            "SM70 NVFP4 Marlin expects [E,K/16,N] E4M3 scales, got "
            f"shape={tuple(scales.shape)}, dtype={scales.dtype}"
        )

    logical = scales.to(torch.float16)
    if bool((logical < 0).any()):
        raise ValueError("NVFP4 block scales must be non-negative")

    # FP16 has enough exponent range for ModelOpt's normalized E4M3 scales.
    # BF16 may need a shared rescale so every non-zero S0E5M3 byte keeps its
    # high bit set, which the integer dequantizer relies on.
    scale_factor = 1.0
    if activation_dtype == torch.bfloat16:
        nonzero = logical[logical > 0]
        if nonzero.numel() > 0:
            min_scaled = nonzero.float().min() * (2**7)
            if min_scaled < 2:
                scale_factor = float(torch.ceil(torch.log2(2 / min_scaled)).exp2())
                logical = (logical.float() * scale_factor).to(torch.float16)

    num_experts, num_groups, size_n = logical.shape
    if size_n % 4 != 0:
        raise ValueError(f"SM70 NVFP4 Marlin requires N divisible by 4, got N={size_n}")
    flat = logical.reshape(-1, size_n)
    # The iterator deliberately reverses each pair when expanding metadata.
    flat = flat.view(-1, 4)[:, [0, 2, 1, 3]].reshape(-1, size_n)
    encoded = (flat * (2**7)).view(torch.int16) << 1
    encoded = encoded.view(torch.float8_e4m3fn).reshape(-1, size_n * 2)
    encoded = encoded[:, 1::2].contiguous()
    return encoded.reshape(num_experts, num_groups, size_n), scale_factor


def sm70_nvfp4_marlin_process_global_scale(
    global_scale: torch.Tensor, activation_dtype: torch.dtype
) -> torch.Tensor:
    """Compensate an NVFP4 global scale for SM70's integer dequantizer."""
    if activation_dtype == torch.float16:
        target_exponent = 5
    elif activation_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise ValueError(
            f"SM70 NVFP4 Marlin supports FP16/BF16 activations, got {activation_dtype}"
        )
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (global_scale.to(torch.float32) * (2.0 ** (exponent_bias - 7))).contiguous()
