"""FP16 hyperconnections and small-row V100 projections."""

import torch
import msgspec
import os


def initialize(original, self, config, *args, **kwargs):
    config = msgspec.structs.replace(config, params_dtype=torch.float16)
    return original(self, config, *args, **kwargs)


def mix(original, self, hyper_input):
    if (
        hyper_input.ndim != 2
        or hyper_input.shape[0] not in (1, 2, 4)
        or hyper_input.shape[1] != 10240
        or hyper_input.dtype != torch.float16
    ):
        return original(self, hyper_input)
    if self.config.hc_per_branch_norm:
        normed = self.hc_norm(hyper_input)
    else:
        normed = self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)
    if not sm70_hc_down_gemv_silu_supported(
        normed, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight
    ):
        return original(self, hyper_input)
    from .kernels.sm70_hc_mix import gate_supported, hc_down_with_gate, hc_down, hc_up

    gate = None
    if getattr(self, "_split_combine_ok", False) and gate_supported(
        normed, self.block_inject_weight.weight
    ):
        down, gate = hc_down_with_gate(
            normed, self.input_mix_weight_down.weight, self.block_inject_weight.weight
        )
    else:
        down = hc_down(normed, self.input_mix_weight_down.weight)
    result = hc_up(down, normed, self.input_mix_weight_up.weight).to(self.params_dtype)
    state = (hyper_input, normed) if gate is None else (hyper_input, normed, gate)
    return result, state


def combine(original, self, block_output, residuals):
    if len(residuals) == 3:
        from .kernels.sm70_hc_mix import hc_apply_gate

        return hc_apply_gate(block_output, residuals[0], residuals[2])
    return original(self, block_output, residuals)


def embedding_output(original, self, *args, **kwargs):
    return original(self, *args, **kwargs).to(torch.float16)


def sm70_hc_down_gemv_silu_supported(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (7, 0)
        and x.dtype == torch.float16
        and w_down.dtype == torch.float16
        and w_up.dtype == torch.float16
        and w_down.device == x.device
        and w_up.device == x.device
        and x.ndim == 2
        and x.shape[1] == 10240
        and (
            x.shape[0] == 1
            or (
                x.shape[0] in (2, 4)
                and os.environ.get("SGLANG_SM70_HC_NATIVE", "1") == "1"
                and os.environ.get("SGLANG_SM70_MTP_HC", "1") == "1"
                and x.data_ptr() % 16 == 0
                and w_down.data_ptr() % 16 == 0
                and w_up.data_ptr() % 16 == 0
            )
        )
        and w_down.shape == (320, 10240)
        and w_up.shape == (10240, 320)
        and x.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )
