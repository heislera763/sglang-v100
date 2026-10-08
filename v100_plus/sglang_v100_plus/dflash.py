"""Volta primary operations and block-FP8 loading for DFlash2 drafts."""

import torch
from torch import nn

from sglang.srt.models.dflash import DFlashGroupedConv, _grouped_conv

from .dispatch import reject_fallback


class SM70DFlashGroupedConv(DFlashGroupedConv):
    def finish(self, hidden_states, coefficients):
        # Real GLM features overflow FP16 at the first output convolution.
        # Retain FP32 through its residual addition and the following norm.
        return _grouped_conv(
            hidden_states.float(),
            coefficients.float(),
            self.base_kernel[1].float(),
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )


class SM70DFlashResidualNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon

    def forward(self, hidden_states, residual=None):
        summed = hidden_states.float()
        if residual is not None:
            summed = summed + residual.float()
        inv_rms = torch.rsqrt(
            summed.square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        output = (summed * inv_rms * self.weight.float()).to(torch.float16)
        return output if residual is None else (output, summed)


def candidate_topk(original, scores, k):
    if (
        scores.is_cuda
        and scores.ndim == 2
        and scores.dtype in (torch.float16, torch.float32)
        and 0 < k <= scores.shape[1]
        and torch.cuda.get_device_capability(scores.device) == (7, 0)
    ):
        # FlashInfer rejects SM70. CUDA Torch top-k is the selected Volta
        # backend, with the same sorted values/indices interface.
        return torch.topk(scores, k, dim=-1, largest=True, sorted=True)
    reject_fallback(
        "dflash.candidate_topk",
        "Volta CUDA FP16/FP32 score rows are required",
        scores=scores,
    )
    return original(scores, k)


def initialize_draft(original, self, config, quant_config=None, prefix=""):
    if quant_config is not None:
        if quant_config.get_name() != "fp8" or quant_config.weight_block_size != [
            128,
            128,
        ]:
            raise ValueError(
                "SM70 DFlash2 supports unquantized or 128x128 block-FP8 draft weights"
            )
        # Mainline already builds quant-aware transformer projections. Enable
        # its replicated quant-aware context projection and weight prefixes as
        # well, so fc.weight_scale_inv is loaded instead of silently ignored.
        self.supports_quantization = True
    result = original(self, config, quant_config=quant_config, prefix=prefix)
    self.norm = SM70DFlashResidualNorm(self.norm)
    for layer in self.layers:
        layer.input_layernorm = SM70DFlashResidualNorm(layer.input_layernorm)
        layer.post_attention_layernorm = SM70DFlashResidualNorm(
            layer.post_attention_layernorm
        )
    return result


def forward_draft(
    original, self, input_ids, positions, forward_batch, input_embeds=None, **kwargs
):
    # EagerRunner's supplied-embedding kwargs currently round through BF16.
    # DFlash keeps the original embeddings on ForwardBatch; use those just as
    # the FP16 graph buffers do, without discarding FP16 mantissa bits first.
    if forward_batch.input_embeds is not None:
        input_embeds = forward_batch.input_embeds
    if input_embeds is not None:
        input_embeds = input_embeds.to(torch.float16)
    return original(
        self, input_ids, positions, forward_batch, input_embeds=input_embeds, **kwargs
    )
