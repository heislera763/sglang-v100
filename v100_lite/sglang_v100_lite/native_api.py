"""Adapters for the retained SM70 AOT operator signatures."""

import torch


def moe_align_block_size(
    original,
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
    ignore_invalid_expert=False,
):
    if ignore_invalid_expert:
        raise NotImplementedError(
            "The SM70 alignment operator requires valid expert IDs"
        )
    return torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )


def moe_sum_reduce(original, input_tensor, output_tensor, routed_scaling_factor=0):
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import moe_sum_reduce_triton

    return moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)
