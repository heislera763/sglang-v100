"""Software E4M3 decoding for mapped PLE embeddings on SM70."""

import triton
import triton.language as tl


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4b15))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float16)
    if is_fp8:
        values = values * 256.0
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


import torch
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding as BaseHostEmbedding


class Qwen4ExpPinnedHostEmbedding(BaseHostEmbedding):
    def allocate_output(self, shape, device):
        return torch.empty(shape, dtype=torch.float16, device=device)

    def gather(self, input_ids, out=None):
        expected = (*input_ids.shape, self.embedding_dim)
        if out is None:
            out = self.allocate_output(expected, input_ids.device)
        if (
            tuple(out.shape) != expected
            or out.dtype != torch.float16
            or out.device != input_ids.device
        ):
            raise ValueError(
                "The SM70 PLE output must be FP16 on the input device with the expected shape"
            )
        ids = input_ids.reshape(-1).long()
        if ids.numel():
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(
                    ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                self.weight.data_ptr(),
                ids,
                out,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
            )
        return out
