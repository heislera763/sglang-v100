"""Capture the first PP stage's accepted recurrent/PLE state copies.

Serialized Qwen chain profiles have stable pools but new acceptance
tensors every round. Refresh those inputs before replay; never capture their
addresses directly. The last stage retains its ordinary verify commit.
"""

import copy

import torch

from sglang.srt.configs.model_config import is_qwen4_exp
from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_buffer,
    get_disagg,
    get_parallel,
    get_schedule,
    get_spec,
)


class CommitGraph:
    def __init__(
        self, original, worker, batch, accept_lens, accept_index, draft_tokens
    ):
        self.original = original
        self.worker = worker
        runner = worker.model_runner
        pool = runner.req_to_token_pool
        # Cached graph nodes borrow state addresses. Retain their owners even
        # when the worker switches pools, so identities/storage cannot be reused.
        self._owners = (pool, pool.mamba_pool, runner.attn_backend)
        self.batch = copy.copy(batch)
        self.inputs = [
            value.clone()
            for value in (batch.req_pool_indices, accept_lens, accept_index)
        ]
        self.batch.req_pool_indices = self.inputs[0]
        self.draft_tokens = draft_tokens
        self.warmups = 0
        self.graph = None
        self.capture_stream = torch.cuda.Stream()

    def _eager(self):
        self.original(self.worker, self.batch, *self.inputs[1:], self.draft_tokens)

    def run(self, batch, accept_lens, accept_index):
        for live, fixed in zip(
            (batch.req_pool_indices, accept_lens, accept_index), self.inputs
        ):
            fixed.copy_(live)
        if self.warmups < 2:
            self.warmups += 1
            self._eager()
            return
        if self.graph is None:
            stream = torch.cuda.current_stream()
            self.capture_stream.wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.capture_stream):
                self._eager()
            stream.wait_stream(self.capture_stream)
        self.graph.replay()


def commit_relayed_states(
    original,
    worker,
    batch,
    accept_lens,
    accept_index,
    draft_token_num,
    prepared_step_indices=None,
):
    parallel, schedule, spec = get_parallel(), get_schedule(), get_spec()
    pool = worker.model_runner.req_to_token_pool
    mamba_pool = getattr(pool, "mamba_pool", None)
    if not (
        envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get()
        # Stable-row PP commits derive pool slots from the refreshed request
        # indices, rather than a forward_metadata view frozen during capture.
        and envs.SGLANG_ENABLE_PP_SPEC.get()
        and is_qwen4_exp(worker.model_runner.model_config.hf_config)
        and parallel.pp_size == 2
        and parallel.pp_rank == 0
        and parallel.pp_async_batch_depth == 0
        and schedule.disable_overlap_schedule
        and schedule.max_running_requests == 1
        and get_disagg().disaggregation_mode == "null"
        and spec.speculative_algorithm == "EAGLE"
        and spec.speculative_num_steps in (1, 2, 3)
        and spec.speculative_eagle_topk == 1
        and spec.speculative_num_draft_tokens == spec.speculative_num_steps + 1
        and spec.speculative_use_rejection_sampling
        and draft_token_num == spec.speculative_num_draft_tokens
        and accept_lens.shape == batch.req_pool_indices.shape == (1,)
        and accept_index.shape == (1, draft_token_num)
        and all(
            value.is_cuda
            and value.device == accept_index.device
            and value.dtype in (torch.int32, torch.int64)
            for value in (batch.req_pool_indices, accept_lens, accept_index)
        )
        # This is a ScheduleBatch snapshot, restored to DECODE by forward
        # isolation, rather than the model's TARGET_VERIFY ForwardBatch.
        and (batch.forward_mode.is_decode() or batch.forward_mode.is_target_verify())
        and prepared_step_indices is None
        and batch.mamba_track_indices is None
        and mamba_pool is not None
        and not getattr(mamba_pool, "replayssm_spec_fold", False)
        and getattr(mamba_pool, "replayssm_cache_base", None) is None
    ):
        return original(
            worker,
            batch,
            accept_lens,
            accept_index,
            draft_token_num,
            prepared_step_indices,
        )
    states = get_buffer("v100_pp_commit_graphs", dict)
    # Captured copies use the worker's own stable state buffers, not its
    # quantized weights or TP/EP communicator. Replacement owners, chain width
    # or input layout must capture a new graph rather than replay old addresses.
    key = (
        id(worker),
        id(pool),
        id(mamba_pool),
        id(worker.model_runner.attn_backend),
        draft_token_num,
        tuple(
            (value.shape, value.dtype, value.device)
            for value in (batch.req_pool_indices, accept_lens, accept_index)
        ),
    )
    state = states.get(key)
    if state is None:
        state = states[key] = CommitGraph(
            original, worker, batch, accept_lens, accept_index, draft_token_num
        )
    state.run(batch, accept_lens, accept_index)
