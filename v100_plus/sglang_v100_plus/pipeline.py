"""Keep the final stage's PP2 result instead of echoing it back.

The output ring still sends the sampled result to stage zero. Stage one
retains that same result/event until its existing postprocessing turn.
"""

from collections import deque


def initialize_local_output(original, scheduler):
    from sglang.srt.runtime_context import (
        get_disagg,
        get_model,
        get_parallel,
        get_schedule,
        get_spec,
    )

    original(scheduler)
    parallel = get_parallel()
    schedule = get_schedule()
    spec = get_spec()
    local_spec = (
        spec.speculative_algorithm == "EAGLE"
        and get_model().quantization == "fp8"
        and parallel.tp_size == parallel.ep_size == 4
        and spec.speculative_eagle_topk == 1
        and spec.speculative_num_steps in (1, 2, 3)
        and spec.speculative_num_draft_tokens == spec.speculative_num_steps + 1
        and spec.speculative_use_rejection_sampling
    )
    enabled = (
        parallel.pp_size == 2
        and parallel.pp_async_batch_depth == 0
        and schedule.disable_overlap_schedule
        and schedule.max_running_requests == 1
        and get_disagg().disaggregation_mode == "null"
        and (scheduler.spec_algorithm.is_none() or local_spec)
    )
    scheduler._v100_pp_local_outputs = deque() if enabled else None


def send_output(
    original, scheduler, next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs
):
    local = scheduler._v100_pp_local_outputs
    if local is None:
        return original(
            scheduler, next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs
        )
    if not scheduler.pp_group.is_last_rank:
        # The final stage already owns this result. It consumes its retained
        # copy at the same microbatch postprocessing turn as the old echo.
        return []
    retained = (
        last_rank_comm_queue[0] if mbs[next_first_rank_mb_id] is not None else None
    )
    work = original(
        scheduler, next_first_rank_mb_id, mbs, last_rank_comm_queue, pp_outputs
    )
    if retained is not None and work:
        local.append(retained)
    return work


def receive_output(original, scheduler):
    local = scheduler._v100_pp_local_outputs
    if local is None:
        return original(scheduler)
    if not scheduler.pp_group.is_last_rank:
        tensors, event = original(scheduler)
        if "__v100_pp_output_layout__" in tensors:
            tensors = unpack_output(tensors)
        return tensors, event
    event, proxy = local.popleft()
    # The caller's copy stream waits for the original forward/sampling event.
    # Keep all fields (logprobs, masks, auxiliary output), not only token IDs.
    # PP+spec receives before sending, but consumes the PREVIOUS microbatch's
    # retained output; the current send queue need not contain any result.
    return proxy.tensors, event


def send_output_dict(
    original,
    scheduler,
    tensor_dict,
    async_send=True,
    msg_type="default",
    ready_event=None,
):
    if (
        scheduler._v100_pp_local_outputs is not None
        and not scheduler.spec_algorithm.is_none()
        and scheduler.pp_group.is_last_rank
        and msg_type == "output"
        and "spec_next_draft_probs" in tensor_dict
    ):
        # Only the last stage samples/accepts; stage zero needs the tree and
        # commit metadata, not its full-vocabulary proposal probabilities.
        # Keep exact q in the retained result. Copy the dictionary so omitting
        # it from the wire cannot remove it from the last stage's next verify.
        tensor_dict = tensor_dict.copy()
        del tensor_dict["spec_next_draft_probs"]
        import torch

        cuda_payload = any(
            value.is_cuda
            for value in tensor_dict.values()
            if isinstance(value, torch.Tensor)
        )
        if cuda_payload and ready_event is not None:
            torch.cuda.current_stream().wait_event(ready_event)
        tensor_dict = pack_output(tensor_dict)
        if tensor_dict["__v100_pp_output_payload__"].is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record()
    return original(
        scheduler,
        tensor_dict,
        async_send=async_send,
        msg_type=msg_type,
        ready_event=ready_event,
    )


def pack_output(tensors):
    """One byte payload; align every typed view without changing its bits."""
    import torch

    values = [
        (key, value)
        for key, value in tensors.items()
        if isinstance(value, torch.Tensor)
    ]
    device = values[0][1].device
    padding = torch.zeros(8, dtype=torch.uint8, device=device)
    chunks, layout = [], []
    offset = 0
    result = {
        key: value
        for key, value in tensors.items()
        if not isinstance(value, torch.Tensor)
    }
    for key, value in values:
        if value.device != device or value.element_size() > 8:
            raise ValueError(
                "PP output packing requires colocated tensors with <=8-byte elements"
            )
        pad = (-offset) % 8
        if pad:
            chunks.append(padding[:pad])
            offset += pad
        data = value.contiguous().reshape(-1).view(torch.uint8)
        layout.append((key, value.shape, value.dtype, offset, data.numel()))
        chunks.append(data)
        offset += data.numel()
    pad = (-offset) % 8
    if pad:
        chunks.append(padding[:pad])
    result["__v100_pp_output_layout__"] = layout
    result["__v100_pp_output_payload__"] = torch.cat(chunks)
    return result


def unpack_output(tensors):
    result = tensors.copy()
    layout = result.pop("__v100_pp_output_layout__")
    payload = result.pop("__v100_pp_output_payload__")
    for key, shape, dtype, offset, size in layout:
        result[key] = payload[offset : offset + size].view(dtype).reshape(shape)
    return result


def set_local_relay(original, scheduler, batch, relayed):
    from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput

    current = batch.spec_info
    if (
        scheduler._v100_pp_local_outputs is not None
        and not scheduler.spec_algorithm.is_none()
        and [req.rid for req in batch.reqs] == relayed.rids
        and (
            not isinstance(current, PPSpecRelayInput)
            or (
                current.rids == relayed.rids
                and (relayed.parents is not None or current.parents is None)
                and (relayed.draft_probs is not None or current.draft_probs is None)
            )
        )
    ):
        # Every live row is replaced. The old relay has no row to preserve,
        # so adopt/index/where would only copy the same tree and exact q.
        # Tail-draft q already owns a clone of the draft graph's output.
        batch.spec_info = relayed
        return
    return original(scheduler, batch, relayed)
