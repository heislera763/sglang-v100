"""Keep the final stage's PP2 result instead of echoing it back.

The output ring still sends the sampled result to stage zero. Stage one
retains that same result/event until its existing postprocessing turn.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class _TensorRef:
    index: int


def _map_tensors(value, visit):
    """Keep Python container structure while replacing tensor leaves."""
    import torch

    if isinstance(value, (torch.Tensor, _TensorRef)):
        return visit(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, visit) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tensors(item, visit) for item in value]
    if isinstance(value, tuple):
        items = [_map_tensors(item, visit) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    return value


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
        and parallel.tp_size == 4
        and (get_model().quantization, parallel.ep_size)
        in (("fp8", 4), ("modelopt_fp4", 1))
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
            if event is not None:
                import torch

                # CPU leaves are restored with a device-to-host copy below.
                # Wait before issuing that copy, not only in later processing.
                torch.cuda.current_stream().wait_event(event)
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
    ):
        # Only the last stage samples/accepts; stage zero needs the tree and
        # commit metadata, not its full-vocabulary proposal probabilities.
        # Keep exact q in the retained result. Copy the dictionary so omitting
        # it from the wire cannot remove it from the last stage's next verify.
        tensor_dict = tensor_dict.copy()
        tensor_dict.pop("spec_next_draft_probs", None)
        import torch

        leaves = []
        _map_tensors(tensor_dict, lambda value: leaves.append(value))
        cuda_payload = any(value.is_cuda for value in leaves)
        if cuda_payload and ready_event is not None:
            torch.cuda.current_stream().wait_event(ready_event)
        tensor_dict = pack_output(tensor_dict)
        payload = tensor_dict.get("__v100_pp_output_payload__")
        if payload is not None and payload.is_cuda:
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
    """One aligned byte payload; never pickle nested CUDA storage as metadata.

    The ordinary tensor-dict transport extracts only top-level tensors. Nested
    logprob lists would otherwise unpickle on the sender's CUDA ordinal. Store
    device *types* so GPU leaves land on the receiver's own device instead.
    """
    import torch

    values = []

    def remember(value):
        values.append(value)
        return _TensorRef(len(values) - 1)

    template = _map_tensors(tensors, remember)
    if not values:
        return tensors.copy()
    cuda_values = [value for value in values if value.is_cuda]
    device = (cuda_values or values)[0].device
    padding = torch.zeros(8, dtype=torch.uint8, device=device)
    chunks, layout = [], []
    offset = 0
    for value in values:
        if (value.is_cuda and value.device != device) or value.element_size() > 8:
            raise ValueError(
                "PP output packing requires colocated tensors with <=8-byte elements"
            )
        pad = (-offset) % 8
        if pad:
            chunks.append(padding[:pad])
            offset += pad
        data = (
            value.to(device, non_blocking=True)
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
        )
        layout.append(
            (value.shape, value.dtype, offset, data.numel(), value.device.type)
        )
        chunks.append(data)
        offset += data.numel()
    pad = (-offset) % 8
    if pad:
        chunks.append(padding[:pad])
    return {
        "__v100_pp_output_template__": template,
        "__v100_pp_output_layout__": layout,
        "__v100_pp_output_payload__": torch.cat(chunks),
    }


def unpack_output(tensors):
    payload = tensors["__v100_pp_output_payload__"]
    values = []
    for shape, dtype, offset, size, device_type in tensors["__v100_pp_output_layout__"]:
        value = payload[offset : offset + size].view(dtype).reshape(shape)
        values.append(value.cpu() if device_type == "cpu" else value)
    return _map_tensors(
        tensors["__v100_pp_output_template__"], lambda ref: values[ref.index]
    )


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
