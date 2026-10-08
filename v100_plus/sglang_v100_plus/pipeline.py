"""Keep the final stage's ordinary PP2 result instead of echoing it back.

The output ring still sends the sampled result to stage zero. Stage one
retains that same result/event until its existing postprocessing turn.
"""

from collections import deque


def initialize_local_output(original, scheduler):
    from sglang.srt.runtime_context import get_disagg, get_parallel, get_schedule

    original(scheduler)
    parallel = get_parallel()
    schedule = get_schedule()
    enabled = (
        parallel.pp_size == 2
        and parallel.pp_async_batch_depth == 0
        and schedule.disable_overlap_schedule
        and schedule.max_running_requests == 1
        and get_disagg().disaggregation_mode == "null"
        and scheduler.spec_algorithm.is_none()
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
    if local is None or not scheduler.pp_group.is_last_rank:
        return original(scheduler)
    event, proxy = local.popleft()
    # The caller's copy stream waits for the original forward/sampling event.
    # Keep all fields (logprobs, masks, auxiliary output), not only token IDs.
    return proxy.tensors, event
