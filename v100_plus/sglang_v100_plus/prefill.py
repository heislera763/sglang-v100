"""Partition replicated Qwen HC work by token within each four-rank group.

Attention and experts still receive the complete token matrix. Residual streams
stay partitioned between HC calls; only the mixed input is gathered per branch.
PLE's causal convolution and the PP/last-HC outputs retain their full layout.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
)

from .dispatch import in_prefill


@dataclass(frozen=True)
class Partition:
    rows: int
    rank: int
    size: int = 4
    group: object = None

    @property
    def local_rows(self):
        return (self.rows + self.size - 1) // self.size

    def local(self, x):
        if x.shape[0] == self.rows:
            start = self.rank * self.local_rows
            local = x[start : start + self.local_rows].contiguous()
            if local.shape[0] != self.local_rows:
                import torch.nn.functional as F

                # Only replicated HC work is padded. Gather trims these rows
                # before attention, causal PLE, PP output or final logits.
                pad = (0, 0) * (x.ndim - 1) + (0, self.local_rows - local.shape[0])
                local = F.pad(local, pad)
            return local
        if x.shape[0] != self.local_rows:
            raise ValueError("HC rows do not match the prefill token partition")
        return x


def _slot():
    return get_buffer(
        "v100_hc_prefill_partition",
        lambda: ContextVar("v100_hc_prefill_partition", default=None),
    )


@contextmanager
def partition_scope(partition):
    slot = _slot()
    token = slot.set(partition)
    try:
        yield
    finally:
        slot.reset(token)


def local_input(x):
    partition = _slot().get()
    return x if partition is None else partition.local(x)


def gather_input(x):
    partition = _slot().get()
    if partition is None:
        return x
    if x.shape[0] != partition.local_rows:
        raise ValueError("HC gather requires one rank's token partition")
    group = partition.group
    if group is None:
        group = get_parallel().attn_tp_group
    gathered = group.all_gather(x.contiguous(), dim=0)
    return (
        gathered if gathered.shape[0] == partition.rows else gathered[: partition.rows]
    )


def _quad_group():
    """Create eager HC-only groups; retain the parent groups for their lifetime."""
    import torch.distributed as dist

    from sglang.srt.distributed import init_model_parallel_group

    parallel = get_parallel()
    parent, world = parallel.attn_tp_group, parallel.world_group
    if parent is None or world is None or len(parent.ranks) != 8:
        raise RuntimeError("TP8 HC partition requires initialized eight-rank groups")
    if parent.ranks != world.ranks:
        raise RuntimeError("TP8 HC partition requires one complete TP8 world")
    groups = get_buffer("v100_hc_prefill_quad_groups", dict)
    key = (id(parent), id(world))
    if key not in groups:
        # Every world rank creates both groups in the same order. Launch ranks
        # must place each NVLink quad consecutively; model TP/EP remain intact.
        group = init_model_parallel_group(
            [parent.ranks[:4], parent.ranks[4:]],
            local_rank=world.local_rank,
            backend=dist.get_backend(parent.device_group),
            use_pynccl=False,
            use_custom_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
            group_name="v100_hc_prefill",
        )
        groups[key] = (parent, world, group)
    return groups[key][2]


def _partition(self, input_ids, forward_batch):
    parallel = get_parallel()
    if (
        envs.SGLANG_OPT_SM70_HC_PREFILL_SP.get()
        and in_prefill()
        and forward_batch.forward_mode.is_extend()
        and input_ids.shape[0] >= 256
        and forward_batch.batch_size == 1
        and self.hc_count == 4
        and self.hidden_size == 2560
        and get_schedule().disable_overlap_schedule
        and get_schedule().max_running_requests == 1
        and get_spec().speculative_algorithm in (None, "EAGLE")
        and get_exec().graph.disable_prefill_cuda_graph
        and not get_exec().features.enable_return_hidden_states
        and get_exec().features.return_hidden_states_mode is None
    ):
        # HC's FP16 token-row operations are independent of weight format.
        if parallel.tp_size == parallel.attn_tp_size == 4 and parallel.pp_size in (
            1,
            2,
        ):
            return Partition(input_ids.shape[0], parallel.attn_tp_rank)
        if parallel.tp_size == parallel.attn_tp_size == 8 and parallel.pp_size == 1:
            group = _quad_group()
            return Partition(input_ids.shape[0], group.rank_in_group, group=group)
    return None


def model_forward(original, self, input_ids, positions, forward_batch, *args, **kwargs):
    partition = _partition(self, input_ids, forward_batch)
    with partition_scope(partition):
        result = original(self, input_ids, positions, forward_batch, *args, **kwargs)
        if partition is not None:
            if not self.pp_group.is_last_rank:
                result.tensors["hidden_states"] = gather_input(result["hidden_states"])
            elif isinstance(result, tuple):
                result = result[0], gather_input(result[1])
        return result


def prepare_attention(original, self, hidden_states, forward_batch, ple_batch):
    partition = _slot().get()
    if (
        partition is not None
        and self.ple is not None
        and hidden_states.shape[0] == partition.local_rows
    ):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        # PLE consumes a completed FFN write. Transfer it through the boundary
        # accessor so gathering does not leave a stale written-residual identity
        # or discard an outstanding producer contribution.
        hidden_states = residual_batch.take_output(hidden_states, forward_batch)
        hidden_states = residual_batch.set_written(
            gather_input(hidden_states), forward_batch
        )
    return original(self, hidden_states, forward_batch, ple_batch)
