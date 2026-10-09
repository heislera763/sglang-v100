"""Leave cold Triton module loads access to idle Torch allocator blocks."""

import torch

from sglang.srt.runtime_context import get_exec


def reclaim_for_triton_load(_module, _function, _name, _metadata_group, _hash):
    # cuModuleLoadData allocates outside Torch. Its OOM cannot trigger the
    # caching allocator's usual retry/reclamation, even when live tensors fit.
    # This callback runs on first device-load, never on warmed kernel replay.
    if get_exec().features.enable_memory_saver:
        return  # Custom allocation arenas have their own segment ownership.
    if torch.cuda.is_current_stream_capturing():
        return
    free, _ = torch.cuda.mem_get_info()
    if free >= 1 << 30:
        return
    idle = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if idle < 64 << 20:
        return
    # Only unused allocator blocks are returned; live state/graph tensors and
    # the selected kernel remain unchanged. A real capacity OOM still raises.
    torch.cuda.empty_cache()


def install():
    import triton.knobs as knobs

    # HookChain.add is idempotent and preserves other first-load callbacks.
    knobs.runtime.kernel_load_start_hook.add(reclaim_for_triton_load)
