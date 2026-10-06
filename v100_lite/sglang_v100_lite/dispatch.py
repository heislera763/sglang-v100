"""Fail closed at guarded V100 dispatch boundaries during development."""

from contextlib import contextmanager
from contextvars import ContextVar

from sglang.srt.environ import envs

# The supported Volta profile uses eager prefill. Keep this plugin-local,
# thread/nesting-safe scope separate from DP's is_extend_in_batch flag.
_prefill = ContextVar("v100_eager_prefill", default=False)


def in_prefill() -> bool:
    return _prefill.get()


@contextmanager
def prefill_scope(enabled: bool):
    token = _prefill.set(enabled)
    try:
        yield
    finally:
        _prefill.reset(token)


def eager_extend(original, self, forward_batch, *args, **kwargs):
    # Eager extend also executes target verification: that must stay strict
    # about small-kernel coverage, rather than being treated as prefill.
    with prefill_scope(forward_batch.forward_mode.is_extend_without_speculative()):
        return original(self, forward_batch, *args, **kwargs)


class V100FallbackError(RuntimeError):
    """A guarded V100 implementation would have delegated to its fallback."""


def reject_fallback(operation: str, reason: str, **tensors) -> None:
    """Check only on a fallback branch; inspect metadata without GPU reads."""
    if not envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.get():
        return
    details = []
    for name, tensor in tensors.items():
        if tensor is None:
            details.append(f"{name}=None")
        else:
            details.append(
                f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"device={tensor.device}, stride={tensor.stride()}, "
                f"contiguous={tensor.is_contiguous()}"
            )
    metadata = " " + "; ".join(details) + "." if details else ""
    raise V100FallbackError(
        f"V100 strict dispatch rejected fallback for {operation}: {reason}."
        + metadata
        + " Add native coverage or explicitly select a supported SM70 backend; "
        "development tests must keep SGLANG_DEBUG_V100_STRICT_DISPATCH=1."
    )
