"""Fail closed at guarded V100 dispatch boundaries during development."""

from sglang.srt.environ import envs


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
