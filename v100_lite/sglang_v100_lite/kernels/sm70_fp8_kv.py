"""Lazy wrapper for the optional SM70 E5M2 cache writer."""

from __future__ import annotations

from pathlib import Path

import torch

from ..dispatch import reject_fallback

_OP = None
_CHECKED = False
_SCALE_TENSORS = {}


def _get_op():
    global _OP, _CHECKED
    if _CHECKED:
        return _OP
    _CHECKED = True
    namespace = getattr(torch.ops, "sglang_sm70_turbomind", None)
    if namespace is None or not hasattr(namespace, "fp8_e5m2_cache_write"):
        from torch.utils.cpp_extension import load

        source_root = Path(__file__).parent / "csrc"
        load(
            name="sglang_v100_fp8_cache",
            sources=[
                str(source_root / "sm70_fp8_e5m2_cache.cu"),
            ],
            is_python_module=False,
            extra_cuda_cflags=["-O3"],
        )
        namespace = torch.ops.sglang_sm70_turbomind
    _OP = getattr(namespace, "fp8_e5m2_cache_write", None)
    return _OP


def _as_scale_tensor(scale, device):
    if scale is None:
        return None
    if isinstance(scale, torch.Tensor):
        return scale
    value = float(scale)
    if value == 1.0:
        return None
    key = (device.index, value)
    tensor = _SCALE_TENSORS.get(key)
    if tensor is None:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)
        _SCALE_TENSORS[key] = tensor
    return tensor


def write_fp8_e5m2_cache_sm70(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    locations: torch.Tensor,
    k_scale=None,
    v_scale=None,
) -> bool:
    op = _get_op()
    if op is None:
        reject_fallback(
            "qwen.fp8_kv_store", "native E5M2 cache operator is unavailable", key=key
        )
        return False
    if (
        key.device.type != "cuda"
        or torch.cuda.get_device_capability(key.device) != (7, 0)
        or key.dtype != torch.float16
        or value.dtype != torch.float16
        or key_cache.dtype != torch.uint8
        or value_cache.dtype != torch.uint8
        or key_cache.ndim != 3
        or value_cache.ndim != 3
        or locations.dtype != torch.int64
    ):
        reject_fallback(
            "qwen.fp8_kv_store",
            "native writer requires SM70 CUDA FP16 keys/values, 3D byte "
            "cache and int64 locations",
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            locations=locations,
        )
        return False
    for scale in (k_scale, v_scale):
        if isinstance(scale, torch.Tensor) and (
            scale.device != key.device
            or scale.dtype != torch.float32
            or scale.numel() != 1
        ):
            reject_fallback(
                "qwen.fp8_kv_store",
                "tensor scales must be FP32 scalars on the key device",
                scale=scale,
                key=key,
            )
            return False
    op(
        key,
        value,
        key_cache,
        value_cache,
        locations,
        _as_scale_tensor(k_scale, key.device),
        _as_scale_tensor(v_scale, key.device),
    )
    return True
