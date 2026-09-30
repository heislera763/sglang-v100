"""Opt-in SM70 gemm support."""

import os

import torch

from sglang_v100_lite.kernels.utils import cache_once, load_jit

# (output features, input features) -> (threads, lanes per row, vector width).
_DENSE_CONFIGS = {
    (4096, 2560): (64, 16, 8),
    (3584, 2560): (64, 32, 8),
    (2560, 1536): (128, 32, 8),
    (320, 2560): (256, 32, 8),
    (2560, 160): (128, 8, 4),
    (512, 2560): (256, 32, 8),
    (640, 2560): (128, 32, 8),
    (24, 2560): (64, 32, 8),
    (1, 2560): (256, 32, 8),
    (10240, 2560): (64, 32, 8),
    (2560, 2560): (128, 32, 8),
}


def supported(x: torch.Tensor, weight: torch.Tensor, bias=None) -> bool:
    from sglang_v100_lite.kernels.gemm import shape_supported

    return (
        os.environ.get("SGLANG_SM70_DENSE_GEMV", "0") == "1"
        and x.is_cuda
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and weight.device == x.device
        and x.ndim == 2
        and x.shape[0] in (1, 2, 4)
        and weight.ndim == 2
        and x.shape[1] == weight.shape[1]
        and bias is None
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and (
            (
                x.shape[0] == 1
                and (
                    tuple(weight.shape) in _DENSE_CONFIGS
                    or (weight.shape[1] == 2560 and weight.shape[0] >= 32768)
                )
            )
            or shape_supported(x, weight)
        )
        and torch.cuda.get_device_capability(x.device) == (7, 0)
    )


@cache_once
def _dense_module(threads: int, lanes: int, vector: int):
    return load_jit(
        "sm70_dense_gemv",
        threads,
        lanes,
        vector,
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[
            ("gemv", f"sglang::sm70_dense_gemv::gemv<{threads},{lanes},{vector}>")
        ],
    )


def linear_dense(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] != 1:
        from sglang_v100_lite.kernels.gemm import linear_small as small_linear

        return small_linear(x, weight)
    config = _DENSE_CONFIGS.get(tuple(weight.shape), (64, 32, 8))
    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _dense_module(*config).gemv(x, weight, out)
    return out

import os

import torch

from sglang_v100_lite.kernels.utils import cache_once, load_jit

# (rows, output features, input features) -> (threads, lanes per output).
# Shapes where cuBLAS is faster deliberately stay on the existing path.
_SMALL_CONFIGS = {
    (2, 24, 2560): (64, 32),
    (2, 640, 2560): (256, 32),
    (2, 1, 2560): (64, 32),
    (2, 2560, 2560): (128, 32),
    (4, 24, 2560): (64, 32),
    (4, 640, 2560): (128, 32),
    (4, 2560, 2560): (64, 32),
    (2, 4096, 2560): (256, 32),
    (4, 4096, 2560): (64, 32),
    (2, 3584, 2560): (64, 32),
    (4, 3584, 2560): (64, 32),
    (2, 2560, 1536): (64, 32),
    (4, 2560, 1536): (128, 16),
    (2, 320, 2560): (256, 32),
    (4, 320, 2560): (256, 32),
    (2, 2560, 160): (128, 8),
    (4, 2560, 160): (128, 8),
    (2, 512, 2560): (256, 32),
    (4, 512, 2560): (256, 32),
    (2, 62080, 2560): (128, 32),
    (4, 62080, 2560): (128, 32),
}


def shape_supported(x, weight):
    return (
        os.environ.get("SGLANG_SM70_MTP_SMALL_GEMM", "1") == "1"
        and (x.shape[0], *weight.shape) in _SMALL_CONFIGS
    )


@cache_once
def _small_module(rows, threads, lanes):
    return load_jit(
        "sm70_small_gemm",
        rows,
        threads,
        lanes,
        cuda_files=["elementwise/sm70_small_gemm.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::sm70_small_gemm::run<{rows},{threads},{lanes}>")
        ],
    )


def linear_small(x, weight):
    threads, lanes = _SMALL_CONFIGS[(x.shape[0], *weight.shape)]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _small_module(x.shape[0], threads, lanes).run(x, weight, out)
    return out
