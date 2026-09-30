"""Compile V100-only sources using mainline's JIT loader and private caches."""

from pathlib import Path
from sglang.kernels.jit.utils import cache_once, make_cpp_args
from sglang.kernels.jit.utils import load_jit as _load_jit


def load_jit(*args, **kwargs):
    import sglang

    source_root = Path(__file__).parent / "csrc"
    for key in ("cpp_files", "cuda_files"):
        if key in kwargs:
            kwargs[key] = [str(source_root / path) for path in kwargs[key]]
    # The SM70 HC gate shares the unchanged mainline combine implementation.
    shared = Path(sglang.__file__).parent / "kernels/jit/csrc/elementwise"
    kwargs["extra_include_paths"] = [
        *kwargs.get("extra_include_paths", []),
        str(shared),
    ]
    return _load_jit(*args, **kwargs)
