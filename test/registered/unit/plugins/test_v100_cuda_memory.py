"""External module loads must not OOM while idle Torch blocks are reclaimable."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import cuda_memory


class _Allocator:
    """Model independent driver allocation and reclaimable cached segments."""

    def __init__(self, *, capturing=False, free=786432):
        self.capturing = capturing
        self.free = free
        self.allocated = 30500569088
        self.reserved = 31358713856
        self.non_releasable = 88489984

    def mem_get_info(self):
        return self.free, 34072559616

    def is_current_stream_capturing(self):
        return self.capturing

    def memory_reserved(self):
        return self.reserved

    def memory_allocated(self):
        return self.allocated

    def empty_cache(self):
        if self.capturing:
            raise RuntimeError("Cannot release allocator blocks during capture")
        released = self.reserved - self.allocated - self.non_releasable
        self.free += released
        self.reserved -= released

    def load_module(self):
        # An external driver allocation cannot invoke Torch's OOM retry.
        if self.free < 32 << 20:
            raise RuntimeError("CUDA module load out of memory")
        self.free -= 32 << 20
        return "same kernel"


class TestV100ModuleMemory(CustomTestCase):
    def test_cold_module_fits_after_reclaim_without_changing_live_allocations(self):
        allocator = _Allocator()
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            allocator.load_module()
        live = allocator.allocated
        config = SimpleNamespace(features=SimpleNamespace(enable_memory_saver=False))
        with (
            patch.object(cuda_memory.torch, "cuda", allocator),
            patch.object(cuda_memory, "get_exec", return_value=config),
        ):
            cuda_memory.reclaim_for_triton_load(None, None, "kernel", {}, "hash")
        self.assertEqual(allocator.load_module(), "same kernel")
        self.assertEqual(allocator.allocated, live)
        self.assertEqual(allocator.reserved - live, allocator.non_releasable)

    def test_capture_custom_arena_and_plenty_of_headroom_preserve_reservations(self):
        for capture, custom, free in (
            (True, False, 786432),
            (False, True, 786432),
            (False, False, 2 << 30),
        ):
            with self.subTest(capture=capture, custom=custom, free=free):
                allocator = _Allocator(capturing=capture, free=free)
                config = SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=custom)
                )
                before = allocator.reserved, allocator.free
                with (
                    patch.object(cuda_memory.torch, "cuda", allocator),
                    patch.object(cuda_memory, "get_exec", return_value=config),
                ):
                    cuda_memory.reclaim_for_triton_load(
                        None, None, "kernel", {}, "hash"
                    )
                self.assertEqual((allocator.reserved, allocator.free), before)


if __name__ == "__main__":
    unittest.main()
