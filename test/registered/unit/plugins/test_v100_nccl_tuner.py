"""The explicit quad tuner must preserve NCCL's policy outside its size band."""

import ctypes
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Init = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
)
Choose = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
)
Destroy = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class Plugin(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("init", Init),
        ("choose", Choose),
        ("destroy", Destroy),
    ]


class TestV100NCCLTuner(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("cc")
        if compiler is None:
            raise unittest.SkipTest("C compiler required for the external NCCL ABI")
        cls.temporary = tempfile.TemporaryDirectory()
        source = Path(__file__).resolve().parents[4] / "v100_plus" / "nccl"
        library = Path(cls.temporary.name) / "tuner.so"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
                "-I",
                str(source / "include"),
                str(source / "quad_tuner.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.library = ctypes.CDLL(str(library))
        cls.plugin = Plugin.in_dll(cls.library, "ncclTunerPlugin_v4")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(
        self, ranks=8, nodes=1, operation=4, size=15360, ignored=False, protocols=3
    ):
        context = ctypes.c_void_p()
        self.assertEqual(self.plugin.init(ranks, nodes, None, ctypes.byref(context)), 0)
        values = [float(10 + i) for i in range(21)]
        if ignored:
            values[0] = -1.0
        table = (ctypes.c_float * 21)(*values)
        channels = ctypes.c_int(7)
        try:
            self.assertEqual(
                self.plugin.choose(
                    context,
                    operation,
                    size,
                    1,
                    table,
                    7,
                    protocols,
                    0,
                    ctypes.byref(channels),
                ),
                0,
            )
        finally:
            self.assertEqual(self.plugin.destroy(context), 0)
        self.assertEqual(channels.value, 7)
        return values, list(table)

    def test_target_reduction_selects_one_available_algorithm_protocol(self):
        for size in (8192, 15360, 262144):
            original, actual = self.invoke(size=size)
            self.assertEqual(actual[0], 0.0)
            self.assertEqual(actual[1:], original[1:])

    def test_large_messages_other_groups_and_other_collectives_keep_default_costs(self):
        for changes in (
            {"size": 5120},
            {"size": 8191},
            {"size": 262145},
            {"size": 22282240},
            {"ranks": 4},
            {"nodes": 2},
            {"operation": 2},
            {"protocols": 2},
        ):
            original, actual = self.invoke(**changes)
            self.assertEqual(actual, original)

    def test_explicitly_unavailable_tree_ll_is_never_reenabled(self):
        original, actual = self.invoke(ignored=True)
        self.assertEqual(actual, original)


if __name__ == "__main__":
    unittest.main()
