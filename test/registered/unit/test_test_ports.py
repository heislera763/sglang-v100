"""Test imports must work with CPU-only and UUID CUDA visibility settings."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, _default_test_port

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDefaultTestPorts(CustomTestCase):
    def test_numeric_lanes_preserve_ci_allocation(self):
        for devices, lane in (("0", 0), ("3,1", 3), ("9", 9), ("12,0", 1)):
            with self.subTest(devices=devices):
                self.assertEqual(_default_test_port(devices, True), 10000 + lane * 2000)
                self.assertEqual(
                    _default_test_port(devices, False), 20000 + lane * 1000
                )

    def test_cpu_visibility_has_default_lane(self):
        for devices in (None, "", "-1", "   ", "-1,0"):
            with self.subTest(devices=devices):
                self.assertEqual(_default_test_port(devices, True), 10000)
                self.assertEqual(_default_test_port(devices, False), 20000)

    def test_uuid_lanes_are_stable_bounded_and_use_first_device(self):
        for devices in ("GPU-93fd1784-7e18", "MIG-GPU-93fd1784/1/0", "MIG-abcdef"):
            with self.subTest(devices=devices):
                for ci in (False, True):
                    port = _default_test_port(devices, ci)
                    self.assertEqual(
                        port, _default_test_port(devices + ",GPU-other", ci)
                    )
                    self.assertEqual(port, _default_test_port(" " + devices + " ", ci))
                    self.assertGreaterEqual(port, 30000)
                    self.assertLess(port + 1000, 50000)


if __name__ == "__main__":
    unittest.main()
