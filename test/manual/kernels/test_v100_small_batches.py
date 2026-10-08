"""Three-row kernels must preserve the validated four-row arithmetic."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[3] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.dispatch import V100FallbackError
    from sglang_v100_plus.kernels import gemm, sm70_hc_mix
    from sglang_v100_plus.runtime import apply_unquant


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires a V100",
)
class TestV100SmallBatches(CustomTestCase):
    def test_glm_primary_projection_routes_preserve_fp32(self):
        def forbidden_fallback(*args):
            self.fail("Declared GLM projections must not delegate dispatch")

        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            for rows in (1, 6):
                for n, dtype in ((3336, torch.float16), (32, torch.float32)):
                    x = torch.randn(rows, 4096, device="cuda", dtype=dtype)
                    weight = torch.randn(n, 4096, device="cuda", dtype=dtype) * 0.02
                    actual = apply_unquant(
                        forbidden_fallback, None, SimpleNamespace(weight=weight), x
                    )
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        actual, torch.nn.functional.linear(x, weight), rtol=0, atol=0
                    )
            x = torch.zeros(6, 4096, device="cuda", dtype=torch.float16)
            weight = torch.zeros(3337, 4096, device="cuda", dtype=torch.float16)
            with self.assertRaisesRegex(V100FallbackError, "gemm.unquantized_linear"):
                apply_unquant(
                    forbidden_fallback, None, SimpleNamespace(weight=weight), x
                )

    def test_hc_three_rows_equal_four_row_prefix(self):
        torch.manual_seed(53)
        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            x = torch.randn(4, 10240, device="cuda", dtype=torch.float16)
            down_weight = (
                torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.02
            )
            up_weight = (
                torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.02
            )
            inject = torch.randn(4, 10240, device="cuda", dtype=torch.float16) * 0.02
            down3 = sm70_hc_mix.hc_down(x[:3], down_weight)
            down4 = sm70_hc_mix.hc_down(x, down_weight)
            torch.testing.assert_close(down3, down4[:3], atol=0, rtol=0)
            up3 = sm70_hc_mix.hc_up(down3, x[:3], up_weight)
            up4 = sm70_hc_mix.hc_up(down4, x, up_weight)
            torch.testing.assert_close(up3, up4[:3], atol=0, rtol=0)
            gated3, partial3 = sm70_hc_mix.hc_down_with_gate(x[:3], down_weight, inject)
            gated4, partial4 = sm70_hc_mix.hc_down_with_gate(x, down_weight, inject)
            torch.testing.assert_close(gated3, gated4[:3], atol=0, rtol=0)
            torch.testing.assert_close(partial3, partial4[:3], atol=0, rtol=0)
            y = torch.randn(4, 2560, device="cuda", dtype=torch.float16)
            combine3 = sm70_hc_mix.hc_apply_gate(y[:3], x[:3], partial3)
            combine4 = sm70_hc_mix.hc_apply_gate(y, x, partial4)
            torch.testing.assert_close(combine3, combine4[:3], atol=0, rtol=0)

    def test_gemm_three_rows_equal_four_row_prefix(self):
        torch.manual_seed(54)
        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            for (rows, n, k), _ in gemm._SMALL_CONFIGS.items():
                if rows != 3:
                    continue
                with self.subTest(n=n, k=k):
                    x = torch.randn(4, k, device="cuda", dtype=torch.float16)
                    weight = (
                        torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
                    )
                    out3 = gemm.linear_small(x[:3], weight)
                    # N=1 had a two-row specialization only; rows are independent.
                    out_ref = (
                        gemm.linear_small(x, weight)[:3]
                        if (4, n, k) in gemm._SMALL_CONFIGS
                        else torch.cat(
                            [
                                gemm.linear_small(x[:2], weight),
                                gemm.linear_small(x[2:], weight),
                            ]
                        )[:3]
                    )
                    torch.testing.assert_close(out3, out_ref, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
