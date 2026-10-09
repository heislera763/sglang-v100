"""Misaligned NVFP4 shards must preserve both gated halves and scale bytes."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.quantization import pad_nvfp4_expert_width


def checkpoint(width):
    layer = torch.nn.Module()
    for prefix, n, k in (("w13", 2 * width, 2560), ("w2", 2560, width)):
        weight = torch.randint(0, 256, (2, n, k // 2), dtype=torch.uint8)
        # Include arbitrary encoded bytes: padding must not do FP8 arithmetic.
        scales = torch.randint(0, 256, (2, n, k // 16), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        layer.register_parameter(
            prefix + "_weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        layer.register_parameter(
            prefix + "_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )
    return layer


class TestV100NVFP4Padding(CustomTestCase):
    def test_each_half_and_fc2_scale_group_preserves_checkpoint_bytes(self):
        layer = checkpoint(80)
        originals = {
            name: value.detach().clone().view(torch.uint8)
            for name, value in layer.named_parameters()
        }
        pad_nvfp4_expert_width(layer)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 192, 1280))
        self.assertEqual(tuple(layer.w2_weight.shape), (2, 2560, 48))
        for suffix in ("_weight", "_weight_scale"):
            first = getattr(layer, "w13" + suffix).view(torch.uint8)
            old = originals["w13" + suffix]
            self.assertTrue(torch.equal(first[:, :80], old[:, :80]))
            self.assertTrue(torch.equal(first[:, 96:176], old[:, 80:]))
            self.assertEqual(first[:, 80:96].count_nonzero().item(), 0)
            self.assertEqual(first[:, 176:].count_nonzero().item(), 0)
            second = getattr(layer, "w2" + suffix).view(torch.uint8)
            old = originals["w2" + suffix]
            self.assertTrue(torch.equal(second[..., : old.shape[-1]], old))
            self.assertEqual(second[..., old.shape[-1] :].count_nonzero().item(), 0)

    def test_aligned_shards_keep_their_original_storage(self):
        layer = checkpoint(160)
        pointers = {name: value.data_ptr() for name, value in layer.named_parameters()}
        pad_nvfp4_expert_width(layer)
        self.assertEqual(
            pointers,
            {name: value.data_ptr() for name, value in layer.named_parameters()},
        )

    def test_padding_does_not_reinterpret_another_weight_format(self):
        layer = checkpoint(80)
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.to(torch.int32), requires_grad=False
        )
        pointers = {name: value.data_ptr() for name, value in layer.named_parameters()}
        pad_nvfp4_expert_width(layer)
        self.assertEqual(
            pointers,
            {name: value.data_ptr() for name, value in layer.named_parameters()},
        )


if __name__ == "__main__":
    unittest.main()
