"""HC token partitions must preserve row ownership and forward lifetime."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus import prefill
    from sglang_v100_plus.dflash import SM70DFlashResidualNorm, forward_draft
    from sglang_v100_plus.dispatch import prefill_scope
    from sglang_v100_plus.hc_state import SM70GatedResidualState


class TestV100Prefill(CustomTestCase):
    def test_fp8_route_alignment_matches_native_tile_and_preserves_other_profiles(self):
        """Only the whole-expert FP8/EP4 profile may replace generic route alignment."""
        from sglang_v100_plus.fp8 import fp8_route_block_size

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            select_marlin_moe_block_size,
        )

        hidden = SimpleNamespace(
            shape=(4352, 2560), ndim=2, dtype=torch.float16, is_cuda=True
        )
        ids = SimpleNamespace(shape=(4352, 10), dtype=torch.int32)
        w1 = SimpleNamespace(shape=(128, 160, 5120))
        w2 = SimpleNamespace(shape=(128, 40, 10240))
        scales = SimpleNamespace(dtype=torch.float16, _sm70_fp8_scale=True)
        fields = dict(
            model_path="dummy",
            tp_size=4,
            ep_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        )
        for changes, tuning, tagged, expected in (
            ({}, False, True, 32),
            ({"ep_size": 1}, False, True, 64),
            ({"tp_size": 8}, False, True, 64),
            ({"pp_size": 1}, False, True, 64),
            ({"max_running_requests": 2}, False, True, 64),
            ({}, True, True, 64),
            ({}, False, False, 64),
        ):
            with (
                self.subTest(changes=changes, tuning=tuning, tagged=tagged),
                get_context().override_server_args(**(fields | changes)),
                patch(
                    "sglang_v100_plus.kernels.moe_marlin._sm70_marlin_user_tuning",
                    tuning,
                ),
            ):
                scales._sm70_fp8_scale = tagged
                actual = fp8_route_block_size(
                    select_marlin_moe_block_size, hidden, ids, w1, w2, scales
                )
                self.assertEqual(actual, expected)

    def test_dflash_residual_addition_does_not_overflow_before_norm(self):
        norm = SM70DFlashResidualNorm(
            SimpleNamespace(
                weight=torch.nn.Parameter(torch.ones(4, dtype=torch.float16)),
                variance_epsilon=1e-5,
            )
        )
        x = torch.full((2, 4), 65504.0, dtype=torch.float16)
        self.assertFalse(torch.isfinite(x + x).all())
        output, residual = norm(x, x)
        self.assertEqual(residual.dtype, torch.float32)
        torch.testing.assert_close(
            residual, torch.full((2, 4), 131008.0), rtol=0, atol=0
        )
        torch.testing.assert_close(output, torch.ones_like(x), rtol=0, atol=0)

    def test_dflash_eager_entry_keeps_fp16_embedding_bits(self):
        embeddings = torch.tensor(
            [[1.0009765625, 1.0029296875, -1.0009765625]], dtype=torch.float16
        )
        rounded = embeddings.bfloat16()
        self.assertFalse(torch.equal(rounded.half(), embeddings))
        batch = SimpleNamespace(input_embeds=embeddings)
        actual = forward_draft(
            lambda owner, ids, positions, fb, input_embeds: input_embeds,
            None,
            None,
            None,
            batch,
            input_embeds=rounded,
        )
        self.assertTrue(
            torch.equal(actual.view(torch.int16), embeddings.view(torch.int16))
        )

    def test_boundary_keeps_each_stages_cached_gate_and_clears_it(self):
        residual = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12)
        gate_a, gate_b = torch.full((4, 3), 2.0), torch.full((4, 3), 3.0)
        seen = []

        def mix_with_gate(gate):
            return lambda value: (value[:, :4], (value, value * 0.5, gate))

        def combine(output, values):
            seen.append(values)
            return values[0] + output.repeat(1, 3)

        state = SM70GatedResidualState(
            expand=lambda value: value,
            attn_mix=mix_with_gate(gate_a),
            ffn_mix=mix_with_gate(gate_b),
            attn_combine=combine,
            ffn_combine=combine,
        )
        ops = state.residual_ops()
        hidden, residual = ops.attn_readout.read(residual, None)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, hidden, residual, None
        )
        self.assertIs(seen[0][2], gate_a)
        self.assertIs(state.gate, gate_b)
        ops.ffn_update.update(hidden, residual)
        self.assertIs(seen[1][2], gate_b)
        self.assertIsNone(state.gate)
        self.assertIsNone(state.normed)
        # A subsequent prefill/empty read returns the ordinary pair and must
        # never reuse a gate from an earlier captured or speculative read.
        state.gate = gate_a
        state.attn_mix = lambda value: (value[:, :4], (value, value * 0.25))
        hidden, residual = ops.attn_readout.read(residual, None)
        ops.attn_update.update(hidden, residual)
        self.assertIsNone(state.gate)
        self.assertEqual(len(seen[-1]), 2)

    def test_boundary_slices_gate_rows_with_normalized_residuals(self):
        residual = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12)
        normed = residual * 0.5
        gate = torch.arange(4 * 8 * 3).reshape(4, 8, 3)
        state = SM70GatedResidualState(None, None, None, None, None, normed, gate)
        parallel = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        with (
            patch("sglang_v100_plus.hc_state.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.layers.layer_boundary.residual.gated.get_parallel",
                return_value=parallel,
            ),
        ):
            selected = state.residual_ops().attn_update.slice_residual_attn_tp(residual)
        torch.testing.assert_close(selected, residual[2:])
        torch.testing.assert_close(state.normed, normed[2:])
        torch.testing.assert_close(state.gate, gate[2:])

    def test_ple_preparation_gathers_before_the_causal_convolution(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        full = torch.arange(256 * 5).reshape(256, 5)
        batch = SimpleNamespace(residual_stream=None)
        ple_batch = object()
        owner = SimpleNamespace(ple=object())

        class Group:
            def all_gather(_, value, dim):
                torch.testing.assert_close(value, full[64:128])
                self.assertEqual(dim, 0)
                return full

        def original(layer, hidden, actual_batch, actual_ple_batch):
            self.assertIs(layer, owner)
            self.assertIs(actual_batch, batch)
            self.assertIs(actual_ple_batch, ple_batch)
            residual_batch.stream_of(batch).check(hidden)
            return hidden

        with (
            get_context().override_server_args(model_path="dummy"),
            get_parallel().override(attn_tp_group=Group()),
            prefill.partition_scope(prefill.Partition(256, 1)),
        ):
            local = full[64:128]
            residual_batch.set_written(local, batch)
            actual = prefill.prepare_attention(original, owner, local, batch, ple_batch)
            self.assertIs(actual, full)
            owner.ple = None
            local = full[64:128]
            residual_batch.set_written(local, batch)
            self.assertIs(
                prefill.prepare_attention(original, owner, local, batch, ple_batch),
                local,
            )

    def test_only_eager_serialized_fp8_prefill_is_partitioned(self):
        """Only ordinary/MTP eager prefill may partition; verification stays full."""
        owner = SimpleNamespace(hc_count=4, hidden_size=2560)
        batch = SimpleNamespace(
            batch_size=1, forward_mode=SimpleNamespace(is_extend=lambda: True)
        )
        fields = {
            "model_path": "dummy",
            "quantization": "fp8",
            "tp_size": 4,
            "pp_size": 2,
            "max_running_requests": 1,
            "disable_overlap_schedule": True,
            "disable_prefill_cuda_graph": True,
        }
        with envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(True), prefill_scope(True):
            for changes in (
                {},
                {"pp_size": 1},
                {"tp_size": 8},
                {"quantization": "nvfp4"},
                {"max_running_requests": 2},
                {"disable_prefill_cuda_graph": False},
                {"enable_return_hidden_states": True},
                {"speculative_algorithm": "EAGLE"},
                {"speculative_algorithm": "DFLASH"},
            ):
                with (
                    self.subTest(changes=changes),
                    get_context().override_server_args(**(fields | changes)),
                    get_parallel().override(
                        attn_tp_size=changes.get("tp_size", 4), attn_tp_rank=2
                    ),
                ):
                    for rows in (128, 256, 257):
                        actual = prefill._partition(owner, torch.empty(rows), batch)
                        if rows >= 256 and changes in (
                            {},
                            {"speculative_algorithm": "EAGLE"},
                        ):
                            self.assertEqual(actual, prefill.Partition(rows, 2))
                        else:
                            self.assertIsNone(actual)

    def test_partitions_gather_in_original_order_and_do_not_leak(self):
        """Offsets cannot cross ranks or survive a failed/nested forward."""
        full = torch.arange(256 * 5).reshape(256, 5)
        with get_context().override_server_args(model_path="dummy"):
            for rank in range(4):
                plan = prefill.Partition(256, rank)
                expected = full.tensor_split(4)[rank]

                class Group:
                    def all_gather(_, value, dim, expected=expected):
                        torch.testing.assert_close(value, expected)
                        self.assertEqual(dim, 0)
                        return full.clone()

                with get_parallel().override(attn_tp_group=Group()):
                    with prefill.partition_scope(plan):
                        local = prefill.local_input(full)
                        torch.testing.assert_close(local, expected)
                        self.assertIs(prefill.local_input(local), local)
                        torch.testing.assert_close(prefill.gather_input(local), full)
                        with prefill.partition_scope(None):
                            self.assertIs(prefill.local_input(full), full)
                        torch.testing.assert_close(prefill.local_input(full), expected)
                        with self.assertRaisesRegex(ValueError, "partition"):
                            prefill.local_input(full[:17])
                    self.assertIs(prefill.local_input(full), full)
            with (
                self.assertRaisesRegex(RuntimeError, "failed"),
                prefill.partition_scope(prefill.Partition(256, 0)),
            ):
                raise RuntimeError("failed forward")
            self.assertIs(prefill.gather_input(full), full)

    def test_ragged_partitions_pad_only_hc_and_trim_before_consumers(self):
        """Equal collective counts must retain every real row, including short tails."""
        with get_context().override_server_args(model_path="dummy"):
            for rows in (257, 325, 3994, 4165):
                with self.subTest(rows=rows):
                    full = torch.arange(rows * 12).reshape(rows, 4, 3)
                    count = (rows + 3) // 4
                    parts = [
                        prefill.Partition(rows, rank).local(full) for rank in range(4)
                    ]
                    padded = torch.cat(parts)
                    self.assertEqual(padded.shape[0], 4 * count)
                    torch.testing.assert_close(padded[:rows], full, rtol=0, atol=0)
                    self.assertTrue(
                        torch.equal(padded[rows:], torch.zeros_like(padded[rows:]))
                    )

                    class Group:
                        def all_gather(_, value, dim):
                            self.assertEqual(dim, 0)
                            self.assertEqual(value.shape[0], count)
                            return padded

                    for rank, part in enumerate(parts):
                        with (
                            get_parallel().override(attn_tp_group=Group()),
                            prefill.partition_scope(prefill.Partition(rows, rank)),
                        ):
                            self.assertIs(prefill.local_input(part), part)
                            torch.testing.assert_close(
                                prefill.gather_input(part), full, rtol=0, atol=0
                            )
        self.assertIs(prefill.gather_input(full), full)

    def test_inactive_forwards_hide_parent_partition_and_restore_it(self):
        """A decode/nested inactive call cannot inherit the previous prefill's rows."""
        full = torch.arange(256).reshape(256, 1)
        owner = SimpleNamespace()
        batch = SimpleNamespace()
        with (
            get_context().override_server_args(model_path="dummy"),
            envs.SGLANG_OPT_SM70_HC_PREFILL_SP.override(False),
            prefill.partition_scope(prefill.Partition(256, 2)),
        ):

            def original(*_args, **_kwargs):
                self.assertIs(prefill.local_input(full), full)
                raise RuntimeError("nested failure")

            with self.assertRaisesRegex(RuntimeError, "nested failure"):
                prefill.model_forward(original, owner, full, None, batch)
            torch.testing.assert_close(prefill.local_input(full), full[128:192])


if __name__ == "__main__":
    unittest.main()
