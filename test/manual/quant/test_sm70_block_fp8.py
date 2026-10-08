"""SM70 block-FP8 loading/routing versus independently dequantized weights.

Run with the V100 plugin and strict dispatch enabled. Distinct scale blocks,
signed E4M3 values and masked EP routes catch scale/layout/type confusion.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase


class TestSM70BlockFP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        assert torch.cuda.get_device_capability() == (7, 0)
        from sglang.srt.plugins import load_plugins

        load_plugins()

    def test_tp8_unquantized_projection_shapes(self):
        from sglang_v100_lite.kernels.gemm import linear_dense, supported

        torch.manual_seed(45)
        for n, k in (
            (2048, 2560),
            (12, 2560),
            (2560, 768),
            (160, 2560),
            (2560, 80),
            (31040, 2560),
        ):
            weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.02
            for rows in (1, 2, 3, 4):
                with self.subTest(n=n, k=k, rows=rows):
                    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
                    self.assertTrue(supported(x, weight))
                    actual = linear_dense(x, weight)
                    torch.testing.assert_close(
                        actual, F.linear(x, weight), rtol=0.005, atol=0.003
                    )

    def test_tp8_three_head_sparse_attention(self):
        from sglang_v100_lite.kernels.qsa_cuda import (
            sm70_cuda_qsa_decode,
            sm70_cuda_qsa_prefill,
        )
        from sglang_v100_lite.qsa import QwenSparseAttnBackend

        torch.manual_seed(31)
        pool, length, rows, heads = 96, 64, 3, 3
        keys = torch.randn(pool, 1, 256, device="cuda", dtype=torch.float16).to(
            torch.float8_e5m2
        )
        values = torch.randn_like(keys, dtype=torch.float16).to(torch.float8_e5m2)
        table = torch.randperm(pool, device="cuda")[:length].to(torch.int32)[None]
        queries = torch.randn(rows, heads, 256, device="cuda", dtype=torch.float16)
        selected = torch.arange(length, device="cuda", dtype=torch.int32)[None].repeat(
            rows, 1
        )
        requests = torch.zeros(rows, device="cuda", dtype=torch.int32)
        lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
        scale = 256**-0.5
        dense_slots = table[0, :rows].long()
        dense_k, dense_v = keys[dense_slots].half(), values[dense_slots].half()
        dense = QwenSparseAttnBackend._forward_sm70_dense_prefill(
            queries, dense_k, dense_v, [rows], scale
        )
        dense_reference = torch.stack(
            [
                (
                    queries[row].float() @ dense_k[: row + 1, 0].float().T * scale
                ).softmax(-1)
                @ dense_v[: row + 1, 0].float()
                for row in range(rows)
            ]
        ).half()
        torch.testing.assert_close(dense, dense_reference, rtol=0.005, atol=0.003)

        def reference(selections, counts):
            result = []
            for row, count in enumerate(counts):
                positions = selections[row, :count].long()
                positions = positions[positions >= 0]
                slots = table[0, positions].long()
                k = keys[slots, 0].float()
                v = values[slots, 0].float()
                result.append((queries[row].float() @ k.T * scale).softmax(-1) @ v)
            return torch.stack(result).half()

        actual = sm70_cuda_qsa_decode(
            queries, keys, values, table, requests, selected, lengths, scale
        )
        torch.testing.assert_close(
            actual, reference(selected, [length] * rows), rtol=0.005, atol=0.003
        )
        # Prefix-sized causal selections vary per query, including a masked
        # candidate; the native prefill kernel must honor these row boundaries.
        selections = torch.full_like(selected, -1)
        counts = [17, 33, 64]
        for row, count in enumerate(counts):
            selections[row, :count] = torch.arange(
                count, device="cuda", dtype=torch.int32
            )
        actual = sm70_cuda_qsa_prefill(
            queries, keys, values, table, requests[:1], selections, lengths[:1], scale
        )
        torch.testing.assert_close(
            actual, reference(selections, counts), rtol=0.005, atol=0.003
        )

    def test_tensor_core_prefill_keeps_each_querys_sparse_causal_set(self):
        """A shared KV tile must not turn the selected union into every row's keys."""
        from sglang_v100_lite.kernels.qsa_prefill import qsa_masked_prefill

        torch.manual_seed(83)
        rows, length, topk, pool = 129, 257, 64, 321
        keys = torch.randn(pool, 1, 256, device="cuda", dtype=torch.float16).to(
            torch.float8_e5m2
        )
        values = torch.randn_like(keys, dtype=torch.float16).to(torch.float8_e5m2)
        table = torch.stack(
            [torch.randperm(pool, device="cuda")[:length] for _ in range(2)]
        ).int()
        requests = torch.tensor([1], dtype=torch.int32, device="cuda")
        # Distinct per-row candidates, including bit 31, future keys, a completely
        # masked row and a ragged last query/head tile.
        indices = (
            torch.rand(rows, length, device="cuda")
            .argsort(1)[:, :topk]
            .int()
            .contiguous()
        )
        indices[0].fill_(-1)
        indices[1, :4] = torch.tensor([-1, length + 5, 31, 32], device="cuda")
        remaining = torch.arange(length, device="cuda")
        indices[1, 4:] = remaining[(remaining != 31) & (remaining != 32)][: topk - 4]
        visible = length - rows + torch.arange(rows, device="cuda") + 1
        valid = (indices >= 0) & (indices < visible[:, None])
        valid &= torch.arange(topk, device="cuda")[None] < visible[:, None]
        slots = table[1, indices.clamp(0, length - 1).long()].long()
        k, v = keys[slots, 0].float(), values[slots, 0].float()
        for heads in (3, 6):
            with self.subTest(heads=heads):
                q = torch.randn(rows, heads, 256, device="cuda", dtype=torch.float16)
                logits = torch.bmm(q.float(), k.transpose(1, 2)) * (256**-0.5)
                probabilities = logits.masked_fill(~valid[:, None], -torch.inf).softmax(
                    -1
                )
                expected = torch.bmm(probabilities.nan_to_num(), v).half()
                actual = qsa_masked_prefill(
                    q, keys, values, table, requests, indices, length, 256**-0.5
                )
                torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.003)
                self.assertFalse(bool(actual[0].any()))

    def test_routed_experts_and_masked_rows(self):
        from sglang_v100_lite.fp8 import prepare_fp8_moe

        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.utils import initialize_moe_config
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
        from sglang.srt.runtime_context import get_context

        with get_context().override_server_args(
            model_path="dummy", moe_runner_backend="marlin"
        ):
            initialize_moe_config()
            torch.manual_seed(24)
            e, h, intermediate = 4, 2560, 640
            layer = torch.nn.Module()
            layer.params_dtype = torch.float16
            reference = {}
            for prefix, n, k in (("w13", 2 * intermediate, h), ("w2", h, intermediate)):
                # Cover every finite E4M3 encoding, including subnormals and
                # negative zero. Scale separately for realistic magnitudes.
                codes = torch.randint(254, (e, n, k), device="cuda", dtype=torch.int32)
                codes = (codes + (codes >= 127)).to(torch.uint8)
                weight = codes.view(torch.float8_e4m3fn)
                scales = (
                    torch.rand(e, n // 128, k // 128, device="cuda") * 0.001 + 0.0005
                )
                expanded = (
                    scales.half().repeat_interleave(128, 1).repeat_interleave(128, 2)
                )
                reference[prefix] = (weight.float() * expanded.float()).half()
                layer.register_parameter(
                    prefix + "_weight", torch.nn.Parameter(weight, requires_grad=False)
                )
                layer.register_parameter(
                    prefix + "_weight_scale_inv",
                    torch.nn.Parameter(scales, requires_grad=False),
                )
            method = Fp8MoEMethod(
                Fp8Config(
                    is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
                )
            )
            prepare_fp8_moe(None, method, layer)
            for rows in (1, 3, 128):
                for masked in (False, True):
                    with self.subTest(rows=rows, masked=masked):
                        x = (
                            torch.randn(rows, h, device="cuda", dtype=torch.float16)
                            * 0.1
                        )
                        ids = (
                            torch.arange(
                                rows * 2, device="cuda", dtype=torch.int32
                            ).reshape(rows, 2)
                            % e
                        )
                        if masked:
                            ids[::2, 1] = -1
                            ids[0].fill_(-1)
                        weights = torch.rand(
                            rows, 2, device="cuda", dtype=torch.float32
                        )
                        expected = torch.zeros_like(x, dtype=torch.float32)
                        for row in range(rows):
                            for route in range(2):
                                expert = int(ids[row, route])
                                if expert < 0:
                                    continue
                                gu = F.linear(
                                    x[row : row + 1], reference["w13"][expert]
                                )
                                gate, up = gu.float().chunk(2, dim=-1)
                                activation = (F.silu(gate) * up).half()
                                down = F.linear(activation, reference["w2"][expert])
                                expected[row] += (
                                    (down.float() * weights[row, route])
                                    .half()[0]
                                    .float()
                                )
                        actual = fused_marlin_moe(
                            x,
                            layer.w13_weight,
                            layer.w2_weight,
                            layer.w13_weight_scale_inv,
                            layer.w2_weight_scale_inv,
                            torch.zeros(rows, e, device="cuda"),
                            weights,
                            ids,
                            num_bits=8,
                            inplace=False,
                            expert_map=torch.arange(e, device="cuda", dtype=torch.int32)
                            if masked
                            else None,
                            global_num_experts=e,
                        )
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                        torch.testing.assert_close(
                            actual, expected.half(), rtol=0.005, atol=0.003
                        )
                        if rows == 1:
                            from sglang_v100_lite.kernels.sm70_fp8_moe_decode import (
                                fp8_moe_decode,
                            )

                            vector = fp8_moe_decode(
                                x,
                                layer.w13_weight,
                                layer.w2_weight,
                                layer.w13_weight_scale_inv,
                                layer.w2_weight_scale_inv,
                                ids,
                                weights,
                            )
                            torch.testing.assert_close(
                                vector, expected.half(), rtol=0.005, atol=0.003
                            )
                            vector_graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(vector_graph):
                                replay = fp8_moe_decode(
                                    x,
                                    layer.w13_weight,
                                    layer.w2_weight,
                                    layer.w13_weight_scale_inv,
                                    layer.w2_weight_scale_inv,
                                    ids,
                                    weights,
                                )
                            vector_graph.replay()
                            torch.testing.assert_close(
                                replay, expected.half(), rtol=0.005, atol=0.003
                            )
                            # Real EP decode retains ten global route slots.
                            # Masked slots may contain stale scratch from the
                            # preceding layer or graph replay and must vanish.
                            ten_ids = torch.full(
                                (1, 10), -1, device="cuda", dtype=torch.int32
                            )
                            ten_ids[0, [0, 3, 6, 9]] = torch.arange(
                                e, device="cuda", dtype=torch.int32
                            )
                            ten_weights = torch.rand(
                                (1, 10), device="cuda", dtype=torch.float32
                            )
                            ten_expected = torch.zeros_like(x, dtype=torch.float32)
                            for route, expert in zip((0, 3, 6, 9), range(e)):
                                gu = F.linear(x, reference["w13"][expert])
                                gate, up = gu.float().chunk(2, dim=-1)
                                act = (F.silu(gate) * up).half()
                                down = F.linear(act, reference["w2"][expert])
                                ten_expected += (
                                    (down.float() * ten_weights[0, route])
                                    .half()
                                    .float()
                                )
                            ten_actual = fp8_moe_decode(
                                x,
                                layer.w13_weight,
                                layer.w2_weight,
                                layer.w13_weight_scale_inv,
                                layer.w2_weight_scale_inv,
                                ten_ids,
                                ten_weights,
                            )
                            torch.testing.assert_close(
                                ten_actual, ten_expected.half(), rtol=0.005, atol=0.003
                            )
                            live_graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(live_graph):
                                live_out = fp8_moe_decode(
                                    x,
                                    layer.w13_weight,
                                    layer.w2_weight,
                                    layer.w13_weight_scale_inv,
                                    layer.w2_weight_scale_inv,
                                    ten_ids,
                                    ten_weights,
                                )
                            saved_ids = ten_ids.clone()
                            ten_ids.fill_(-1)
                            live_graph.replay()
                            self.assertEqual(int(torch.count_nonzero(live_out)), 0)
                            ten_ids.copy_(saved_ids)
                            live_graph.replay()
                            torch.testing.assert_close(
                                live_out, ten_expected.half(), rtol=0.005, atol=0.003
                            )
                        if masked:
                            self.assertEqual(int(torch.count_nonzero(actual[0])), 0)
                        if rows == 3 and masked:
                            # Verify the same expert masking under the graph
                            # replay used for ordinary/speculative decoding.
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                graphed = fused_marlin_moe(
                                    x,
                                    layer.w13_weight,
                                    layer.w2_weight,
                                    layer.w13_weight_scale_inv,
                                    layer.w2_weight_scale_inv,
                                    torch.zeros(rows, e, device="cuda"),
                                    weights,
                                    ids,
                                    num_bits=8,
                                    inplace=False,
                                    expert_map=torch.arange(
                                        e, device="cuda", dtype=torch.int32
                                    ),
                                    global_num_experts=e,
                                )
                            for _ in range(2):
                                graph.replay()
                                torch.testing.assert_close(
                                    graphed, expected.half(), rtol=0.005, atol=0.003
                                )


if __name__ == "__main__":
    unittest.main()
