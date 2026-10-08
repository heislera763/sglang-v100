"""Small-batch projections, HC gates and draft caches on SM70."""

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
    from sglang_v100_plus.runtime import apply_unquant, store


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0),
    "Requires a V100",
)
class TestV100SmallBatches(CustomTestCase):
    def test_pp_commit_graph_refreshes_request_acceptance_and_source_state(self):
        """New request slots/acceptance addresses must not replay the first inputs."""
        from sglang_v100_plus.commit_graph import CommitGraph

        destination = torch.zeros(4, 8, device="cuda")
        source = torch.arange(3 * 8, device="cuda", dtype=torch.float32).reshape(3, 8)
        expected = torch.zeros(4, 8)

        def original(worker, batch, accept_lens, accept_index, draft_tokens):
            steps = accept_index.gather(1, (accept_lens - 1).reshape(-1, 1)).flatten()
            destination.index_copy_(
                0, batch.req_pool_indices, source.index_select(0, steps)
            )

        state = None
        with torch.inference_mode():
            for iteration in range(9):
                slot, count = iteration % 4, iteration % 3 + 1
                # Every turn uses new GPU tensors, while the graph must also
                # read the current contents of the persistent source pool.
                batch = SimpleNamespace(
                    req_pool_indices=torch.tensor([slot], device="cuda")
                )
                accept = torch.tensor([count], device="cuda")
                indices = torch.tensor([[2, 0, 1]], device="cuda")
                source.copy_(
                    torch.arange(24, device="cuda").reshape(3, 8) + iteration * 100
                )
                if state is None:
                    state = CommitGraph(original, None, batch, accept, indices, 3)
                state.run(batch, accept, indices)
                step = [2, 0, 1][count - 1]
                expected[slot] = torch.arange(8) + step * 8 + iteration * 100
                torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
        self.assertIsNotNone(state.graph)

    def test_pp_output_pack_preserves_bits_and_cross_stream_readiness(self):
        from collections import deque

        from sglang_v100_plus.pipeline import (
            receive_output,
            send_output_dict,
            unpack_output,
        )

        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        owner = SimpleNamespace(
            _v100_pp_local_outputs=deque(),
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        tensors = {
            "next_token_ids": torch.arange(3, device="cuda", dtype=torch.int32),
            "seq_lens": torch.tensor([8193], device="cuda", dtype=torch.int64),
            "probabilities": torch.tensor([-0.0, 0.125, 0.875], device="cuda"),
            "hidden_states": torch.arange(12, device="cuda", dtype=torch.float16)
            .reshape(3, 4)
            .T,
            "empty_parents": torch.empty(1, 0, device="cuda", dtype=torch.int64),
            "mask": torch.tensor([True, False, True], device="cuda"),
            "scalar": torch.tensor(7, device="cuda", dtype=torch.int64),
            "spec_next_draft_probs": torch.ones(1, 2, 3, device="cuda"),
            "__msg_type__": "output",
            "nested_logprobs": {
                "values": [torch.tensor([-0.0, -0.125], device="cuda")],
                "indices": (torch.tensor([7, 13], device="cuda"), None),
                "cpu": torch.empty(2, dtype=torch.int64, pin_memory=True),
            },
        }
        producer, sender = torch.cuda.Stream(), torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            torch.cuda._sleep(200000)
            tensors["hidden_states"].add_(16)
            tensors["nested_logprobs"]["values"][0].add_(0.5)
            tensors["nested_logprobs"]["cpu"].copy_(
                tensors["nested_logprobs"]["indices"][0], non_blocking=True
            )
            ready = torch.cuda.Event()
            ready.record()
        sent = []

        def wire(_, payload, **kwargs):
            sent.append((payload, kwargs["ready_event"]))
            return []

        with torch.cuda.stream(sender):
            send_output_dict(wire, owner, tensors, msg_type="output", ready_event=ready)
        payload, packed_ready = sent[0]
        self.assertIsNot(packed_ready, ready)
        torch.cuda.current_stream().wait_event(packed_ready)
        actual = unpack_output(payload)
        self.assertNotIn("spec_next_draft_probs", actual)
        self.assertIn("spec_next_draft_probs", tensors)
        self.assertEqual(payload["__v100_pp_output_payload__"].numel() % 8, 0)

        def compare(actual_value, expected):
            if isinstance(expected, torch.Tensor):
                self.assertEqual(actual_value.shape, expected.shape)
                self.assertEqual(actual_value.dtype, expected.dtype)
                self.assertEqual(actual_value.device.type, expected.device.type)
                self.assertTrue(
                    torch.equal(
                        actual_value.contiguous().reshape(-1).view(torch.uint8),
                        expected.contiguous().reshape(-1).view(torch.uint8),
                    )
                )
            elif isinstance(expected, dict):
                for key in expected:
                    compare(actual_value[key], expected[key])
            elif isinstance(expected, (list, tuple)):
                self.assertIsInstance(actual_value, type(expected))
                for item, reference in zip(actual_value, expected):
                    compare(item, reference)
            else:
                self.assertEqual(actual_value, expected)

        for key, expected in tensors.items():
            if key != "spec_next_draft_probs":
                compare(actual[key], expected)

        # Prefill has no proposal q, but nested GPU logprobs still must travel
        # as tensor payloads. Restoring CPU leaves must wait for receive DMA.
        prefill = {"nested_logprobs": tensors["nested_logprobs"]}
        with torch.cuda.stream(sender):
            send_output_dict(wire, owner, prefill, msg_type="output", ready_event=ready)
        packed, sent_ready = sent[-1]
        receiver = SimpleNamespace(
            _v100_pp_local_outputs=deque(), pp_group=SimpleNamespace(is_last_rank=False)
        )
        copied = packed.copy()
        with torch.cuda.stream(sender):
            sender.wait_event(sent_ready)
            torch.cuda._sleep(200000)
            copied["__v100_pp_output_payload__"] = packed[
                "__v100_pp_output_payload__"
            ].clone()
            received_ready = torch.cuda.Event()
            received_ready.record()
        restored, event = receive_output(lambda _: (copied, received_ready), receiver)
        self.assertIs(event, received_ready)
        compare(restored, prefill)
        if torch.cuda.device_count() > 1:
            copied["__v100_pp_output_payload__"] = copied[
                "__v100_pp_output_payload__"
            ].to("cuda:1")
            restored = unpack_output(copied)
            self.assertEqual(restored["nested_logprobs"]["values"][0].device.index, 1)
            torch.testing.assert_close(
                restored["nested_logprobs"]["values"][0].cpu(),
                prefill["nested_logprobs"]["values"][0].cpu(),
                rtol=0,
                atol=0,
            )

    def test_wide_topk_preserves_ties_mass_and_graph_inputs(self):
        """The faster renorm must retain cutoff ties and read each graph replay."""
        from sgl_kernel.sampling import _top_k_renorm_probs_internal
        from sglang_v100_plus.runtime import top_k_renorm_probs

        from sglang.srt.sampling.sampling_params import TOP_K_ALL

        torch.manual_seed(531)
        for vocab in (131072, 154880, 248320, 262144):
            for rows in (1, 2, 3, 4):
                probs = torch.softmax(
                    torch.randn(rows, vocab, device="cuda") * 4, dim=-1
                )
                top_ks = torch.full((rows,), 20, device="cuda", dtype=torch.int32)
                top_k_renorm_probs(probs, top_ks)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = top_k_renorm_probs(probs, top_ks)
                for tied in (False, True, False):
                    with self.subTest(vocab=vocab, rows=rows, tied=tied):
                        if tied:
                            probs.fill_(1 / probs.shape[-1])
                        else:
                            probs.copy_(
                                torch.softmax(torch.randn_like(probs) * 4, dim=-1)
                            )
                        top_ks.copy_(
                            torch.tensor(
                                [1, 20, 50, TOP_K_ALL][:rows],
                                device="cuda",
                                dtype=torch.int32,
                            ).roll(1 if tied else 0)
                        )
                        graph.replay()
                        # An independent FP64 mass calculation and the old
                        # AOT implementation both constrain the new output.
                        indices = top_ks.long().clamp(max=vocab)[:, None] - 1
                        pivot = (
                            probs.double()
                            .sort(dim=-1, descending=True)
                            .values.gather(1, indices)
                        )
                        kept = torch.where(probs >= pivot, probs.double(), 0.0)
                        reference = (kept / kept.sum(-1, keepdim=True)).float()
                        legacy = _top_k_renorm_probs_internal(probs, top_ks, 0)
                        self.assertTrue(torch.equal(actual > 0, reference > 0))
                        self.assertTrue(torch.equal(actual > 0, legacy > 0))
                        torch.testing.assert_close(
                            actual, reference, rtol=3e-6, atol=1e-8
                        )
                        torch.testing.assert_close(actual, legacy, rtol=3e-6, atol=1e-8)

    def test_dflash_output_convolution_keeps_range_until_residual_norm(self):
        from sglang_v100_plus.dflash import SM70DFlashGroupedConv

        conv = SM70DFlashGroupedConv(16, 8, 2, 16).cuda().half()
        conv.base_kernel.data.fill_(2.0)
        hidden = torch.full((16, 16), 2048.0, device="cuda", dtype=torch.float16)
        coefficients = torch.full((16, 2, 1), 63.0, device="cuda", dtype=torch.float16)
        conv.finish(hidden, coefficients)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = conv.finish(hidden, coefficients)
        for value in (2048.0, 1024.0):
            hidden.fill_(value)
            graph.replay()
            expected = torch.full(
                (16, 16), 2 * 65 * value, device="cuda", dtype=torch.float32
            )
            expected[::8] = 65 * value
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(output).all())

    def test_dflash_topk_graph_reads_fresh_logits(self):
        from sglang_v100_plus.dflash import candidate_topk

        def forbidden_backend(*args):
            self.fail("Volta candidate selection must not delegate to FlashInfer")

        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype), torch.inference_mode():
                scores = torch.full((8, 19360), -4096.0, device="cuda", dtype=dtype)
                rows = torch.arange(8, device="cuda")[:, None]
                positions = torch.arange(16, device="cuda")[None, :] * 1024
                values = torch.arange(16, device="cuda", dtype=dtype)[None, :].expand(
                    8, -1
                )
                scores[rows, positions.expand(8, -1)] = values
                candidate_topk(forbidden_backend, scores, 16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual_values, actual_ids = candidate_topk(
                        forbidden_backend, scores, 16
                    )
                for shift in (5, 91):
                    scores.fill_(-4096)
                    expected_ids = (positions + rows * 37 + shift).expand(8, -1)
                    scores[rows, expected_ids] = values
                    graph.replay()
                    torch.testing.assert_close(
                        actual_values, values.flip(-1), rtol=0, atol=0
                    )
                    self.assertTrue(torch.equal(actual_ids, expected_ids.flip(-1)))

    def test_fp16_draft_cache_preserves_strided_rows_and_graph_replay(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        with torch.inference_mode():
            pool = MHATokenToKVPool(
                size=64,
                page_size=64,
                dtype=torch.float16,
                head_num=1,
                head_dim=128,
                layer_num=1,
                device="cuda",
                enable_memory_saver=False,
                enable_alt_stream=False,
            )
            pool.k_buffer[0].zero_()
            pool.v_buffer[0].zero_()
            qkv = torch.randn(8, 768, device="cuda", dtype=torch.float16)
            key = qkv[:, 512:640].view(8, 1, 128)
            value = qkv[:, 640:].view(8, 1, 128)
            self.assertFalse(key.is_contiguous())
            locations = torch.arange(8, device="cuda", dtype=torch.int64)
            layer = SimpleNamespace(layer_id=0)
            k_scale = torch.tensor(2.0, device="cuda")
            v_scale = torch.tensor(3.0, device="cuda")

            def write():
                return store(
                    MHATokenToKVPool.set_kv_buffer,
                    pool,
                    layer,
                    locations,
                    key,
                    value,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

            write()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                write()
            expected_k, expected_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()
            for start in (0, 17, 41):
                qkv.normal_()
                key[..., 0] = 0.0
                key[..., 1] = -0.0
                value[..., 0] = 2**-24
                locations.copy_(torch.arange(start, start + 8, device="cuda"))
                graph.replay()
                # Slot 0 is reserved graph padding and the primary CUDA writer
                # skips it, including during capture/warmup.
                live = locations != 0
                expected_k[locations[live]] = key[live]
                expected_v[locations[live]] = value[live]
                self.assertTrue(
                    torch.equal(
                        pool.k_buffer[0].view(torch.int16), expected_k.view(torch.int16)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        pool.v_buffer[0].view(torch.int16), expected_v.view(torch.int16)
                    )
                )

    def test_glm_primary_projection_routes_preserve_fp32(self):
        def forbidden_fallback(*args):
            self.fail("Declared GLM projections must not delegate dispatch")

        with (
            envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True),
            torch.inference_mode(),
        ):
            for rows in (1, 6, 8):
                for n, dtype in (
                    (3336, torch.float16),
                    (32, torch.float32),
                    (768, torch.float16),
                    (3072, torch.float16),
                ):
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
