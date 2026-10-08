"""Reject silent V100 fallbacks while retaining opt-out dispatch semantics."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

# The fork plugin is separately packaged; CPU CI need not install that wheel.
with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    import sglang_v100_plus
    from sglang_v100_plus import glm_mhc, mqa, qsa, quantization, runtime
    from sglang_v100_plus.dispatch import (
        V100FallbackError,
        eager_extend,
        in_prefill,
        prefill_scope,
    )
    from sglang_v100_plus.kernels import sm70_fp8_kv


class TestV100StrictDispatch(CustomTestCase):
    def test_prefill_scope_excludes_verify_and_restores_after_errors(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        def check(owner, batch):
            return in_prefill()

        self.assertFalse(in_prefill())
        with prefill_scope(True):
            self.assertTrue(
                eager_extend(
                    check, None, SimpleNamespace(forward_mode=ForwardMode.EXTEND)
                )
            )
            self.assertFalse(
                eager_extend(
                    check, None, SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
                )
            )
            self.assertTrue(in_prefill())
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with prefill_scope(False):
                    raise RuntimeError("test failure")
            self.assertTrue(in_prefill())
        self.assertFalse(in_prefill())

    def test_inactive_plugin_cannot_silently_ignore_strict_mode(self):
        with patch.object(
            sglang_v100_plus, "os", SimpleNamespace(environ={"SGLANG_V100_PLUS": "0"})
        ):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                self.assertIsNone(sglang_v100_plus.register())
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                with self.assertRaisesRegex(V100FallbackError, "plugin.activation"):
                    sglang_v100_plus.register()

    def test_uncovered_hc_rows_fail(self):
        """Extending small-batch coverage must not reopen silent fallback."""
        x = torch.ones(5, 10240, dtype=torch.float16)
        original = lambda owner, tensor: tensor.sum(-1)
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
            torch.testing.assert_close(runtime.mix(original, None, x), x.sum(-1))
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
            with self.assertRaisesRegex(
                V100FallbackError, r"qwen.hc_mix.*rows in \(1, 2, 3, 4\).*(5, 10240)"
            ):
                runtime.mix(original, None, x)

    def test_guarded_fallbacks_fail_before_delegating(self):
        """A missing check at any separate dispatch boundary permits silent work."""
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        weight = torch.ones(4, 2)
        residual = torch.ones(3, 4, 2)
        q = torch.ones(3, 4, 128)
        k = torch.ones(3, 1, 128)
        starts = torch.zeros(3, dtype=torch.int32)
        ends = torch.full((3,), 3, dtype=torch.int32)
        cases = [
            (
                "gemm.unquantized_linear",
                lambda: runtime.apply_unquant(
                    lambda owner, layer, a, bias: a @ layer.weight.T,
                    None,
                    SimpleNamespace(weight=weight),
                    x,
                ),
                x @ weight.T,
            ),
            (
                "moe.top10_router",
                lambda: runtime.route_top10(
                    lambda scores, *a, **kw: scores.sum(-1),
                    x,
                    None,
                    10,
                    scoring_func="softmax",
                ),
                x.sum(-1),
            ),
            (
                "qwen.hc_combine",
                lambda: runtime.combine(
                    lambda owner, a, state: a + state[0], None, x, (x, x)
                ),
                x + x,
            ),
            (
                "qwen.fp8_kv_store",
                lambda: runtime.store(
                    lambda owner, layer, loc, key, value, *a: key + value,
                    SimpleNamespace(dtype=torch.float16),
                    None,
                    None,
                    x,
                    x,
                ),
                x + x,
            ),
            (
                "glm.mhc_pre",
                lambda: glm_mhc.mhc_pre(lambda a: a.sum(-1), residual),
                residual.sum(-1),
            ),
            (
                "glm.mhc_post",
                lambda: glm_mhc.mhc_post(
                    lambda a, *args: a + 1, x, residual, None, None
                ),
                x + 1,
            ),
            (
                "qsa.indexer_decode",
                lambda: qsa.mqa_decode(lambda a, *args: a.sum(-1), q, k, None, None, 3),
                q.sum(-1),
            ),
            (
                "qsa.indexer_prefill",
                lambda: mqa.qsa_mqa_prefill(q, k, starts, ends),
                mqa.torch_qsa_mqa_prefill(q, k, starts, ends),
            ),
        ]
        for operation, call, expected in cases:
            with self.subTest(operation=operation):
                with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                    torch.testing.assert_close(call(), expected)
                with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                    with self.assertRaises(V100FallbackError) as caught:
                        call()
                    self.assertIn(operation, str(caught.exception))
                    self.assertIn("device=cpu", str(caught.exception))

    def test_native_moe_alternate_dispatch_is_explicit(self):
        h = torch.ones(3, 2)
        dispatch = SimpleNamespace(
            hidden_states=h,
            topk_output=SimpleNamespace(topk_ids=torch.zeros(3, 10, dtype=torch.int32)),
        )
        quant = SimpleNamespace(w13_qweight=torch.zeros(1), w2_qweight=torch.zeros(1))
        original = lambda *args: (
            lambda dispatch, quant, config: dispatch.hidden_states * 2
        )
        call = quantization.moe_runner(original, None, "none", "marlin")
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
            torch.testing.assert_close(call(dispatch, quant, None), h * 2)
        with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
            with self.assertRaisesRegex(V100FallbackError, "moe.nvfp4_decode"):
                call(dispatch, quant, None)

    def test_cached_missing_cache_operator_cannot_silently_fall_back(self):
        x = torch.zeros(3, 1, 256, dtype=torch.float16)
        cache = torch.zeros(4, 1, 256, dtype=torch.uint8)
        locations = torch.arange(3)
        with patch.object(sm70_fp8_kv, "_get_op", return_value=None):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                self.assertFalse(
                    sm70_fp8_kv.write_fp8_e5m2_cache_sm70(x, x, cache, cache, locations)
                )
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                with self.assertRaisesRegex(
                    V100FallbackError, "operator is unavailable"
                ):
                    sm70_fp8_kv.write_fp8_e5m2_cache_sm70(x, x, cache, cache, locations)

    def test_qsa_attention_fallback_boundaries(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(qsa.QwenSparseAttnBackend)
        cache = torch.zeros(4, 1, 256, dtype=torch.uint8)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda layer: cache,
            get_value_buffer=lambda layer: cache,
        )
        backend.qsa_profile = None
        backend._resolve_metadata = lambda batch: None
        x = torch.ones(3, 6, 256)
        layer = SimpleNamespace(tp_q_head_num=6, head_dim=256, layer_id=0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=None,
            extend_seq_lens_cpu=None,
        )
        indices = torch.zeros(3, 2, dtype=torch.int32)
        original = lambda owner, q, *args, **kwargs: q.flatten(1)
        with (
            patch.object(qsa.BaseQSA, "forward_extend", original),
            patch.object(qsa.BaseQSA, "_forward_paged_attention", original),
        ):
            cases = [
                (
                    "qsa.prefill",
                    lambda: backend.forward_extend(
                        x, None, None, layer, batch, topk_indices=indices
                    ),
                ),
                (
                    "qsa.paged_attention",
                    lambda: backend._forward_paged_attention(x, layer, batch, indices),
                ),
            ]
            for operation, call in cases:
                with self.subTest(operation=operation):
                    with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(False):
                        torch.testing.assert_close(call(), x.flatten(1))
                    with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                        with self.assertRaisesRegex(V100FallbackError, operation):
                            call()

    def test_speculative_qsa_parent_delegation_is_not_a_fallback(self):
        """The base extend method routes verification into our paged backend."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(qsa.QwenSparseAttnBackend)
        backend.token_to_kv_pool = None
        x = torch.ones(3, 6, 256)
        layer = SimpleNamespace(tp_q_head_num=6, head_dim=256)
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        indices = torch.zeros(3, 2, dtype=torch.int32)
        with patch.object(
            qsa.BaseQSA, "forward_extend", lambda owner, q, *a, **kw: q.sum(-1)
        ):
            with envs.SGLANG_DEBUG_V100_STRICT_DISPATCH.override(True):
                torch.testing.assert_close(
                    backend.forward_extend(
                        x, None, None, layer, batch, topk_indices=indices
                    ),
                    x.sum(-1),
                )


if __name__ == "__main__":
    unittest.main()
