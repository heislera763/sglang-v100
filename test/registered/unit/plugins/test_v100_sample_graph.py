"""Unsupported sampling features must keep the existing sampler contract."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_plus"), *sys.path]
):
    from sglang_v100_plus.sample_graph import eligible, sample_graph


def inputs():
    info = SimpleNamespace(
        is_any_greedy=False,
        need_min_p_sampling=False,
        sampling_seed=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        has_custom_logit_processor=False,
        return_sampling_masks=[False],
    )
    batch = SimpleNamespace(
        sampling_info=info,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([32]),
        has_grammar=False,
        return_logprob=False,
    )
    plan = SimpleNamespace(
        tree_topk=1,
        draft_token_num=3,
        max_tree_depth=3,
        draft_probs=torch.zeros(1, 2, 8),
    )
    return plan, batch, SimpleNamespace(next_token_logits=torch.ones(3, 8))


class TestV100SampleGraph(CustomTestCase):
    def test_supported_policy_is_independent_of_tp_and_pp_placement(self):
        for tp, pp in ((4, 1), (4, 2), (8, 1)):
            with get_context().override_server_args(
                model_path="dummy",
                tp_size=tp,
                pp_size=pp,
                disable_overlap_schedule=True,
                max_running_requests=1,
                speculative_algorithm="EAGLE",
                speculative_use_rejection_sampling=True,
            ):
                plan, batch, _ = inputs()
                self.assertTrue(eligible(plan, batch, None, None))

    def test_feature_exclusions_preserve_all_sampler_arguments(self):
        cases = (
            ("return_logprob", True),
            ("has_grammar", True),
            ("is_any_greedy", True),
            ("need_min_p_sampling", True),
            ("sampling_seed", torch.tensor([531])),
            ("acc_additive_penalties", torch.zeros(1, 8)),
            ("acc_scaling_penalties", torch.ones(1, 8)),
            ("logit_bias", torch.zeros(1, 8)),
            ("has_custom_logit_processor", True),
            ("return_sampling_masks", [True]),
        )
        with (
            get_context().override_server_args(
                model_path="dummy",
                disable_overlap_schedule=True,
                max_running_requests=1,
                speculative_algorithm="EAGLE",
                speculative_use_rejection_sampling=True,
            ),
            patch(
                "sglang_v100_plus.sample_graph.SampleGraph",
                side_effect=AssertionError("GPU allocation"),
            ),
        ):
            for name, value in cases:
                plan, batch, output = inputs()
                owner = batch if hasattr(batch, name) else batch.sampling_info
                setattr(owner, name, value)
                calls = []

                def ordinary(*args, **kwargs):
                    calls.append((args, kwargs))
                    return "ordinary"

                self.assertEqual(
                    sample_graph(ordinary, plan, batch, output), "ordinary"
                )
                self.assertEqual(len(calls), 1)
                self.assertIs(calls[0][0][0], plan)
                self.assertIs(calls[0][0][1], batch)
                self.assertIs(calls[0][0][2], output)

    def test_incomplete_and_branched_trees_do_not_allocate_a_graph(self):
        with get_context().override_server_args(
            model_path="dummy",
            disable_overlap_schedule=True,
            max_running_requests=1,
            speculative_algorithm="EAGLE",
            speculative_use_rejection_sampling=True,
        ):
            for topk, depth in ((2, 3), (1, 4)):
                plan, batch, _ = inputs()
                plan.tree_topk, plan.max_tree_depth = topk, depth
                self.assertFalse(eligible(plan, batch, None, None))


if __name__ == "__main__":
    unittest.main()
