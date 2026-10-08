"""The local PP2 result must follow the output ring's FIFO and event order."""

import sys
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

with patch.object(
    sys, "path", [str(Path(__file__).resolve().parents[4] / "v100_lite"), *sys.path]
):
    from sglang_v100_lite.pipeline import (
        initialize_local_output,
        receive_output,
        send_output,
    )


class TestV100Pipeline(CustomTestCase):
    def owner(self, *, last=True, spec=SpeculativeAlgorithm.NONE):
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=last), spec_algorithm=spec
        )

    def test_result_fifo_preserves_all_fields_and_original_events(self):
        """Removing the echo must not shift one result into the next token turn."""
        owner = self.owner()
        with get_context().override_server_args(
            model_path="dummy",
            tp_size=4,
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        ):
            initialize_local_output(lambda _: None, owner)
        first_event, second_event = object(), object()
        first = SimpleNamespace(
            tensors={
                "next_token_ids": torch.tensor([7]),
                "logprobs": torch.tensor([-0.1]),
            }
        )
        second = SimpleNamespace(tensors={"next_token_ids": torch.tensor([11])})
        queue = deque([(first_event, first), (second_event, second)])
        sent = []

        def send(_, mb, mbs, results, pp_outputs):
            if mbs[mb] is not None:
                sent.append(results.popleft())
                return [object()]
            return []

        # Empty slots do not consume or retain anything. Two pending results
        # must remain FIFO even if the receiver processes them later.
        send_output(send, owner, 0, [None, True], queue, None)
        send_output(send, owner, 1, [None, True], queue, None)
        send_output(send, owner, 1, [None, True], queue, None)
        for expected_event, expected_proxy in sent:
            tensors, event = receive_output(None, owner)
            self.assertIs(event, expected_event)
            self.assertIs(tensors, expected_proxy.tensors)
        self.assertFalse(owner._v100_pp_local_outputs)
        owner.pp_group.is_last_rank = False
        # Stage zero still receives from the original ring, but must not echo.
        self.assertEqual(send_output(None, owner, 0, [True], deque(), first), [])
        self.assertEqual(receive_output(lambda _: "wire", owner), "wire")

    def test_skipped_sends_and_unsupported_schedules_keep_protocol(self):
        """A skipped prefill output must not become a later decode result."""
        owner = self.owner()
        with get_context().override_server_args(
            model_path="dummy",
            pp_size=2,
            max_running_requests=1,
            disable_overlap_schedule=True,
        ):
            initialize_local_output(lambda _: None, owner)
        queue = deque([(object(), SimpleNamespace(tensors={}))])

        def skip(_, mb, mbs, results, pp_outputs):
            results.popleft()
            return []

        self.assertEqual(send_output(skip, owner, 0, [True], queue, None), [])
        self.assertFalse(owner._v100_pp_local_outputs)
        for fields, spec in (
            ({"pp_size": 3}, SpeculativeAlgorithm.NONE),
            ({"pp_async_batch_depth": 1}, SpeculativeAlgorithm.NONE),
            ({"max_running_requests": 2}, SpeculativeAlgorithm.NONE),
            ({"disable_overlap_schedule": False}, SpeculativeAlgorithm.NONE),
            ({"disaggregation_mode": "prefill"}, SpeculativeAlgorithm.NONE),
            ({}, SpeculativeAlgorithm.EAGLE),
        ):
            with self.subTest(fields=fields, spec=spec):
                defaults = {
                    "model_path": "dummy",
                    "pp_size": 2,
                    "max_running_requests": 1,
                    "disable_overlap_schedule": True,
                }
                defaults.update(fields)
                with get_context().override_server_args(**defaults):
                    owner = self.owner(spec=spec)
                    initialize_local_output(lambda _: None, owner)
                self.assertIsNone(owner._v100_pp_local_outputs)
                self.assertEqual(
                    send_output(lambda *args: "send", owner, 0, [], deque(), None),
                    "send",
                )
                self.assertEqual(receive_output(lambda _: "receive", owner), "receive")


if __name__ == "__main__":
    unittest.main()
