import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(**attrs):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(attrs)
    return scheduler


class TestPPCommOverlap(CustomTestCase):
    def test_chain_relay_keeps_accept_indices_for_recurrent_state_commit(self):
        """Earlier PP stages need accepted steps even when KV needs no compaction.

        Omitting chain indices left their recurrent state at the prompt and
        produced incorrect cold-cache Qwen tool arguments with PP + MTP.
        """
        scheduler = _make_scheduler(_pp_spec_relay=True)
        accept_indices = torch.tensor([[0, 1, -1, -1], [4, 5, 6, -1]])
        result = GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.tensor([[10, 11, -1, -1], [12, 13, 14, -1]]),
            accept_lens=torch.tensor([2, 3]),
            accept_index=accept_indices,
            new_seq_lens=torch.tensor([302, 403]),
            next_draft_input=EagleDraftInput(bonus_tokens=torch.tensor([11, 14])),
        )
        batch = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.EAGLE, return_logprob=False
        )
        with get_context().override_server_args(speculative_eagle_topk=1):
            payload = scheduler._pp_prepare_tensor_dict(result, batch)
        self.assertIn("spec_accept_index", payload)
        torch.testing.assert_close(payload["spec_accept_index"], accept_indices)

    def test_graph_proxy_send_records_forward_reuse_fence(self):
        comm_stream = FakeStream(4)
        work = Mock()
        works = [SimpleNamespace(work=work)]
        scheduler = _make_scheduler(
            pp_comm_stream=comm_stream,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
            device_module=SimpleNamespace(Event=FakeEvent),
        )

        scheduler._pp_commit_comm_work(works, fence_next_forward=True)

        work.wait.assert_called_once_with()
        self.assertEqual(works, [])
        self.assertIs(scheduler.pp_send_done_event.recorded_stream, comm_stream)

    def test_no_fence_event_without_comm_stream(self):
        scheduler = _make_scheduler(
            pp_comm_stream=None,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
        )

        scheduler._pp_commit_comm_work([SimpleNamespace(work=Mock())], True)

        self.assertIsNone(scheduler.pp_send_done_event)

    def test_forward_waits_for_graph_send_and_proxy_receive(self):
        schedule_stream = FakeStream(1)
        send_done_event = object()
        recv_event = object()
        forward_stream = Mock()
        scheduler = _make_scheduler(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            pp_send_done_event=send_done_event,
            pp_proxy_recv_event=recv_event,
        )

        scheduler._pp_wait_forward_dependencies()

        forward_stream.wait_stream.assert_called_once_with(schedule_stream)
        self.assertEqual(
            forward_stream.wait_event.call_args_list,
            [call(send_done_event), call(recv_event)],
        )
        self.assertIsNone(scheduler.pp_send_done_event)
        self.assertIsNone(scheduler.pp_proxy_recv_event)

    def test_inbox_returns_original_receive_event(self):
        recv_event = object()
        tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
        scheduler = _make_scheduler(
            _pp_tensor_dict_inbox=defaultdict(
                deque, {"output": deque([(tensor_dict, recv_event)])}
            ),
        )

        received, event = scheduler._pp_recv_typed_dict("output")

        self.assertIs(received, tensor_dict)
        self.assertIs(event, recv_event)


if __name__ == "__main__":
    unittest.main()
