"""Request video controls must not leak into shared defaults or image caches."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Video(VideoDecoderWrapper):
    # Fake the decoder boundary; execute real Qwen frame sampling/resizing.
    avg_fps = 8.0

    def __init__(self):
        pass

    def __len__(self):
        return 32

    def get_frames_as_tensor(self, indices):
        return torch.tensor(indices, dtype=torch.uint8)[:, None, None, None].expand(
            -1, 28, 28, 3
        )


class TestQwenVideoConfig(CustomTestCase):
    def test_request_overrides_are_isolated_and_video_cache_identity_changes(self):
        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        defaults = dict(nframes=8, min_pixels=28 * 28, max_pixels=28 * 28)
        processor.video_config = defaults
        processor.model_type = "qwen4_exp"
        processor.hf_config = SimpleNamespace(model_type="qwen4_exp")
        processor.mm_tokens = SimpleNamespace(
            image_token_id=1, video_token_id=2, audio_token_id=None
        )
        processor.vision_start_token_id = 3
        processor.vision_end_token_id = 4
        processor.load_mm_data = AsyncMock(
            side_effect=lambda **kwargs: SimpleNamespace(videos=[_Video()])
        )
        processor._mark_cuda_ipc_features_for_deferred_reconstruction = Mock()
        processor._get_grid_from_output_or_items = Mock(return_value=None)
        processor._get_precomputed_mrope_from_output = Mock(
            return_value=(torch.zeros(3, 2, dtype=torch.long), torch.tensor([0]))
        )
        processor.prepare_media_artifacts = AsyncMock(
            side_effect=AssertionError("video must bypass the image artifact cache")
        )
        seen = []

        async def combine(base, tokens, **kwargs):
            # Yield across requests to expose shared-config mutation or reuse.
            await asyncio.sleep(0)
            video = base.videos[0]
            config = kwargs["processor_video_config"]
            self.assertNotIn("_question", config)
            seen.append((video.shape[0], dict(config)))
            item = MultimodalDataItem(modality=Modality.VIDEO, feature=video)
            item.set_pad_value()
            return [item], torch.tensor([1, 2]), {"padded_input_ids": [1, 2]}

        processor.process_and_combine_mm_data_async = combine
        requests = [
            SimpleNamespace(video_data=["clip"], audio_data=None, video_config=config)
            for config in (
                {"nframes": 4, "_question": "internal"},
                None,
                {"nframes": 4},
            )
        ]

        async def run():
            return await asyncio.gather(
                *(
                    processor.process_mm_data_async(None, "video", req)
                    for req in requests
                )
            )

        with patch(
            "sglang.srt.multimodal.processors.qwen_vl.is_cpu", return_value=True
        ):
            results = asyncio.run(run())
        self.assertEqual([count for count, _ in seen], [4, 8, 4])
        self.assertEqual(defaults, dict(nframes=8, min_pixels=784, max_pixels=784))
        self.assertEqual(
            requests[0].video_config, {"nframes": 4, "_question": "internal"}
        )
        hashes = [result.mm_items[0].hash for result in results]
        self.assertEqual(hashes[0], hashes[2])
        self.assertNotEqual(hashes[0], hashes[1])

        # Already decoded frames have no sampling metadata; their effective
        # settings must still reach the HF processor rather than its defaults.
        processor.load_mm_data.side_effect = lambda **kwargs: SimpleNamespace(
            videos=[torch.zeros(8, 3, 28, 28)]
        )
        asyncio.run(run())
        self.assertEqual([config["nframes"] for _, config in seen[-3:]], [4, 8, 4])
        self.assertEqual(defaults, dict(nframes=8, min_pixels=784, max_pixels=784))


if __name__ == "__main__":
    unittest.main()
