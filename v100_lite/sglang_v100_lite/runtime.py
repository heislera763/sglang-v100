"""Opt-in SM70 runtime support."""

import os
from functools import wraps


def install():
    from .runtime import redact_logs

    redact_logs()
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("SGLANG_V100_LITE=1 requires an SM70 CUDA device")
    # Prefer the built SM70 norm ops and Triton LayerNorm over FlashInfer's
    # SM75+ JIT. These modules retain their mainline public signatures.
    import sgl_kernel.elementwise as norm_ops
    import sglang.srt.layers.layernorm as norms

    norm_ops._has_flashinfer = False
    norms._flashinfer_layernorm_available = False
    norms._flashinfer_rmsnorm_quant_available = False
    # Fused indexer kernels use the model's FP16 type for pending/compressed keys.
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    QSATokenToKVPool.index_state_dtype = torch.float16
    from sglang.srt.arg_groups.choices import add_linear_attn_kernel_backend_choices

    add_linear_attn_kernel_backend_choices(["tilelang_v100"])

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from .runtime import top_k_renorm_probs, top_p_renorm_probs

    # EAGLE imports these functions inside verification, independently of the
    # ordinary sampler backend. Keep its algorithm and use our SM70 AOT ops.
    HookRegistry.register(
        "flashinfer.sampling.top_k_renorm_probs", top_k_renorm_probs, HookType.REPLACE
    )
    HookRegistry.register(
        "flashinfer.sampling.top_p_renorm_probs", top_p_renorm_probs, HookType.REPLACE
    )
    def legacy_dtype(original, model_config):
        if model_config.dtype != torch.float16:
            raise ValueError("The enabled SM70 profile requires float16 model weights")
        # install() already requires SM70; upstream rejects everything below SM75.

    HookRegistry.register(
        "sglang.srt.model_executor.model_runner_components.load_model_utils.maybe_downgrade_dtype_for_legacy_gpu",
        legacy_dtype,
        HookType.AROUND,
    )
    from .quantization import (
        prepare_nvfp4_moe,
        minimum_capability,
        marlin_gemm,
        moe_runner,
    )
    from .runtime import dispatcher_init

    HookRegistry.register(
        "sglang.srt.layers.quantization.modelopt_quant.ModelOptFp4Config.get_min_capability",
        minimum_capability,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.modelopt_quant.prepare_moe_nvfp4_layer_for_marlin",
        prepare_nvfp4_moe,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.moe.moe_wna16_marlin.moe_wna16_marlin_gemm",
        marlin_gemm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.linear.gdn_backend.GDNKernelDispatcher.__init__",
        dispatcher_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.base.FusedOpPool.get_fused_func",
        moe_runner,
        HookType.AROUND,
    )

    from .runtime import round_verify_state

    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_kernel.run",
        round_verify_state,
        HookType.AROUND,
    )
    from .runtime import store

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer",
        store,
        HookType.AROUND,
    )
    from .runtime import arguments

    HookRegistry.register(
        "sglang.srt.server_args.prepare_server_args", arguments, HookType.AROUND
    )

    from .qsa import QwenSparseAttnBackend, project_qk, mqa_decode
    from .mqa import qsa_mqa_prefill

    HookRegistry.register(
        "sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend",
        QwenSparseAttnBackend,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer.project_qk",
        project_qk,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.qsa.mqa.qsa_mqa_decode",
        mqa_decode,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.attention.qsa.mqa.qsa_mqa_prefill",
        qsa_mqa_prefill,
        HookType.REPLACE,
    )

    from .ple import _gather_ple_embedding_from_pinned_kernel

    HookRegistry.register(
        "sglang.srt.models.qwen4_exp._gather_ple_embedding_from_pinned_kernel",
        _gather_ple_embedding_from_pinned_kernel,
        HookType.REPLACE,
    )

    from .ple import Qwen4ExpPinnedHostEmbedding

    HookRegistry.register(
        "sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding",
        Qwen4ExpPinnedHostEmbedding,
        HookType.REPLACE,
    )
    from .runtime import apply_unquant

    HookRegistry.register(
        "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
        apply_unquant,
        HookType.AROUND,
    )
    global REQUIRED_HOOKS

    from .runtime import initialize, mix, combine, embedding_output

    HookRegistry.register(
        "sglang.srt.layers.hyperconnection.GatedResidual.__init__",
        initialize,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.hyperconnection.GatedResidual.mix", mix, HookType.AROUND
    )
    HookRegistry.register(
        "sglang.srt.layers.hyperconnection.GatedResidual.combine",
        combine,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding._finish_embedding_lookup",
        embedding_output,
        HookType.AROUND,
    )
    REQUIRED_HOOKS = frozenset(HookRegistry._hooks)

import os


def arguments(original, *args, **kwargs):
    result = original(*args, **kwargs)
    if result.api_key is None:
        result.api_key = os.environ.get("LLAMA_API_KEY")
    return result


def redact_logs():
    import logging

    secret = os.environ.get("LLAMA_API_KEY")
    if not secret:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        message = record.getMessage()
        if secret in message:
            record.msg = message.replace(secret, "[REDACTED]")
            record.args = ()
        return record

    logging.setLogRecordFactory(factory)

import torch
import msgspec
import os


def initialize(original, self, config, *args, **kwargs):
    config = msgspec.structs.replace(config, params_dtype=torch.float16)
    return original(self, config, *args, **kwargs)


def mix(original, self, hyper_input):
    if (
        hyper_input.ndim != 2
        or hyper_input.shape[0] not in (1, 2, 4)
        or hyper_input.shape[1] != 10240
        or hyper_input.dtype != torch.float16
    ):
        return original(self, hyper_input)
    if self.config.hc_per_branch_norm:
        normed = self.hc_norm(hyper_input)
    else:
        normed = self.hc_norm(
            hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        ).flatten(-2)
    if not sm70_hc_down_gemv_silu_supported(
        normed, self.input_mix_weight_down.weight, self.input_mix_weight_up.weight
    ):
        return original(self, hyper_input)
    from .kernels.sm70_hc_mix import gate_supported, hc_down_with_gate, hc_down, hc_up

    gate = None
    if getattr(self, "_split_combine_ok", False) and gate_supported(
        normed, self.block_inject_weight.weight
    ):
        down, gate = hc_down_with_gate(
            normed, self.input_mix_weight_down.weight, self.block_inject_weight.weight
        )
    else:
        down = hc_down(normed, self.input_mix_weight_down.weight)
    result = hc_up(down, normed, self.input_mix_weight_up.weight).to(self.params_dtype)
    state = (hyper_input, normed) if gate is None else (hyper_input, normed, gate)
    return result, state


def combine(original, self, block_output, residuals):
    if len(residuals) == 3:
        from .kernels.sm70_hc_mix import hc_apply_gate

        return hc_apply_gate(block_output, residuals[0], residuals[2])
    return original(self, block_output, residuals)


def embedding_output(original, self, *args, **kwargs):
    return original(self, *args, **kwargs).to(torch.float16)


def sm70_hc_down_gemv_silu_supported(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (7, 0)
        and x.dtype == torch.float16
        and w_down.dtype == torch.float16
        and w_up.dtype == torch.float16
        and w_down.device == x.device
        and w_up.device == x.device
        and x.ndim == 2
        and x.shape[1] == 10240
        and (
            x.shape[0] == 1
            or (
                x.shape[0] in (2, 4)
                and os.environ.get("SGLANG_SM70_HC_NATIVE", "1") == "1"
                and os.environ.get("SGLANG_SM70_MTP_HC", "1") == "1"
                and x.data_ptr() % 16 == 0
                and w_down.data_ptr() % 16 == 0
                and w_up.data_ptr() % 16 == 0
            )
        )
        and w_down.shape == (320, 10240)
        and w_up.shape == (10240, 320)
        and x.is_contiguous()
        and w_down.is_contiguous()
        and w_up.is_contiguous()
    )

def store(
    original,
    self,
    layer,
    loc_info,
    key,
    value,
    k_scale=None,
    v_scale=None,
    layer_id_override=None,
    dcp_kv_mask=None,
):
    import torch

    if self.dtype == torch.float8_e5m2 and not self.use_hnd and dcp_kv_mask is None:
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
        from .kernels.sm70_fp8_kv import write_fp8_e5m2_cache_sm70

        loc, _, _ = unwrap_write_loc(loc_info)
        idx = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        ) - self.start_layer
        if write_fp8_e5m2_cache_sm70(
            key, value, self.k_buffer[idx], self.v_buffer[idx], loc, k_scale, v_scale
        ):
            return
    return original(
        self,
        layer,
        loc_info,
        key,
        value,
        k_scale,
        v_scale,
        layer_id_override,
        dcp_kv_mask,
    )

def apply_unquant(original, self, layer, x, bias=None):
    from .kernels.gemm import supported, linear_dense

    if supported(x, layer.weight, bias):
        return linear_dense(x, layer.weight)
    return original(self, layer, x, bias)

def dispatcher_init(
    original, self, decode_backend, prefill_backend, verify_backend=None
):
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend

    use_tilelang = prefill_backend.is_custom()
    original(
        self,
        decode_backend,
        LinearAttnKernelBackend.TRITON if use_tilelang else prefill_backend,
        verify_backend,
    )
    if use_tilelang:
        from .gdn_tilelang import TileLangGDNKernel

        self.extend_kernel = TileLangGDNKernel()


def round_verify_state(original, *args, **kwargs):
    import torch

    states = kwargs.get("intermediate_states_buffer")
    kwargs["QUANTIZE_STATE_EACH_STEP"] = (
        states is not None
        and states.dtype != torch.float32
        and not kwargs.get("HAS_EAGLE_TREE_CUSTOM_ATTN_MASK", False)
    )
    return original(*args, **kwargs)

from sgl_kernel.sampling import (
    _top_k_renorm_probs_internal,
    _top_p_renorm_probs_internal,
)
from sgl_kernel.utils import _to_tensor_scalar_tuple


def top_k_renorm_probs(probs, top_k):
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def top_p_renorm_probs(probs, top_p):
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))
