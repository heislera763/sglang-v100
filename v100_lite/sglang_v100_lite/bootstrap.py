"""Opt-in SM70 dispatch. No registration occurs unless explicitly enabled."""

import os
from functools import wraps


def install():
    from .api_key import redact_logs

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
    from .sampling import top_k_renorm_probs, top_p_renorm_probs

    # EAGLE imports these functions inside verification, independently of the
    # ordinary sampler backend. Keep its algorithm and use our SM70 AOT ops.
    HookRegistry.register(
        "flashinfer.sampling.top_k_renorm_probs", top_k_renorm_probs, HookType.REPLACE
    )
    HookRegistry.register(
        "flashinfer.sampling.top_p_renorm_probs", top_p_renorm_probs, HookType.REPLACE
    )
    from .native_api import moe_align_block_size, moe_sum_reduce

    HookRegistry.register(
        "sgl_kernel.moe.moe_align_block_size", moe_align_block_size, HookType.AROUND
    )
    HookRegistry.register(
        "sgl_kernel.moe.moe_sum_reduce", moe_sum_reduce, HookType.AROUND
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
    from .linear_attention import dispatcher_init

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

    from .gdn_recurrent import fused_sigmoid_gating_delta_rule_update

    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update",
        fused_sigmoid_gating_delta_rule_update,
        HookType.REPLACE,
    )
    from .kv_cache import store

    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MHATokenToKVPool.set_kv_buffer",
        store,
        HookType.AROUND,
    )
    from .api_key import arguments

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
    from .linear import apply

    HookRegistry.register(
        "sglang.srt.layers.quantization.unquant.UnquantizedLinearMethod.apply",
        apply,
        HookType.AROUND,
    )
    global REQUIRED_HOOKS

    from .hyperconnection import initialize, mix, combine, embedding_output

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
