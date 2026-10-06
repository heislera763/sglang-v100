"""V100 KDA prefill via the existing numerically checked recurrent kernel."""

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


class SM70KDAKernel(TritonKDAKernel):
    supports_packed_decode = False
    supports_track_state_snapshot = False

    def extend(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
        beta_is_raw=False,
        return_intermediate_states=False,
        **kwargs,
    ):
        if (
            return_intermediate_states
            or kwargs.get("track_state") is not None
            or kwargs.get("is_spec_decode")
        ):
            raise NotImplementedError(
                "V100 recurrent KDA requires ordinary decode and no_buffer radix state"
            )
        if A_log is not None:
            raw = g.float() + dt_bias.float().reshape(1, 1, q.shape[2], -1)
            rate = A_log.float().exp().reshape(1, 1, q.shape[2], 1)
            g = (
                -rate * F.softplus(raw)
                if lower_bound is None
                else lower_bound * torch.sigmoid(rate * raw)
            )
        if beta_is_raw:
            beta = beta.sigmoid().reshape(q.shape[:3])
        indices = cache_indices.long()
        initial = ssm_states.index_select(0, indices)
        output, final = fused_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states.index_copy_(0, indices, final)
        return output
