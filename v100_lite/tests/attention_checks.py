"""Check SM70 attention and the FP16 speculative recurrence against references."""

import torch
from sglang_v100_lite.tilelang_attention._kernels_dense_d256 import (
    get_dense_prefix_d256_kernel,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as recurrent,
)

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_v100_lite.runtime import round_verify_state

HookRegistry.register(
    "sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update_kernel.run",
    round_verify_state,
    HookType.AROUND,
)
HookRegistry.apply_hooks()

assert torch.cuda.get_device_capability() == (7, 0)
torch.manual_seed(51)
q = torch.randn(64, 6, 256, device="cuda", dtype=torch.float16) * 0.1
k = torch.randn(64, 1, 256, device="cuda", dtype=torch.float16) * 0.1
v = torch.randn_like(k)
actual = get_dense_prefix_d256_kernel(6, 1)(q, k, v, 0, 256**-0.5)
expected = (
    torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0).repeat(1, 6, 1, 1),
        v.transpose(0, 1).unsqueeze(0).repeat(1, 6, 1, 1),
        is_causal=True,
    )
    .squeeze(0)
    .transpose(0, 1)
)
torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
print("D256 causal dense attention: SDPA reference agrees", flush=True)

# Current mainline uses compressed page 16; retain the fork's page 4 too.
# Random physical page order catches addressing errors hidden by identity maps.
from sglang_v100_lite.tilelang_attention._decode_cuda import (
    sm70_cuda_qsa_indexer_decode,
)

for page in (4, 16):
    queries = torch.randn(4, 4, 128, device="cuda", dtype=torch.float16)
    keys = torch.randn(80, page, 1, 128, device="cuda", dtype=torch.float16)
    pages = (
        torch.randperm(80, device="cuda")[: 4 * (64 // page)]
        .reshape(4, 64 // page)
        .int()
    )
    lengths = torch.tensor([1, 13, 31, 64], device="cuda", dtype=torch.int32)
    scores = sm70_cuda_qsa_indexer_decode(queries, keys, pages, lengths, 64, 128**0.5)
    for row, length in enumerate(lengths.tolist()):
        logical = keys[pages[row].long()].reshape(-1, 128)[:length].float()
        ref = (queries[row].float() @ logical.T).relu().sum(0) / (128**0.5)
        torch.testing.assert_close(scores[row, :length], ref, rtol=0.001, atol=0.0001)
    print(
        f"QSA indexer compressed page {page}: permuted-page reference agrees",
        flush=True,
    )

from sglang_v100_lite.tilelang_attention._decode_cuda import (
    sm70_cuda_qsa_prefill,
    sm70_cuda_qsa_decode,
)

# Sparse attention must respect selected logical rows, causal masks and physical
# addressing. Use a permuted cache and padded selections, not contiguous pages.
keys = torch.randn(129, 1, 256, device="cuda", dtype=torch.float16) * 0.1
values = torch.randn_like(keys)
keys = keys.to(torch.float8_e5m2)
values = values.to(torch.float8_e5m2)
req_table = torch.randperm(128, device="cuda").add(1).reshape(1, 128).int()


def selected_reference(queries, selections, lengths):
    outputs = []
    for row, length in enumerate(lengths):
        selected = selections[row].long()
        selected = selected[(selected >= 0) & (selected < length)]
        slots = req_table[0, selected].long()
        kk = keys[slots, 0].float()
        vv = values[slots, 0].float()
        probs = (queries[row].float() @ kk.T / (256**0.5)).softmax(-1)
        outputs.append(probs @ vv)
    return torch.stack(outputs).half()


for mode, rows in [("prefill", 11), ("decode", 4)]:
    queries = torch.randn(rows, 6, 256, device="cuda", dtype=torch.float16) * 0.1
    lengths = list(range(64 - rows + 1, 65))
    selections = torch.randint(0, 48, (rows, 17), device="cuda", dtype=torch.int32)
    selections[:, -1] = -1
    if mode == "prefill":
        actual = sm70_cuda_qsa_prefill(
            queries,
            keys,
            values,
            req_table,
            torch.tensor([0], device="cuda", dtype=torch.int32),
            selections,
            torch.tensor([64], device="cuda", dtype=torch.int32),
            256**-0.5,
        )
    else:
        actual = sm70_cuda_qsa_decode(
            queries,
            keys,
            values,
            req_table,
            torch.zeros(rows, device="cuda", dtype=torch.int32),
            selections,
            torch.tensor(lengths, device="cuda", dtype=torch.int32),
            256**-0.5,
        )
    expected = selected_reference(queries, selections, lengths)
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.002)
    print(f"QSA sparse {mode}: selected-row reference agrees", flush=True)

q = torch.randn(1, 4, 4, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(1, 4, 8, 128, device="cuda", dtype=torch.float16)
a = torch.randn(4, 8, device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
alog = torch.zeros(8, device="cuda", dtype=torch.float32)
bias = torch.zeros(8, device="cuda", dtype=torch.float16)
state = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float16) * 0.01
idx = torch.zeros(1, device="cuda", dtype=torch.int32)
snapshots = torch.empty(1, 4, 8, 128, 128, device="cuda", dtype=torch.float16)
common = dict(
    A_log=alog,
    dt_bias=bias,
    softplus_beta=1.0,
    softplus_threshold=20.0,
    initial_state_indices=idx,
    use_qk_l2norm_in_kernel=True,
)
actual = recurrent(
    a=a,
    b=b,
    q=q,
    k=k,
    v=v,
    initial_state_source=state,
    disable_state_update=True,
    intermediate_states_buffer=snapshots,
    intermediate_state_indices=idx,
    cache_steps=4,
    **common,
)
sequential = state.clone()
outputs = []
for t in range(4):
    outputs.append(
        recurrent(
            a=a[t : t + 1],
            b=b[t : t + 1],
            q=q[:, t : t + 1].contiguous(),
            k=k[:, t : t + 1].contiguous(),
            v=v[:, t : t + 1].contiguous(),
            initial_state_source=sequential,
            **common,
        )
    )
    torch.testing.assert_close(snapshots[:, t], sequential, rtol=0.001, atol=0.0001)
torch.testing.assert_close(actual, torch.cat(outputs, dim=1), rtol=0.001, atol=0.0001)
print("Four-step FP16 MTP recurrence: sequential decode agrees", flush=True)
