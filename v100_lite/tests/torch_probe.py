"""Record actual SM70 execution, not just CUDA discovery."""

import json, os, traceback
import torch

result = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "arches": torch.cuda.get_arch_list(),
}
for name, fn in [
    ("cuda_add", lambda: (torch.ones(32, device="cuda") + 1).sum().item()),
    (
        "fp16_matmul",
        lambda: (
            (
                torch.ones((128, 128), device="cuda", dtype=torch.float16)
                @ torch.ones((128, 128), device="cuda", dtype=torch.float16)
            )
            .float()
            .mean()
            .item()
        ),
    ),
]:
    try:
        result[name] = {"passed": True, "value": fn()}
    except Exception as e:
        result[name] = {"passed": False, "error": str(e)}
try:
    import triton, triton.language as tl

    @triton.jit
    def add(X, Y, N: tl.constexpr):
        i = tl.arange(0, 128)
        tl.store(Y + i, tl.load(X + i) + 1)

    x = torch.ones(128, device="cuda")
    y = torch.empty_like(x)
    add[(1,)](x, y, 128)
    torch.cuda.synchronize()
    result["triton"] = {
        "version": triton.__version__,
        "passed": bool(torch.all(y == 2).item()),
    }
except Exception as e:
    result["triton"] = {"passed": False, "error": str(e)}
print(json.dumps(result, indent=2))
