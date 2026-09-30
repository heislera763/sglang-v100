"""The installed plugin must not import kernels or register hooks by default."""

import os, sys

os.environ.pop("SGLANG_V100_LITE", None)
import sglang_v100_lite

sglang_v100_lite.register()
assert "sglang_v100_lite.bootstrap" not in sys.modules
print("Plugin disabled: no bootstrap or kernel imports")
