"""Fail closed if any required V100 plugin hook could not be installed."""

import os
import sys
from sglang.launch_server import run_server
from sglang.srt.plugins import load_plugins
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if os.environ.get("SGLANG_V100_LITE") != "1":
    raise RuntimeError("Use SGLANG_V100_LITE=1 to select the compatibility runtime")
load_plugins()
from . import runtime

required = getattr(runtime, "REQUIRED_HOOKS", None)
if not required or required - HookRegistry._patched:
    raise RuntimeError(
        f"V100 compatibility installation incomplete: {required - HookRegistry._patched if required else 'registration failed'}"
    )
try:
    run_server(prepare_server_args(sys.argv[1:]))
finally:
    kill_process_tree(os.getpid(), include_parent=False)
