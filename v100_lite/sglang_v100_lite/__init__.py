"""Opt-in SM70 compatibility for current SGLang."""

import os


def register():
    if os.environ.get("SGLANG_V100_LITE") != "1":
        return
    from .bootstrap import install

    install()
