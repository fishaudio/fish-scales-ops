"""Cached compute-capability probe shared by ops/* modules.

`cudaDeviceGetAttribute` is forbidden during CUDA graph capture, so the
result is resolved once on first use and cached.
"""
from __future__ import annotations

import torch

_sm_major_cache: int = -1


def sm_major() -> int:
    global _sm_major_cache
    if _sm_major_cache < 0:
        if torch.cuda.is_available():
            _sm_major_cache = torch.cuda.get_device_capability(0)[0]
        else:
            _sm_major_cache = 0
    return _sm_major_cache
