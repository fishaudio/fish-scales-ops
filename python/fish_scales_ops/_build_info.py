"""How this copy of fish_scales_ops was built, and the import-time torch check.

A wheel built by ``scripts/build_wheel.sh`` carries ``BUILD_INFO.json`` next to
this file: the package version and commit, the build image and toolchain, the
torch the extension was compiled against, the CUTLASS commit, the bundled NVRTC,
the architectures and the newest glibc symbol version the extension needs. An
in-place build of the source tree has no such file.

The extension is compiled against one torch build and its C++ ABI, so another
torch in the running process can crash inside the extension rather than fail
cleanly. ``check_torch`` turns that into an ``ImportError`` before
``fish_scales_ops/__init__.py`` loads ``_C``. This module therefore imports
nothing from the extension.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BUILD_INFO.json")


def read() -> Optional[dict]:
    """The contents of ``BUILD_INFO.json``, or None when the package has none
    (an in-place build). A file that cannot be parsed raises ImportError."""
    try:
        with open(FILE, encoding="utf-8") as f:
            info = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise ImportError(f"fish_scales_ops: cannot read its build information {FILE}: {e}") from e
    if not isinstance(info, dict):
        raise ImportError(f"fish_scales_ops: {FILE} does not hold a JSON object")
    return info


def check_torch(info: Optional[dict]) -> None:
    """Raise ImportError when ``info`` names the torch the extension was built
    against and the running torch is a different version."""
    built = info.get("torch") if info else None
    running = str(torch.__version__)
    if not built or running == built:
        return
    local = built.partition("+")[2]
    install = f"pip install 'torch=={built}'"
    if local.startswith("cu"):
        install += f" --index-url https://download.pytorch.org/whl/{local}"
    raise ImportError(
        f"fish_scales_ops {info.get('version', '?')} was built against torch {built}, but this process runs "
        f"torch {running}. The extension is compiled against one torch build, and loading it with another can "
        f"crash. Install the torch it was built against ({install}), or install a fish-scales-ops wheel built "
        f"against torch {running} (scripts/build_wheel.sh), or build fish-scales-ops in place against this torch "
        f"(scripts/build.sh).")
