"""Boolean ``FSO_*`` environment switches, parsed the way the extension parses them.

The C++ side of the library reads a boolean switch as off when the variable is
unset, empty, or starts with the character ``0``, and as on for any other value
(``read_print_tile_info`` in
``csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/dispatch.cuh`` is the
reference). Every boolean ``FSO_*`` variable the Python package reads goes
through :func:`env_flag`, so ``FSO_DISABLE_DSL=0`` means "not disabled" here
exactly as ``FSO_PRINT_TILE_INFO=0`` means "do not print" in the extension.
Valued variables (paths, integers, sizes) keep their own parsing.
"""
from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    """Read the boolean switch ``name``.

    Unset or empty gives ``default``; a value whose first character is ``0``
    gives ``False``; any other value gives ``True``. With the default
    ``default=False`` this is the extension's rule exactly. A switch that is on
    unless turned off (``FSO_MOE_BLOCK_OVERLAP``) passes ``default=True``.
    """
    value = os.environ.get(name)
    if not value:
        return default
    return value[0] != "0"
