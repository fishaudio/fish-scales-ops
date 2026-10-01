"""The sm_120/sm_121 MXFP8 attention backends.

Their entry points are exported from :mod:`fish_scales_ops.attention`; import
them from there.
"""


def _require_sm120(name: str) -> None:
    """The native attention kernels exist for sm_120/sm_121 only. Refuse other
    architectures up front, with the alternative, instead of failing inside the op."""
    from ..._arch import sm_major
    major = sm_major()
    if major != 12:
        raise NotImplementedError(
            f"fish_scales_ops.attention.{name} runs the sm_120/sm_121 MXFP8 attention kernel and this "
            f"device is sm_{major}x. On sm_90 and sm_100/103 fso has no native attention kernel; use "
            "fish_scales_ops.attention.flash_attn_fwd (torch SDPA).")
