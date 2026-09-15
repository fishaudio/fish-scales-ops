"""Cosine-similarity sanity test for blockscale_gemm vs torch native bf16."""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

import fish_scales_ops as fso


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(F.cosine_similarity(a, b, dim=0))


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-8))


def test_linear(M: int, N: int, K: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") / (K**0.5)

    ref = F.linear(x, w)  # bf16 @ bf16 -> bf16
    out = fso.gemm.linear_bf16(x, w)

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    print(
        f"linear  M={M:5d} N={N:5d} K={K:5d}  cos={cos_sim(out, ref):.6f}  "
        f"rel_err={rel_err(out, ref):.4e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print(
        f"Device: {torch.cuda.get_device_name(0)} "
        f"(sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})"
    )
    print()
    print("=== functional linear (FP8 1×128) ===")
    for M, N, K in (
        (128, 4096, 4096),
        (1024, 4096, 4096),
        (4096, 4096, 4096),
        (256, 2048, 768),   # K % 512 != 0: sm_120 partial final round + zero-padded scale word
        (2048, 8192, 8192) if not args.quick else (1024, 4096, 4096),
    ):
        test_linear(M, N, K)


if __name__ == "__main__":
    main()
