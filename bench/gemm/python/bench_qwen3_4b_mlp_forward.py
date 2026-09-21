"""End-to-end MLP forward bench for Qwen3-4B.

The per-linear `bench_qwen3_4b_mlp.py` doesn't capture the host overhead
of chaining 5 ops back-to-back in a real forward (quantize → gate_up →
SwiGLU → quantize → down). This bench runs the full SwiGLU MLP block as
one closure, both eagerly and inside a CUDA-graph capture, so we see the
**actual** per-token MLP cost a production decode loop pays.

Pipeline (per call):

    BF16:
      gate_up = F.linear(x_bf16,  W_gate_up_bf16)
      h_bf16  = F.silu(gate) * up      where gate, up = gate_up.chunk(2, -1)
      y       = F.linear(h_bf16, W_down_bf16)

    BSFP8 (1×128):
      xq, sx     = quantize_1x128_fp8(x_bf16)         (+ repack on sm_120)
      gate_up_bf = linear_fp8(xq, Wgu_fp8, sx_p, sgu_p)
      h_bf16     = silu(gate) * up
      hq, sh     = quantize_1x128_fp8(h_bf16)          (+ repack)
      y          = linear_fp8(hq, Wd_fp8, sh_p, sd_p)

    MXFP8 (1×32):
      xq, sx     = quantize_1x32_fp8(x_bf16)
      gate_up_bf = linear_mxfp8(xq, Wgu_fp8, sx, sgu)
      h_bf16     = silu(gate) * up
      hq, sh     = quantize_1x32_fp8(h_bf16)
      y          = linear_mxfp8(hq, Wd_fp8, sh, sd)

Weights are pre-quantized + pre-packed at startup (production pattern —
model init does this once). Activations are quantised inside the timed
region.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F


# Qwen3-4B
HIDDEN = 2560
INTERMEDIATE = 9728
M_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # docs/perf/README.md §4


def _make_bf16_weights(device="cuda"):
    torch.manual_seed(0xc0ffee)
    w_gate_up = (torch.randn(2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device=device) / (HIDDEN ** 0.5)).contiguous()
    w_down = (torch.randn(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16, device=device) / (INTERMEDIATE ** 0.5)).contiguous()
    return w_gate_up, w_down


@torch.compile(mode="default", dynamic=False)
def _silu_chunk_mul(gu):
    """SwiGLU: split gu in two halves along last dim, return silu(gate) * up.

    Compiled (Inductor) — the eager F.silu(gate) * up path materialises a
    silu_out intermediate (~M*INTER bf16 of memory traffic) that Inductor
    can fuse away, cutting total mem traffic from 5/8 → 3/8 of the M*INTER
    BF16 tensor count. Save: ~160 µs at M=4096 INTER=9728 on a 170-SM
    Blackwell GPU.
    """
    gate, up = gu.chunk(2, dim=-1)
    return F.silu(gate) * up


def _build_mlp_fn(dtype, sm_major):
    import fish_scales_ops as fso
    w_gate_up_bf, w_down_bf = _make_bf16_weights()

    if dtype == "bf16":
        def fn(x_bf):
            gu = F.linear(x_bf, w_gate_up_bf)
            gate, up = gu.chunk(2, dim=-1)
            h = F.silu(gate) * up
            return F.linear(h, w_down_bf)
        return fn

    if dtype == "bsfp8":
        wgu_q, sgu = fso.gemm.quantize_128x128_fp8(w_gate_up_bf)
        wd_q, sd = fso.gemm.quantize_128x128_fp8(w_down_bf)
        if sm_major >= 10:
            sgu = fso.gemm.repack_fp8_wgt_scales(sgu)
            sd = fso.gemm.repack_fp8_wgt_scales(sd)

        if sm_major >= 10:
            # sm_120 / sm_100: use the fused single-kernel `quantize_1x128_fp8_packed`
            # which emits the int32-packed UE8M0 K-major scale layout directly,
            # skipping the separate `repack_fp8_act_scales` pass.
            def fn(x_bf):
                xq, sx = fso.gemm.quantize_1x128_fp8_packed(x_bf)
                gu = fso.gemm.linear_fp8(xq, wgu_q, sx, sgu)
                h = _silu_chunk_mul(gu)
                hq, sh = fso.gemm.quantize_1x128_fp8_packed(h)
                return fso.gemm.linear_fp8(hq, wd_q, sh, sd)
        else:
            # sm_90: deep_gemm path consumes FP32 scales directly. The
            # underlying `fp8bs_quantize_1x128` already routes to the fast
            # uint64 LDG.64 kernel when K%512==0.
            def fn(x_bf):
                xq, sx = fso.gemm.quantize_1x128_fp8(x_bf, use_ue8m0=False)
                gu = fso.gemm.linear_fp8(xq, wgu_q, sx, sgu)
                h = _silu_chunk_mul(gu)
                hq, sh = fso.gemm.quantize_1x128_fp8(h, use_ue8m0=False)
                return fso.gemm.linear_fp8(hq, wd_q, sh, sd)
        return fn

    if dtype == "mxfp8":
        if sm_major not in (10, 12):
            return None
        wgu_q, sgu = fso.gemm.quantize_1x32_fp8(w_gate_up_bf)
        wd_q, sd = fso.gemm.quantize_1x32_fp8(w_down_bf)

        def fn(x_bf):
            xq, sx = fso.gemm.quantize_1x32_fp8(x_bf)
            gu = fso.gemm.linear_mxfp8(xq, wgu_q, sx, sgu)
            # Fused silu(gate) * up + quantize → fp8 + packed scale,
            # no `h` intermediate.
            hq, sh = fso.gemm.silu_chunk_mul_quantize_1x32_fp8(gu)
            return fso.gemm.linear_mxfp8(hq, wd_q, sh, sd)
        return fn

    if dtype in ("smm", "smm_fast"):
        # cuBLAS MXFP8 path via torch.nn.functional.scaled_mm. Same closure
        # shape as the MXFP8 path so the comparison is apples-to-apples at
        # the MLP-block level (activation quantize → gate_up → SwiGLU →
        # activation quantize → down). Two variants:
        #   "smm"      — reference quantize via `to_mxfp` + `to_blocked`
        #                (~10 small kernels per call).
        #   "smm_fast" — same closure but wrapped in `torch.compile(default)`
        #                so Inductor fuses the quantize + blocked-layout
        #                permute into a single kernel. ~5-19× faster on the
        #                quantize step alone; works under outer CUDA-Graph
        #                capture because mode="default" suppresses Inductor's
        #                own cudagraph layer.
        if sm_major not in (10, 12):
            return None
        from torch.testing._internal.common_quantized import to_mxfp, to_blocked
        from torch._C import _ScalingType as ST, _SwizzleType as SW

        sgu_un, wgu_q = to_mxfp(w_gate_up_bf.contiguous(), 32, "mxfp8")
        sd_un,  wd_q  = to_mxfp(w_down_bf.contiguous(),    32, "mxfp8")
        sgu_b = to_blocked(sgu_un)
        sd_b  = to_blocked(sd_un)
        wgu_t = wgu_q.t()
        wd_t  = wd_q.t()

        def _quantize_to_blocked(t):
            s_un, q = to_mxfp(t.contiguous(), 32, "mxfp8")
            return q, to_blocked(s_un)

        if dtype == "smm_fast":
            _quantize_to_blocked = torch.compile(_quantize_to_blocked, mode="default", dynamic=False)

        def fn(x_bf):
            xq, sx_b = _quantize_to_blocked(x_bf)
            gu = F.scaled_mm(xq, wgu_t, sx_b, ST.BlockWise1x32, sgu_b, ST.BlockWise1x32,
                             swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                             output_dtype=torch.bfloat16)
            h = _silu_chunk_mul(gu)
            hq, sh_b = _quantize_to_blocked(h)
            return F.scaled_mm(hq, wd_t, sh_b, ST.BlockWise1x32, sd_b, ST.BlockWise1x32,
                               swizzle_a=SW.SWIZZLE_32_4_4, swizzle_b=SW.SWIZZLE_32_4_4,
                               output_dtype=torch.bfloat16)
        return fn

    raise ValueError(f"unknown dtype {dtype!r}")


def _busy_warm():
    """Optional DVFS settle before a cell's timing (same knob as
    bench_qwen3_4b_mlp.py / bench_moe_*.py): on machines without a clock
    lock (B300: idles at 120 MHz between subprocess cells, ramps to the
    flat 2032 MHz max under load) the 15-iteration eager warmup of a
    small-M cell is a few hundred µs of GPU work and does not finish the
    ramp. FSO_BENCH_WARM_MS=<ms> spins a dummy matmul for that long first.
    Default 0 → protocol identical to the locked-clock machines."""
    import time
    ms = int(os.environ.get("FSO_BENCH_WARM_MS", "0"))
    if ms <= 0:
        return
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.monotonic()
    while (time.monotonic() - t0) * 1000.0 < ms:
        a = a @ a * 1e-3  # keep values bounded; result reused to defeat DCE
    torch.cuda.synchronize()


def _time_graph(call, iters=50, warmup=15, repeats=3):
    _busy_warm()
    # Eager warmup (sets static cudaFuncSetAttribute guards, Params cache,
    # Stream-K pool — all must be hot before stream capture starts).
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        call()

    samples = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters)
    return statistics.median(samples)


def bench_worker(M):
    """Run one (M, all-dtypes) cell. Designed for subprocess use."""
    sm_major = torch.cuda.get_device_capability(0)[0]
    device = "cuda"
    torch.manual_seed(M * 1009)
    x_bf = (torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1).contiguous()

    out = {"M": M, "hidden": HIDDEN, "intermediate": INTERMEDIATE}

    # BF16 reference for cosine check
    fn_bf = _build_mlp_fn("bf16", sm_major)
    y_ref = fn_bf(x_bf)

    # CUDA-Graph capture-and-replay is the only metric we report. Eager
    # is dropped because it conflates kernel cost with PyTorch op dispatch
    # + cudaLaunchKernelEx overhead — both elided in production by graph
    # capture. torch.compile is also dropped: on top of cudagraph it adds
    # at most a few percent (Inductor can fuse silu·gate and pick a
    # Triton matmul autotune), and that win does not change kernel-side
    # tuning decisions, which is what this bench is meant to inform.
    for dtype in ("bf16", "bsfp8", "mxfp8", "smm", "smm_fast"):
        fn = _build_mlp_fn(dtype, sm_major)
        if fn is None:
            continue
        record = {}
        try:
            y = fn(x_bf)
            record["cos"] = float(F.cosine_similarity(y.float().flatten(), y_ref.float().flatten(), dim=0).item())
            record["graph_us"] = _time_graph(lambda: fn(x_bf)) * 1000.0
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        out[dtype] = record

    return out


def run_grid(out_path):
    import subprocess
    sm_major = torch.cuda.get_device_capability(0)[0]
    name = torch.cuda.get_device_name(0)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        f = open(out_path, "w")
    else:
        f = sys.stdout
    sm_cap = torch.cuda.get_device_capability(0)
    f.write(json.dumps({"_device": name, "_sm": sm_cap[0] * 10 + sm_cap[1]}) + "\n")
    f.flush()

    total = len(M_GRID)
    t_start = __import__("time").time()
    for i, M in enumerate(M_GRID):
        result = subprocess.run(
            [sys.executable, __file__, "--worker"],
            input=str(M).encode(),
            capture_output=True,
            check=True,
        )
        rec = json.loads(result.stdout.decode().strip())
        f.write(json.dumps(rec) + "\n")
        f.flush()
        elapsed = __import__("time").time() - t_start
        eta = elapsed / (i + 1) * (total - i - 1)
        cells = [(k, rec[k]) for k in ("bf16", "bsfp8", "mxfp8", "smm", "smm_fast") if k in rec and "graph_us" in rec[k]]
        summary = "  ".join(
            f"{k}: graph={v['graph_us']:.1f}" for k, v in cells
        )
        print(f"[{i+1:>2}/{total}] M={M:>4}  {summary}  (t={elapsed:.0f}s eta={eta:.0f}s)",
              flush=True)
    if out_path:
        f.close()


def render_md(jsonl_path, label):
    recs = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    meta = recs[0] if "_device" in recs[0] else {"_device": "?", "_sm": "?"}
    rows = [r for r in recs if "M" in r]

    lines = []
    lines.append(f"### {label} — {meta['_device']} (sm_{meta['_sm']})")
    lines.append("")
    has_mx       = any("mxfp8"    in r and "graph_us" in r.get("mxfp8",    {}) for r in rows)
    has_smm      = any("smm"      in r and "graph_us" in r.get("smm",      {}) for r in rows)
    has_smm_fast = any("smm_fast" in r and "graph_us" in r.get("smm_fast", {}) for r in rows)

    # Single `µs` column per dtype = CUDA-Graph capture-and-replay timing
    # for the full SwiGLU MLP forward (5 ops for FP8 paths, 2 for BF16).
    # `sMM` is the cuBLAS MXFP8 path with reference `to_mxfp + to_blocked`
    # quantize. `sMM-c` is the same path with the quantize wrapped in
    # `torch.compile(mode="default")` so Inductor fuses it into one kernel.
    cols_us  = ["BF16 µs", "BSFP8 µs"]
    cos_keys = [("BSFP8 cos", "bsfp8")]
    if has_mx:
        cols_us.append("MXFP8 µs");  cos_keys.append(("MXFP8 cos", "mxfp8"))
    if has_smm:
        cols_us.append("sMM µs");    cos_keys.append(("sMM cos", "smm"))
    if has_smm_fast:
        cols_us.append("sMM-c µs");  cos_keys.append(("sMM-c cos", "smm_fast"))
    head_cells = ["M"] + cols_us + [c for c, _ in cos_keys]
    sep_cells  = ["---:"] * len(head_cells)
    lines.append("| " + " | ".join(f"{c:>7}" for c in head_cells) + " |")
    lines.append("| " + " | ".join(f"{c:>7}" for c in sep_cells)  + " |")

    def cell(r, k, field):
        v = r.get(k, {}).get(field)
        if not isinstance(v, float):
            return "—"
        return f"{v:.4f}" if field == "cos" else f"{v:.2f}"

    for r in sorted(rows, key=lambda x: x["M"]):
        M = r["M"]
        parts = [f"{M:>5}"]
        parts += [f"{cell(r, 'bf16',  'graph_us'):>7}",
                  f"{cell(r, 'bsfp8', 'graph_us'):>8}"]
        if has_mx:       parts.append(f"{cell(r, 'mxfp8',    'graph_us'):>8}")
        if has_smm:      parts.append(f"{cell(r, 'smm',      'graph_us'):>8}")
        if has_smm_fast: parts.append(f"{cell(r, 'smm_fast', 'graph_us'):>8}")
        for _, k in cos_keys:
            parts.append(f"{cell(r, k, 'cos'):>9}")
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--out")
    p.add_argument("--format", action="store_true")
    p.add_argument("--from", dest="srcs", nargs="+",
                   help="LABEL=jsonl pairs for format mode")
    p.add_argument("--md-out", default="-")
    args = p.parse_args()

    if args.worker:
        M = int(sys.stdin.read().strip())
        rec = bench_worker(M)
        sys.stdout.write(json.dumps(rec) + "\n")
        return

    if args.run:
        run_grid(args.out)
        return

    if args.format:
        out = ""
        for src in args.srcs or []:
            label, path = src.split("=", 1)
            out += render_md(path, label) + "\n"
        if args.md_out == "-":
            sys.stdout.write(out)
        else:
            with open(args.md_out, "w") as f:
                f.write(out)
        return

    p.error("specify --run or --format or --worker")


if __name__ == "__main__":
    main()
