# Performance reference — methodology and shape families

This directory is the **only** place where fish-scales-ops performance numbers
are published (README carries a copy of a few hot rows, nothing else). Files
are split by domain, then by SM version:

| file | device | content |
|---|---|---|
| `gemm/sm90.md` | NVIDIA H200 (sm_90) | Family A / B / C GEMM tables |
| `gemm/sm120.md` | NVIDIA RTX 5090 (sm_120); RTX PRO 6000 optional | Family A / B / C GEMM tables |
| `gemm/sm100.md` | NVIDIA B300 (sm_103) | Family A / B / C GEMM tables (unlocked clocks) |
| `layer/sm90.md` | NVIDIA H200 (sm_90) | whole MLP / MoE block per family (A dense MLP, B routed MoE, C routed + shared expert) — definitions in `layer/README.md` |
| `layer/sm120.md` | NVIDIA RTX 5090 (sm_120) | whole-block tables |
| `layer/sm100.md` | NVIDIA B300 (sm_103) | whole-block tables (unlocked clocks) |
| `attention/sm90.md` | — | n/a (no native sm_90 attention kernel) |
| `attention/sm120.md` | NVIDIA RTX 5090 (sm_120) | prefill / paged decode / paged prefill |
| `attention/sm100.md` | — | n/a (no native sm_100 attention kernel) |

Status (2026-09-22): structure frozen; every sm_90, sm_120 and sm_103 table
is generated from `tests/baselines/`. The B300 (sm_103) GEMM and layer tables
were first filled on 2026-09-15, re-measured in full on 2026-09-17 and
re-measured in full twice on 2026-09-22 and again on 2026-09-23, most
recently from the run in `/data/bench-runs/b300_final7_20260923/`; the
2026-09-17 run and the two 2026-09-22 runs were each taken on a different card
of the same pod than the one before, and the 2026-09-23 run on the same card
as the last of them, so each replaced those tables rather than being merged
with them. They are unlocked-clock numbers and
labelled as such. The old single `perf.md` is archived
outside the repository (`../fso-doc_review-backup-20260915/repo/docs/perf.md`).
Rows marked `TBD` have no accepted baseline yet.

---

## 1. Timing protocol

The rules below are normative. Section 5 gives the clock locks, section 7
the acceptance rule and the regeneration steps, and section 8 the caveats a
reader of the tables needs.

- **One process per cell.** Each `(shape, M)` cell runs in its own subprocess
  so env knobs and JIT/autotune caches cannot leak between cells.
- **CUDA-graph replay median.** Per cell: eager warmup → side-stream warmup →
  `torch.cuda.graph` capture → replay batches timed with CUDA events; report
  the median. The reported `µs` is the graph-replay cost (what a production
  decode/prefill loop pays), never eager single-call time.
- **Locked clocks** (section 5). A run on an unlocked device is labelled as such
  in the environment block and is not comparable across commits.
- **Same commit, same device, same protocol** for any before/after claim.
  Baselines live in `tests/baselines/*.jsonl` and are regenerated together with
  the tables (section 7).
- **Accuracy gate** is part of the cell: a row is published only if its
  correctness check passed (bit-exact vs reference for BF16 paths, cosine ≥
  the documented threshold for FP8/MXFP8 paths). The gate value is a column.

## 2. Reported quantities

| quantity | definition |
|---|---|
| `µs` | graph-replay median per call (or per layer for grouped MoE), microseconds |
| `TFLOPS` | `FLOPs / µs × 1e-6`, FLOPs as defined per table kind below |
| `GB/s` (MoE tables) | active weight bytes read per layer / time |
| `cos` | cosine similarity vs the BF16 reference (FP8/MXFP8 rows) |

FLOPs conventions:

- **Dense GEMM** `[M,K] × [K,N]`: `2·M·N·K`.
- **Grouped MoE layer** (routing + quant + gate_up + silu·mul + down + combine),
  `M` tokens, `topk` experts per token, per-expert projections `(N₁,K₁)` and
  `(N₂,K₂)`: `2·M·topk·(N₁·K₁ + N₂·K₂)`. Only routed FLOPs count; glue kernels
  add time but no FLOPs (this is deliberate — the layer number is what serving pays).
- **Attention** forward, `B` batch, `H_q` query heads, `S_q × S_kv`, head dim `D`:
  `4·B·H_q·S_q·S_kv·D`; causal rows use half of that.

## 3. GEMM shape families (the only definition)

The GEMM tables use **three fixed shape families**, always in this order, with
these names. Adding a shape to a family or a fourth family is a documented
structural change, not a table edit.

### Family A — Qwen3-4B (dense)

Source: model config (hidden 2560, intermediate 9728, 32 query heads, 8 KV
heads, head_dim 128, 36 layers). Matches `bench/gemm/python/bench_qwen3_4b_mlp.py`.

| op | N | K | derivation |
|---|---|---|---|
| `wqkv` | 6144 | 2560 | (32 + 2·8) · 128 |
| `wo` | 2560 | 4096 | K = 32 · 128 |
| `gate_up` | 19456 | 2560 | 2 · 9728 fused |
| `down` | 2560 | 9728 | |

The legacy single `gate` row (N=9728, K=2560) is not part of the family; keep it
in the bench only for continuity if a pick decides so.

### Family B — Qwen3-30B-A3B (MoE)

Source: model config (hidden 2048, 128 routed experts, top-8, moe_intermediate
768, no shared expert, 48 layers). MoE geometry matches
`bench/gemm/python/bench_moe_qwen3_30a3.py`. Attention heads 32 query / 4 KV /
head_dim 128 are from the public config — **verify against a local
config.json before the first table is filled** (the checkout is not on this host).

| op | N | K | kind |
|---|---|---|---|
| `wqkv` | 5120 | 2048 | dense, (32 + 2·4) · 128 |
| `wo` | 2048 | 4096 | dense, K = 32 · 128 |
| `moe.gate_up` | 1536 | 2048 | grouped, per expert, 2 · 768 |
| `moe.down` | 2048 | 768 | grouped, per expert |

Reported units for the MoE part: **whole-layer µs** at `M` tokens (the 6-kernel
layer: routing, gather-quant, gate_up, silu-quant, down, combine) plus per-GEMM
grouped kernel µs. Family B on sm_120 is MXFP8 (1×32) — the sm_120 grouped
path is MXFP8-only — and on sm_90 it is block-FP8 via deep_gemm (K % 128).
(Until 2026-09-05 the sm_120 dense block-FP8 path also required K % 512, which
`moe.down`'s K = 768 fails; that limit is now K % 128, but there is still no
grouped block-FP8 kernel on sm_120.) The dtype column makes this explicit per
row.

### Family C — Qwen3.5-35B-A3B (hybrid MoE)

Source: `config.json` of `Qwen3.5-35B-A3B-Base` (read 2026-09-03 from
`apex-fish-inference/checkpoints/`): hidden 2048, 256 routed experts, top-8,
moe_intermediate 512, shared-expert intermediate 512, 40 layers = 30
linear-attention (Gated DeltaNet) + 10 full-attention (every 4th layer);
full attention 16 query heads / 2 KV heads / head_dim 256 with output gate
(`attn_output_gate`); linear attention 32 value heads × 128, 16 key heads × 128,
conv kernel 4.

| op | N | K | kind / applies to |
|---|---|---|---|
| `wqkv_gated` | 9216 | 2048 | dense, full-attn layers (10/40): q+gate 16·256·2 = 8192, k 512, v 512 |
| `wo` | 2048 | 4096 | dense, full-attn layers, K = 16 · 256 |
| `gdn.in_proj` | 12288 | 2048 | dense, linear-attn layers (30/40): q 2048 + k 2048 + v 4096 + z 4096 |
| `gdn.out_proj` | 2048 | 4096 | dense, linear-attn layers, K = 32 · 128 |
| `moe.gate_up` | 1024 | 2048 | grouped, per expert, 2 · 512 |
| `moe.down` | 2048 | 512 | grouped, per expert |
| `shared.gate_up` | 1024 | 2048 | dense (shared expert, all tokens) |
| `shared.down` | 2048 | 512 | dense (shared expert) |

Notes: `gdn.in_proj` is benched as one 12288-wide GEMM (the fused qkvz layout);
a stack that serves qkv (8192) + z (4096) separately pays two smaller GEMMs
instead. The tiny `in_proj_ba` (N = 64) and the GDN recurrence/conv are not
GEMM work and are out of scope. Bench entries: dense projections via
`bench_qwen3_4b_mlp.py --family qwen3.5-35a3` (tags `wqkv_gated`, `wo`,
`gdn_in_proj`, `gdn_out_proj`, `shared_gate_up`, `shared_down`; Family B's two
dense projections via `--family qwen3-30a3`); the routed MoE layer via
`bench_moe_qwen3_35a3.py` (= `bench_moe_qwen3_30a3.py --model qwen3.5-35a3`,
E=256, moe_inter 512). The shared expert is reported as dense rows, not folded
into the MoE layer number.

## 4. M grids

Identical across families and SMs so columns line up when diffing.

| band | M values |
|---|---|
| decode | 1, 2, 4, 8, 16, 32, 64, 128 |
| prefill | 256, 512, 1024, 2048, 4096, 8192 |

Off-grid points (96 for the MoE layer, 640 / 768 / 896 in the 2026-09-04 tile
sweeps) are allowed in sweeps and A/Bs through the benches' `--Ms` argument;
they are not baseline rows and do not appear in the tables. The benches'
default grids equal this table.

## 5. Devices and clock locks

Official numbers are taken with the clock locked at the no-boost frequency so
that runs are comparable across days and commits. CUDA 13 toolchain everywhere.

| SM | device | SMs | lock command | sustained clock | notes |
|---|---|---|---|---|---|
| sm_90 | NVIDIA H200 | 132 | `nvidia-smi -lgc 1980` | 1830 MHz | |
| sm_120 | NVIDIA GeForce RTX 5090 | 170 | `nvidia-smi -lgc 2407` | 2400 MHz | primary sm_120 device |
| sm_120 | NVIDIA RTX PRO 6000 Blackwell | 188 | `nvidia-smi -lgc 2430` | 2400 MHz | optional section |
| sm_103 | NVIDIA B300 | 148 | no lock available | ~2032 MHz flat under load | needs `FSO_BENCH_WARM_MS=300` — the dense bench, the MLP-layer forward bench (`bench_qwen3_4b_mlp_forward.py`, knob added 2026-09-15) and the MoE bench all honour it; label rows "unlocked" and read them with §8's B300 caveats |

Verify the lock took effect (`clocks.sm` reads the locked value under load)
before every run; record driver, CUDA, torch and the fso commit in the file's
environment block.

## 6. dtype matrix per SM

| SM | BF16 | BSFP8 (1×128 act / 128×128 wgt) | MXFP8 (1×32) | notes |
|---|---|---|---|---|
| sm_90 | reference column | ✓ deep_gemm JIT (dense + grouped) | — | K % 128 |
| sm_120 | reference column | ✓ CUTLASS block-scaled (dense; K % 128, UE8M0 activation and weight scales) | ✓ dense + grouped | Family B/C MoE rows are MXFP8; block-FP8 required K % 512 and had an FP32-weight-scale bug until 2026-09-05 (section 8) |
| sm_103 | reference column | ✓ since 2026-09-05: 1×128 scales expanded ×4 onto the MXFP8 tcgen05 tiers (same kernels and bytes as MXFP8; K % 128, N % 128); tables published 2026-09-15, re-measured 2026-09-17 and 2026-09-22 | ✓ dense + grouped since 2026-09-15 | grouped MoE (M3) landed 2026-09-15: CUTLASS pointer-array block-scaled kernel on the masked slab layout, eight kernels in the captured layer; cascade v2 and the programmatic dependent launch on the prep kernel and the grouped GEMM since 2026-09-17, and the dense path gained the wave-tile rule with its 64- and 192-wide N tiles on the same day; since 2026-09-22 the captured layer is five kernels in the decode band, where a slot-bound grouped route indexed by the routing kernel's packed active-expert list needs no argument-preparation launch and carries its own fused SwiGLU epilogue, and since 2026-09-23 five kernels at every M, because the pointer-array route reads its per-group problem shapes from the same routing kernel instead of an argument-preparation launch and the slot route's row-capacity clause is per kernel (the plain slot kernel to its full 64-wide tile, the fused-SwiGLU one to one 32-column chunk); the dense path additionally gained a vendored CuTe-DSL decode row, for M ≤ 32 on 2026-09-22 and extended to M ≤ 64 on 2026-09-23, which needs `nvidia-cutlass-dsl` 4.5.0 (the `sm100` extra) and is inert below it |

## 7. Regeneration

Tables are generated, never hand-edited. `bench/gemm/python/render_perf_docs.py`
rewrites every GEMM and layer table in `gemm/sm90.md`, `gemm/sm120.md`,
`gemm/sm100.md`, `layer/sm90.md`, `layer/sm120.md`, `layer/sm100.md` and the
two README hot tables from
`tests/baselines/*.jsonl`; `--check` exits non-zero if any table has drifted
from the baselines (run it before committing a baseline change). Prose around
the tables (provenance, readings) is edited by hand in the same commit.

1. Lock the clock (section 5) and verify.
2. Run the bench for the family/device; the jsonl lands outside the git tree
   (`/data/bench-runs/<run>/` on the H200 box, `/mnt/share/stone-bench-runs/<run>/`
   on the 5090 host).
3. Accept only if every affected cell is faster or within ±1% of the committed
   baseline. A MoE / MLP tile or cascade change is judged on the layer cell
   (section 8), and the kernel table records the consequence.
4. Copy the accepted jsonl to `tests/baselines/` and regenerate the table file;
   update the README hot-shape rows from the same data; commit together.

## 8. Reading the tables

Verified measurement effects that change how a row is read. The tuning
rulebook itself (what counts as a real improvement, the do-not-retry list) is
kept outside the repository with the maintainer's notes; this section carries
only what a reader of the tables needs.

- **A kernel-cell tile ranking does not transfer to the layer cell.** In the
  isolated grouped-GEMM cell the kernel runs alone on a warm L2. In the layer
  cell (`layer/`) the same kernel runs under programmatic dependent launch
  between the quantize, gate_up and routing kernels, and the tile that wins
  alone can lose there: on 2026-09-04 (RTX 5090, Family C `moe.down`) the
  kernel cell preferred the TileM=16 instances by 6–13 %, while the layer cell
  gained 3.7 % at M = 512 and lost 3.0 % at M = 1024 with the same rule. A
  grouped tile or cascade change is therefore accepted on the layer cell only,
  and the kernel table records the consequence. `FSO_FORCE_TILE` together with
  `FSO_FORCE_TILE_K=<K>` forces one projection's tile inside the captured layer
  for such sweeps.
- **Small-M graph-replay cells are L2-resident.** Replay re-runs one cell's
  kernels with the same weights and the bench does not flush L2, so at
  M ≤ 128 the active weights (10–50 MB) stay in the RTX 5090's 96 MB, the
  H200's 60 MB or the B300's 126.5 MiB L2 and the µs undercut the DRAM floor. A `weight GB/s` above the
  device's DRAM bandwidth means the cell is L2-fed. These rows bound the
  kernel, not the serving cost, which rotates every layer's weights through
  DRAM.
- **Small-M graph µs on the RTX 5090 quantise in ~2 µs steps** (12.4 / 14.4 /
  16.5 …). One cell moving by one step between runs is not a signal. The
  warm-vs-warm noise floor on both devices is about 0.1 % on the median and
  up to ±5 % on isolated cells, which is why acceptance is judged on every
  affected cell against the ±1 % band and a single outlier is re-run, not
  accepted or rejected on its own.
- **The B300 rows are unlocked-clock numbers, with three consequences.** That
  pod cannot set a clock lock, so `gemm/sm100.md` and `layer/sm100.md` rely on
  the `FSO_BENCH_WARM_MS=300` warm-up and on labelling instead (section 5).
  First, cells at M ≥ 4096 swing by about ±3 % rather than ±1 %: the MLP
  block's BF16 cell at M = 4096 was measured nine times on identical code on
  2026-09-15 and spanned 555.1 to 586.0 µs. Second, sustained replay of the
  longest cells droops the clock — the M = 8192 MLP graph falls from 2032 MHz
  to 1822–1980 MHz over twelve blocks of 50 replays while board power rises
  from 228 W to 538 W, while the same test at M = 1024 holds 2032 MHz
  throughout. Third, every dense reading below about 17 µs lands on a ~2.05 µs
  grid (4.16 / 6.20 / 8.24 / 10.29 …), so a one-step flip reads as a 20–25 %
  change and is not one. A B300 comparison that is not larger than these
  effects is not a result.
- **The B300's graph-replay median comes in whole ticks of about 2.05 µs.**
  That quantisation is not an approximation. On 2026-09-17 a least-squares fit
  over the eighteen distinct replay medians of one dense run put every one of
  them on an integer multiple of 2.0604 µs, with a worst residual of 0.135 µs
  (0.65 %). Two rules for reading a small B300 cell follow. A kernel that gets
  faster by less than one tick does not move its cell at all, so an unchanged
  reading is not evidence that a change did nothing. And when a cell does move,
  it moves by a whole tick, which on a 6–10 µs cell is a 20–25 % step; a single
  cell moving one tick is therefore timer granularity rather than a result, and
  what counts as a signal is a run of consecutive M cells of the same shape
  moving together. This is a B300 property: the RTX 5090 has its own, coarser
  step (the bullet above), and the H200 rows are not on a visible grid.
- **The M = 8192 cells of the B300 MLP block scatter by 4–8 %, including the
  pure-cuBLAS column.** Twelve independent worker processes measured that one
  cell on 2026-09-17, alternating between two library builds. The fso BSFP8
  column spanned 587.6 to 613.8 µs and the two builds' medians differed by
  0.1 %; the cuBLAS `scaled_mm` comparator column measured in the same passes,
  which contains no fso kernel and cannot move with a library change, spanned
  586.4 to 607.5 µs. The M = 8192 row of `layer/sm100.md` is therefore a
  published number and not an acceptance unit: a difference read off it is
  inside the cell's own noise unless it is larger than about 8 %.
- **The NVRTC build the process binds is part of the sm_90 kernel.** The sm_90
  block-FP8 kernels are compiled at first call by whichever `libnvrtc.so.13`
  torch already loaded (13.0.88 bundled with the cu130 wheel, 13.2.78 from the
  system CUDA in the `.dev` venv). The 13.0 build read 2–10 % slower on the
  memory-bound cells with an identical BF16 column, so it looks exactly like
  a code regression. Every sm_90 environment block records the NVRTC version,
  and rows measured under different builds are not compared.
- **The clock lock does not hold under the power cap.** During the 16384³
  cubic cell the H200 fell to 1260 MHz at its 700 W cap and the RTX 5090 to
  1815 MHz at 575 W with the lock in force. Cells whose sustained load reaches
  the cap are power-bound and not comparable across units or days; the cubic
  sweep is unpublished for this reason. The family tables (per-call ≤ 1.2 ms)
  showed no dip.
- **sm_120 block-FP8 accuracy before 2026-09-05 was a weight-scale bug, not a
  kernel property.** `quantize_128x128_fp8` produced plain `amax/448` FP32
  scales while the sm_120 repack keeps only the exponent byte, so every weight
  block was dequantised 0.5–1.0× too small and the tables showed cosine
  0.987–0.996 against the BF16 truth. The quantizer now produces power-of-two
  scales by default on sm_120, the pre-pack ops refuse any other scale, and the
  cosine is 0.9992–0.9993. Speed was never affected (same kernel, same bytes).
  The unit tests compare against the dequantised inputs as well as the BF16
  truth for this reason: the BF16 gate alone cannot separate quantization
  error from a kernel or scale bug.
- **A carried-over table is not a baseline.** A row copied from an older
  document without a jsonl in `tests/baselines/` is not comparable to a fresh
  run. Every table in this tree is generated from a baseline file (section 7).
