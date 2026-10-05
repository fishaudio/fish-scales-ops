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

Status: structure frozen; every sm_90, sm_120 and sm_103 GEMM and layer table
is rendered from `tests/baselines/` (section 7) under the cold-weight protocol
of section 1, and each file names the runs its tables come from in its
provenance section. Every published table measures the fish-scales-ops 0.2.0
release code. The release perf run of 2026-10-01
(`/data/bench-runs/fso_release_perf_20261001/`, the release build, commit
`633f5ca`) produced the H200 dense tables and every RTX 5090 and B300 table;
the H200 MoE tables come from a run of the same code earlier that day, which
the release run re-measured and matched (`layer/sm90.md`). The release run
replaced the B300 tables of 2026-09-29 (PCI `06:00.0`) wholesale with rows from
PCI `66:00.0` rather than merging them, because it ran on a different card, and
it moved the RTX 5090 tables from card C1 to the cards at PCI `21:00.0` and
`22:00.0` (section 5). The B300 rows are natural-clock numbers and labelled as
such. The attention tables have no baseline yet
(`attention/sm120.md`). The old single `perf.md` is archived outside the
repository (`../fso-doc_review-backup-20260915/repo/docs/perf.md`). Rows
marked `TBD` have no accepted baseline yet.

---

## 1. Timing protocol

The rules below are normative. Section 5 gives the clock policy, section 7
the acceptance rule and the regeneration steps, and section 8 the caveats a
reader of the tables needs.

- **One process per cell.** Each `(shape, M)` cell runs in its own subprocess
  so env knobs and JIT/autotune caches cannot leak between cells.
- **CUDA-graph replay median, cold weights.** Per cell: eager warmup →
  side-stream warmup → `torch.cuda.graph` capture → replay batches timed with
  CUDA events; report the median. Since 2026-09-28 the captured graph holds
  R copies of the cell over R copies of its weights (same values, distinct
  memory, every quantized form built per copy), with R × active weight bytes
  at least twice the device's L2 (R ∈ [2, 256]; the cap was 16 until
  2026-09-29, which left the 1–2 MB shared-expert projections warm on every
  device and `wo` / `wqkv` at 1–1.9 × L2 on the B300), and the reported `µs` is the
  replay time / R: each copy's weights have been evicted by the others before
  the graph returns to it, as a serving step evicts every layer's weights by
  streaming the next layers' through. Activations, routing and the
  intermediate slabs stay warm, as they do in a model. Rows carry
  `weight_copies`, and every row in `tests/baselines/` now does. Files
  measured before 2026-09-28 replayed one warm copy and overstated the
  M ≤ 128 rows; none of them is published any more. The reported `µs` is the
  graph-replay cost (what a production decode/prefill loop pays), never eager
  single-call time; the one exception is the two torch `_grouped_mm`
  comparator columns of `layer/sm120.md`, which cannot be captured on that
  card, are timed eagerly and say so in their titles.
- **Clock policy** (section 5). The RTX 5090 runs under a clock lock; the
  datacenter cards (H200, B300) run at their natural clock with a clock
  sampler beside the run, and their environment blocks say so.
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

## 5. Devices and clocks

Since 2026-09-29 (stone): only gaming cards get a clock lock. The RTX 5090 is
locked at its no-boost frequency so that its runs are comparable across days
and commits, because a gaming card's boost behaviour depends on temperature,
power and the neighbouring cards' load. Datacenter cards (H200 and B300)
run at their natural clock, as they do in serving, with a
clock sampler recording what the card actually did; their published numbers
are natural-clock numbers, and the power cap is part of the device. CUDA 13
toolchain everywhere.

| SM | device | SMs | clock policy | what the sampler reads under load | notes |
|---|---|---|---|---|---|
| sm_90 | NVIDIA H200 | 132 | **natural clock, no lock** (`nvidia-smi -rgc` before a run; until 2026-09-29 the tables were taken under `-lgc 1980`) | 1980 MHz in every decode cell; the 700 W power cap holds the dense prefill cells lower, to 1440–1620 MHz at M = 8192 in the release run behind the published dense tables (`gemm/sm90.md`) | |
| sm_120 | NVIDIA GeForce RTX 5090 | 170 | `nvidia-smi -lgc 2407` | a median of 2392–2400 MHz; the SW power-cap flag appears on prefill cells although the 500 ms sampler reads only 110–330 W of 575 W there, and how far it pulls the clock depends on the card: in the release run card `21:00.0` held 2325–2400 MHz in compute-heavy cells, while card `22:00.0` and the earlier card C1 dropped to 2085–2272 MHz | primary sm_120 device; the release tables come from two cards, `21:00.0` (every dense table, the MLP block and the Family B MoE tables) and `22:00.0` (the Family C MoE tables, routed and routed + shared), so absolute prefill numbers are not comparable between the two groups or with the earlier C1 tables, while the ratios inside a table are (`gemm/sm120.md`); release the lock (`-rgc`) when the run ends |
| sm_120 | NVIDIA RTX PRO 6000 Blackwell | 188 | `nvidia-smi -lgc 2430` (a gaming-class board) | 2400 MHz | optional section |
| sm_103 | NVIDIA B300 | 148 | natural clock (the pod cannot lock; it never could) | 2032 MHz on the MoE tables apart from a few samples; under the 1100 W power cap 34 to 42 of the 134 to 140 busy samples of each Family A dense pass read below 2000 MHz in the release run, down to 1050 MHz (`gemm/sm100.md`) | needs `FSO_BENCH_WARM_MS=300` — the dense bench, the MLP-layer forward bench (`bench_qwen3_4b_mlp_forward.py`, knob added 2026-09-15) and the MoE layer bench all honour it |

Before every run: on the 5090 verify the lock took effect (`clocks.sm` reads
the locked value under load); on a datacenter card verify no lock is left
from an earlier session (`clocks.applications.graphics` / `-rgc`); run a
sampler (`nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,clocks_event_reasons.active
-lms 1000`) next to the chain and keep its csv in the run directory; record
driver, CUDA, torch and the fso commit in the file's environment block.

Environments of record (the 0.2.0 release tables; each file's environment
block has the details):
- **H200:** the apex-0520 venv (torch 2.13.0+cu130 with cuBLAS 13.1.1.3,
  triton 3.7.1, sglang 0.5.20, deep_gemm 0.2.0); the sm_90 kernels are
  JIT-compiled by the NVRTC 13.0.88 of the torch wheel (section 8).
- **RTX 5090:** the `.venv-sgl` venv on gpu5090-03 (Python 3.12.3, torch
  2.13.0+cu130, triton 3.7.1, vLLM 0.29.0, FlashInfer 0.6.18) for the fso rows
  and most comparators, `.venv-sglang` (sglang 0.5.20) for the sglang columns
  and `.venv-fi` (cuDNN 9.26) for the cuDNN column.
- **B300:** the env0520 venv (torch 2.13.0+cu130, nvidia-cutlass-dsl 4.6.2,
  cuBLAS 13.1.1.3, NVRTC 13.0.88, triton 3.7.1), with an overlay directory on
  `PYTHONPATH` that adds only `expecttest` 0.3.0 for the files that use torch's
  reference MXFP8 quantize.

Power cap. On the H200 and the B300 the cold protocol (section 1) streams
the weights from HBM for R times longer per replay than the warm one did,
and the heaviest dense cells run into the board's power limit. On the H200,
in the release run behind the published dense tables (natural clock, 200 ms
sampler), 70 of the 206 dense worker windows carried the SW power-cap flag at
the 700 W limit: none at M ≤ 256, most of the windows at M = 512 to 2048,
and every window at M = 4096 and 8192, where the clock fell to 1440–1620 MHz. The MoE
tables, measured earlier the same day (`layer/sm90.md`), ran at a median of
1980 MHz with only short capped bursts at the heavy prefill cells. On the
B300, in the release run, 34 to 42 of the 134 to 140 busy samples of each
Family A dense pass read below 2000 MHz, down to 1050 MHz, and the dense
prefill cells differ by up to 21 % between two passes of the same run. Those
dense rows are power-bound numbers: sample the clock on both arms before
accepting or rejecting a change on them, and prefer the MoE tables or the
decode band for a verdict (`gemm/sm90.md`, `gemm/sm100.md`, Environment).

## 6. dtype matrix per SM

| SM | BF16 | BSFP8 (1×128 act / 128×128 wgt) | MXFP8 (1×32) | notes |
|---|---|---|---|---|
| sm_90 | reference column | ✓ deep_gemm JIT (dense + grouped) | — | K % 128 |
| sm_120 | reference column | ✓ CUTLASS block-scaled (dense; K % 128, UE8M0 activation and weight scales) | ✓ dense + grouped | Family B/C MoE rows are MXFP8; block-FP8 required K % 512 and had an FP32-weight-scale bug until 2026-09-05 (section 8) |
| sm_103 | reference column | ✓ since 2026-09-05: 1×128 scales expanded ×4 onto the MXFP8 tcgen05 tiers (same kernels and bytes as MXFP8; K % 128, N % 128); tables published 2026-09-15, re-measured 2026-09-17, 2026-09-22 (twice) and 2026-09-23, re-measured under the cold-weight protocol on 2026-09-29, and re-measured by the 0.2.0 release run on 2026-10-01 (card `66:00.0`) | ✓ dense + grouped since 2026-09-15 | grouped MoE (M3) landed 2026-09-15: CUTLASS pointer-array block-scaled kernel on the masked slab layout, eight kernels in the captured layer; cascade v2 and the programmatic dependent launch on the prep kernel and the grouped GEMM since 2026-09-17, and the dense path gained the wave-tile rule with its 64- and 192-wide N tiles on the same day; since 2026-09-22 the captured layer is five kernels in the decode band, where a slot-bound grouped route indexed by the routing kernel's packed active-expert list needs no argument-preparation launch and carries its own fused SwiGLU epilogue, and since 2026-09-23 five kernels at every M, because the pointer-array route reads its per-group problem shapes from the same routing kernel instead of an argument-preparation launch and the slot route's row-capacity clause is per kernel (the plain slot kernel to its full 64-wide tile, the fused-SwiGLU one to one 32-column chunk); the dense path additionally gained a vendored CuTe-DSL decode row, for M ≤ 32 on 2026-09-22 and extended to M ≤ 64 on 2026-09-23, which needs `nvidia-cutlass-dsl` 4.5.0 (the `sm100` extra) and is inert below it |

## 7. Regeneration

Tables are generated, never hand-edited. `bench/gemm/python/render_perf_docs.py`
rewrites every GEMM and layer table in `gemm/sm90.md`, `gemm/sm120.md`,
`gemm/sm100.md`, `layer/sm90.md`, `layer/sm120.md`, `layer/sm100.md` and the
two README hot tables from
`tests/baselines/*.jsonl`; `--check` exits non-zero if any table has drifted
from the baselines (run it before committing a baseline change). Prose around
the tables (provenance, readings) is edited by hand in the same commit.

1. Set the clock as section 5 says (lock the RTX 5090; on a datacenter card
   release any lock and start the clock sampler) and verify it.
2. Run the bench for the family/device; the jsonl lands outside the git tree
   (`/data/bench-runs/<run>/` on the H200 box, `/mnt/share/stone-bench-runs/<run>/`
   on the 5090 host).
3. Accept only if every affected cell is faster or within ±1% of the committed
   baseline. A MoE / MLP tile or cascade change is judged on the layer cell
   (section 8), and the kernel table records the consequence.
4. Copy the accepted jsonl to `tests/baselines/` and regenerate the table file;
   update the README hot-shape rows from the same data; commit together.

## 7b. Regenerating baselines, tables and reports from a run directory

Every perf run leaves its raw jsonl files in a run directory outside the git
tree; the path from there to the published numbers is one tool,
`bench/gemm/python/perf_report.py`, and never a hand-written script:
`merge --run <dir>` builds the canonical baseline files (the fso rows as they
are, the layer-level comparators merged into `ref_moe_*`, the kernel-level ones
into `ref_kern_moe_*`, the dense files copied; cells without a µs are listed
with their reason), `diff --a baselines --b <dir>` prints every shared cell's
delta with decode (M ≤ 128) and prefill (M > 128) medians and the cells beyond
a threshold — the A/B of any change — `report [--run <dir>]` prints the
comparison tables with ×fso and band geometric means, and `install --run <dir>`
takes a run directory that `bench/run_perf.py` wrote (section 7c): it needs the
run's `manifest.json`, takes the device from it, and does merge into
`<dir>/merged` → diff → copy into `tests/baselines/` → the run's provenance
into `tests/baselines/provenance/<device>.json` → `render_perf_docs.py`
(→ `--check`). It refuses a smoke run, a run that did not complete, a file
whose step exited non-zero or that no step wrote, and a drifted run unless
`--accept-drift` is given. The docstring carries the raw-file manifest. `report` and
`diff` never read a merged file that is lying in a run directory: a directory
that was installed once keeps its `ref_moe_*` / `ref_kern_moe_*` from that
install next to any raw files added later, so both commands rebuild the merged
files from the raw ones in a scratch copy first (`merged_view`) — otherwise one
run's fso rows get paired with an earlier run's comparators (found on
2026-09-28: a `down` M = 1 comparator cell read from the stale merge was the
half-warm number of the earlier protocol). A comparator file whose name is
not in the manifest is never merged. `merge` (and so `install`) also checks
every row that carries `weight_copies` against the cold protocol's target and
prints the rows whose copies × weight bytes fall short of 2 × the device's L2
(`COLD-PROTOCOL: …`); such rows were served from L2 for part of their replay
and must not be published as cold (the 16-copy cap of 2026-09-28 produced
exactly those rows for the 1–2 MB shared-expert projections and for the Family
C `down` kernel cell at M = 1).

## 7c. Reproducing the tables

The tables of record are measured with one command per machine,
`bench/run_perf.py`, under an environment lock that the run refuses to deviate
from. Each machine's lock is `bench/env/<machine>.lock.json`, where the machine
is `h200`, `5090` or `b300` (the device suffixes of the baseline files). It
names the interpreter of every environment and the exact versions of the
packages the tables depend on, the extra environment variables and `PYTHONPATH`
overlays, the driver, the card, the clock and compute-mode policy, the GPU lock
file, the bench environment, and on sm_90 the compiler `jit_compiler_sm90()`
must report. `bench/perf_suite.py` lists every step of every table per machine:
the bench, its arguments, the output file name and the table group.

    python3 bench/run_perf.py --machine h200 --out /data/bench-runs/<run> --fso-path <dir>
    python3 bench/run_perf.py --machine 5090 --out /mnt/share/stone-bench-runs/<run> --fso-path <dir>
    python3 bench/run_perf.py --machine b300 --out ~/bench-runs/<run> --fso-path <dir>

`--fso-path` is the directory that holds the `fish_scales_ops` package to
measure; it goes first on `PYTHONPATH`. `--tables` selects table groups
(`dense`, `moe`, `moe_ref`, `moe_kern`, `shared`; all by default). `--dry-run`
prints the plan and the preflight result without touching the card.
`--smoke` runs every step on M = 1 and 64 only, and such a run can never be
installed.

**What a refusal means.** Before the run takes the card, the preflight asks
every interpreter, in a subprocess that sees no GPU, for its package versions
and for the `fish_scales_ops` it imports, and asks `nvidia-smi` for the driver
and the card. Any difference from the lock is drift: a package or Python
version, the driver, or the sm_90 compiler. The run refuses drift with exit
code 3 and prints every difference. `--allow-drift` proceeds instead and
records the differences in the manifest. A blocker refuses the run in every
case: another host, a missing interpreter or overlay, a `fish_scales_ops` that
does not import or does not come from `--fso-path`, the wrong card, or a card
with a compute process on it. The run takes the machine's GPU lock file with
`flock` and never queues behind another process on the card.

**What a run leaves.** `<run>/manifest.json` records the machine, the time
window, the lock and its sha256, the versions found, the drift list, the
fish-scales-ops version, `build_info()`, extension sha256 and sm_90 compiler,
the card, the clock policy and the clocks a 200 ms sampler observed during
every step, every step's command, window, rows and failures (`logs/`), and the
commit of the tree or test kit that holds the benches. After the steps the run
merges its raw files into `<run>/merged` and diffs them against the installed
baselines; it installs nothing.

**How install uses the manifest.** `perf_report.py install --run <run>`
(section 7b) accepts only a run directory with a manifest and takes the device
from it. It refuses a smoke run, a run that did not complete, a file whose
step exited non-zero or that no step of the manifest wrote, and a drifted run
unless `--accept-drift` is given. For every file it installs, it records the
run's provenance in `tests/baselines/provenance/<device>.json` and keeps the
entries of the files it does not install. `render_perf_docs.py` renders those
files into the block below, and its `--check` covers the block.
`tests/bench/test_run_perf_plan.py` checks the locks, the suite and the plan
without a GPU.

<!-- BEGIN GENERATED: environments of record (bench/gemm/python/render_perf_docs.py) -->
### Environments of record

Generated by `bench/gemm/python/render_perf_docs.py` from `tests/baselines/provenance/<device>.json`, which `perf_report.py install` writes from the `manifest.json` of the run it installs; do not edit it by hand. One row per installed baseline file: the run that produced it, its time window, the card, driver and clocks, the package versions of the environments its steps ran in, and the fish-scales-ops build.

#### H200 (sm_90)

| baseline file                           | run                                        | window (UTC) | card | driver | clocks | environments | fish-scales-ops | sm_90 JIT compiler |
|-----------------------------------------|--------------------------------------------|--------------|------|--------|--------|--------------|-----------------|--------------------|
| `gemm_sm90_qwen3_30a3_dense.jsonl`      | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `gemm_sm90_qwen3_35a3_dense.jsonl`      | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `gemm_sm90_qwen3_4b.jsonl`              | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `gemm_sm90_qwen3_4b_mlp_fwd.jsonl`      | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `perf_moe_qwen3_30a3_h200.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `perf_moe_qwen3_35a3_h200.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `perf_moe_qwen3_35a3_shared_h200.jsonl` | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `ref_moe_qwen3_30a3_h200.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |
| `ref_moe_qwen3_35a3_h200.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               | —                  |

#### RTX 5090 (sm_120)

| baseline file                           | run                                        | window (UTC) | card | driver | clocks | environments | fish-scales-ops |
|-----------------------------------------|--------------------------------------------|--------------|------|--------|--------|--------------|-----------------|
| `gemm_sm120_qwen3_30a3_dense.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm120_qwen3_35a3_dense.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm120_qwen3_4b.jsonl`             | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm120_qwen3_4b_mlp_fwd.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_30a3_5090.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_35a3_5090.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_35a3_shared_5090.jsonl` | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_kern_moe_qwen3_30a3_5090.jsonl`    | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_kern_moe_qwen3_35a3_5090.jsonl`    | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_moe_qwen3_30a3_5090.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_moe_qwen3_35a3_5090.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |

#### B300 (sm_103; the dense files are tagged sm100)

| baseline file                           | run                                        | window (UTC) | card | driver | clocks | environments | fish-scales-ops |
|-----------------------------------------|--------------------------------------------|--------------|------|--------|--------|--------------|-----------------|
| `gemm_sm100_qwen3_30a3_dense.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm100_qwen3_35a3_dense.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm100_qwen3_4b.jsonl`             | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `gemm_sm100_qwen3_4b_mlp_fwd.jsonl`     | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_30a3_b300.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_35a3_b300.jsonl`        | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `perf_moe_qwen3_35a3_shared_b300.jsonl` | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_moe_qwen3_30a3_b300.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_moe_qwen3_35a3_b300.jsonl`         | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |
| `ref_moe_qwen3_35a3_shared_b300.jsonl`  | installed before the harness (no manifest) | —            | —    | —      | —      | —            | —               |

<!-- END GENERATED: environments of record -->

## 8. Reading the tables

Verified measurement effects that change how a row is read. The tuning
rulebook itself (what counts as a real improvement, the do-not-retry list) is
kept outside the repository with the maintainer's notes; this section carries
only what a reader of the tables needs.

- **A kernel-cell tile ranking does not transfer to the layer cell.** In the
  isolated grouped-GEMM cell the kernel runs alone, with its weights rotated
  like every other cell's (section 1) and its activations warm. In the layer
  cell (`layer/`) the same kernel runs under programmatic dependent launch
  between the quantize, gate_up and routing kernels, and the tile that wins
  alone can lose there: on 2026-09-04 (RTX 5090, Family C `moe.down`) the
  kernel cell preferred the TileM=16 instances by 6–13 %, while the layer cell
  gained 3.7 % at M = 512 and lost 3.0 % at M = 1024 with the same rule. A
  grouped tile or cascade change is therefore accepted on the layer cell only,
  and the kernel table records the consequence. `FSO_FORCE_TILE` together with
  `FSO_FORCE_TILE_K=<K>` forces one projection's tile inside the captured layer
  for such sweeps.
- **Warm-L2 small-M cells overstate the kernel; the protocol rotates weights
  since 2026-09-28.** Replaying one cell's graph with the same weights keeps
  the M ≤ 128 active set (10–50 MB) in the RTX 5090's 96 MB, the H200's 60 MB
  or the B300's 126.5 MiB L2, and the µs undercut the DRAM floor — a
  `weight GB/s` above the device's DRAM bandwidth is the tell. A serving step
  streams every layer's weights through in turn, so no layer finds its
  weights in L2; the bench now rotates ≥ 2 × L2 of weight copies inside the
  captured graph (section 1). Rows without `weight_copies` were taken warm and
  their small-M values bound the kernel, not the serving cost; no published
  row lacks it now. When the protocol changed on 2026-09-28, the RTX 5090's
  Family B M = 1 layer read 22.7 µs warm and 35.2 µs cold.
- **A cell whose active weight footprint lands within a few MB of the L2 is
  sensitive to the routing draw, not just to the code.** Family C M = 4 on the
  5090 holds 30 active experts, 94 MB of weights against a 96 MB L2: re-drawing
  the routing with six seeds moved one A/B's verdict from −9.4 % to +21.4 % and
  the absolute time from 39 to 59 µs, with the active count barely changing
  (30–32). The affected 5090 cells are Family C M = 4–5 and Family B M = 3–4.
  Average over draws before calling such a cell a regression (2026-09-28).
- **Small-M graph µs on the RTX 5090 quantise in ~2 µs steps** (12.4 / 14.4 /
  16.5 …). One cell moving by one step between runs is not a signal. Measured
  again 2026-09-28 at M = 1: 24 points of a width sweep landed on six distinct
  values spaced 2.05 µs apart, so an M = 1 layer cell cannot resolve better
  than 9 % of itself. The
  noise floor measured between repeated runs of the earlier single-copy
  protocol, on both devices, was about 0.1 % on the median and up to ±5 % on
  isolated cells, which is why acceptance is judged on every
  affected cell against the ±1 % band and a single outlier is re-run, not
  accepted or rejected on its own.
- **The B300 rows are unlocked-clock numbers, with three consequences.** That
  pod cannot set a clock lock, and under section 5 a datacenter card runs at
  its natural clock in any case, so `gemm/sm100.md` and `layer/sm100.md` rely
  on the `FSO_BENCH_WARM_MS=300` warm-up, the clock sampler and labelling.
  First, cells at M ≥ 4096 swing by about ±3 % rather than ±1 %: the MLP
  block's BF16 cell at M = 4096 was measured nine times on identical code on
  2026-09-15 and spanned 555.1 to 586.0 µs. Second, sustained replay of the
  longest cells droops the clock — the M = 8192 MLP graph falls from 2032 MHz
  to 1822–1980 MHz over twelve blocks of 50 replays while board power rises
  from 228 W to 538 W, while the same test at M = 1024 holds 2032 MHz
  throughout. Third, under the single-copy protocol every dense reading below
  about 17 µs landed on a ~2.05 µs grid (4.16 / 6.20 / 8.24 / 10.29 …), so a
  one-step flip read as a 20–25 % change and was not one; the cold tables
  (since 2026-09-29) divide each replay by its R weight copies (section 1), so their
  small rows are not on that grid (next bullet). A B300 comparison that is not
  larger than these effects is not a result.
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
  step (the bullet above), and the H200 rows are not on a visible grid. These
  rules describe single-copy replays, which every B300 table before 2026-09-29
  used; a cold-protocol cell is the replay time divided by its R weight copies,
  which spreads the tick over the copies (`gemm/sm100.md`, Cold weights).
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
  system CUDA in the `.dev` venv). On 2026-10-01 the release kernels were
  compiled both ways with everything else the same
  (`/data/bench-runs/sm90_nvrtc_ab_20261001/`, `REPORT.md`). NVRTC 13.0 emits
  3.6–14.7 % more instructions per kernel than 13.2 at the same register
  count, mostly uniform-datapath address and index arithmetic that 13.2 folds
  away, 44–90 % of it inside loop bodies. On the MoE layer that costs at most
  0.84 % in any cell (band medians 0.04–0.48 %); on the dense FP8 GEMMs it
  costs a median 3.1 % at decode (0.9–7.3 %) and 2.0 % at prefill, and up to
  12.3 % on the small N = 2048, K = 4096 projections at M = 256. NVRTC does
  not compile the BF16 column, so across two NVRTC builds the BF16 column stays
  put while the FP8 columns slow down, which looks exactly like a code
  regression. `FSO_JIT_USE_NVCC=1` with a 13.2 nvcc reproduces the 13.2 code:
  its SASS is identical to the NVRTC 13.2 build apart from the offset of the
  kernel-parameter block. Every sm_90 environment block records the NVRTC
  version, and rows measured under different builds are not compared. All
  published sm_90 tables are NVRTC 13.0.88 builds in the apex-0520 venv
  (2026-10-01); the 2026-09-29 tables they replaced were 13.2.78 builds.
- **The clock lock does not hold under the power cap.** During the 16384³
  cubic cell the H200 (then still measured under a lock) fell to 1260 MHz at
  its 700 W cap and the RTX 5090 to 1815 MHz at 575 W with the lock in force.
  Cells whose sustained load reaches the cap are power-bound and not
  comparable across units or days; the cubic sweep is unpublished for this
  reason. Under the single-copy protocol the family tables (per-call
  ≤ 1.2 ms) showed no dip; under the cold protocol the heaviest dense family
  cells reach the cap on the H200 and the B300 (section 5, power cap), and on
  the RTX 5090 the SW power-cap flag appears on the prefill cells despite the
  lock, how far it pulls the clock depending on the card (section 5).
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
  run. Every GEMM and layer table in this tree is generated from a baseline
  file (section 7); the attention tables (`attention/sm120.md`) are skeletons
  with no baseline and hold no numbers.
