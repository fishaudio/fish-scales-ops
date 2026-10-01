# fish-scales-ops documentation map

This file defines **where each kind of content lives** and what is forbidden
where. It is the structural contract behind the docs; content is filled in
per file. Status (2026-09-15): structure frozen and every tracked document
migrated. The staging area `doc_review/` was moved out of the repository on
2026-09-15 to `../fso-doc_review-backup-20260915/` (a sibling directory of the
repository, not under version control); it keeps the old README, the old
`docs/api/*.md` and `docs/perf.md`, the memory and casebook copies, and the
`INDEX.md` manifest, and remains the source for the open items listed under
Status below. The gitignored design records went back to `docs/design/`
unchanged.

## Layout

```
README.md                  front page: what fso is, hardware/dtype matrix, install, usage,
                           ONE perf section (hot shapes, sm_90 + sm_120 only), status, layout
docs/
  README.md                this file — placement rules
  guide.md                 integration guide for a serving engine: what to call, build, dense and
                           MoE serving, capture/compile, deployment variables, migrating call sites
  api/
    dense.md               fish_scales_ops.dense, the stable dense linear (no perf numbers)
    moe.md                 fish_scales_ops.moe, the stable MoE layer (no perf numbers)
    attention.md           frozen op contracts for fish_scales_ops.attention (no perf numbers)
    compat.md              fish_scales_ops.compat: the explicit dense ops and the MoE per-step
                           pieces, plus the reference sections every GEMM and MoE op shares
                           (scale layouts, constraints, the generated torch.ops schemas, CUDA
                           graphs, environment variables) (no perf numbers)
  perf/
    README.md              methodology, shape families (the ONLY definition), M grids, FLOPs
                           conventions, dtype-per-arch matrix, clock policy, file map, regeneration
    gemm/
      sm90.md              H200        — Family A / B / C per-GEMM tables (dense ops, grouped kernels)
      sm120.md             RTX 5090    — Family A / B / C per-GEMM tables (RTX PRO 6000 optional section)
      sm100.md             B300 sm_103 — Family A / B / C tables (unlocked clocks)
    layer/
      README.md            what each family's MLP / MoE block contains, model-ms convention
      sm90.md              H200        — whole-block tables (A dense MLP, B routed MoE, C routed + shared)
      sm120.md             RTX 5090    — whole-block tables
      sm100.md             B300 sm_103 — whole-block tables (unlocked clocks)
    attention/
      sm90.md              n/a — fso has no native sm_90 attention kernel (SDPA fallback)
      sm120.md             RTX 5090    — prefill / paged decode / paged prefill tables
      sm100.md             n/a — no native sm_100 attention kernel
  design/                  (gitignored) design records, tuning logs, negative results
    archive/               closed experiments kept as do-not-retry evidence
```

## Placement rules

1. **Performance numbers live in `docs/perf/` and nowhere else.** The one
   exception is the README hot-shape table, which is a *copy* of rows from
   `docs/perf/` taken at the same commit. `docs/api/*` never carries numbers.
   `docs/design/*` may quote numbers only together with the artifact path
   (jsonl / log outside the git tree) they came from; they are not reference values.
2. **README talks about sm_90 and sm_120 only**, gives a *few* hot shapes with
   absolute µs and TFLOPS, and **never compares against another library**
   (no cuBLAS / torch SDPA / FlashInfer / triton columns, no "beats X" prose).
   Comparisons drift with the other library's version and cannot be tracked.
   The one place comparisons are allowed is `docs/perf/layer/` (end-to-end
   block numbers), under three conditions: the comparator ran on the same
   GPU, the same day, under the same graph-replay protocol; its library
   versions are recorded in that file's environment block; and untuned or
   fallback configurations of the comparator are stated next to the column.
   `gemm/` tables carry no comparison columns.
3. **`docs/perf/` is split by domain first (gemm, layer, attention), then by
   SM version** (sm90, sm120, sm100). One file per (domain, SM). `gemm/` holds
   one GEMM per table; `layer/` holds the whole MLP / MoE block of a family as
   one graph-replay number (what a serving loop pays per layer). Inside a
   file the sections are the three fixed shape families in fixed order:
   Family A Qwen3-4B dense, Family B Qwen3-30B-A3B, Family C Qwen3.5-35B-A3B.
   Family definitions, M grids and FLOPs conventions are defined once in
   `docs/perf/README.md` (block contents in `docs/perf/layer/README.md`);
   other files reference, never redefine. A number lives in exactly one
   file: block-level rows are in `layer/`, never repeated in `gemm/`.
4. **Tables are generated, not hand-edited.** Every table row must be
   reproducible from a committed baseline jsonl under `tests/baselines/` plus
   the protocol in `docs/perf/README.md`. A number without a baseline row is
   not a reference number.
5. **Every arch is stated explicitly.** Where fso has no kernel for an SM the
   perf file says "n/a" with the reason; it is never silently omitted.
6. **The optimization rulebook (harness) is not in the repository.** What
   counts as a real improvement, the accuracy gates and the do-not-retry list
   are kept outside the repo with the maintainer's notes (decided 2026-09-15).
   In-repo docs carry only the normative protocol (`docs/perf/README.md`) and
   the caveats a reader of the tables needs (its section 8). Design records in
   `docs/design/` may cite the rulebook by name; they do not restate it.
7. **Design records stay gitignored** (`docs/design/`) until a deliberate
   decision changes that. Do not describe a `docs/design/` file as "committed".

## Status

Every GEMM and layer table in `docs/perf/`, and the two hot-shape tables of the
README, is rendered from `tests/baselines/` by
`bench/gemm/python/render_perf_docs.py`, whose `--check` exits non-zero when a
table has drifted from its baseline. Each perf file names the runs its tables
come from in its provenance section. The attention tables
(`docs/perf/attention/sm120.md`) are a skeleton with no baseline. Open items:
the sm_90 grouped GEMM kernel-level rows (not measured), the RTX PRO 6000
optional section of `gemm/sm120.md` (keep or drop), the attention tables, and
the `docs/design/` consolidation in the last row of the table below.

## Migration record

Each document's content was migrated in September 2026 from sources staged in
`doc_review/`, now archived at `../fso-doc_review-backup-20260915/` (see its
`INDEX.md`). The status column records the migration; for the numbers, the
provenance section of each perf file is the current record. Per target file:

| target | source(s) to cherry-pick from | status |
|---|---|---|
| `README.md` | `doc_review/repo/README.md` (strip comparisons, strip non-sm90/sm120 perf) | **done 2026-09-13**: hot tables rendered from `tests/baselines/`; Install / Usage / Status / Layout / Tests / Knobs / License / Acknowledgments migrated and updated to the current op surface (grouped MoE on sm_90 and sm_120, sm_100 block-FP8, UE8M0 defaults, dropped stale claims) |
| `docs/perf/README.md` | old `perf.md` Methodology + Environment; workspace CLAUDE.md harness invariants; casebook case-01 | protocol, families, M grid, clock policy, dtype matrix, regeneration filled; M grid = bench default |
| `docs/perf/gemm/sm90.md` | old `perf.md` H200 sections + grouped MoE section; `/data/bench-runs/moe_h200_20260903_full/` | filled 2026-09-03; every table rendered from `tests/baselines/`, provenance at the top of the file; grouped kernel-level rows **not measured** |
| `docs/perf/gemm/sm120.md` | old `perf.md` 5090 + RTX PRO 6000 sections; `moe_m1_sm120_grouped.md` measured tables (only rows with artifacts) | filled 2026-09-04; every table rendered from `tests/baselines/`, provenance at the top of the file, which also records the shared-memory regression found and fixed on 2026-09-03/04, grouped cascade rule v5 and the 2026-09-05 block-FP8 weight-scale fix; PRO 6000 not migrated |
| `docs/perf/gemm/sm100.md` | measured fresh on the B300 (`/data/bench-runs/b300_final2_20260917/`); the old `perf.md` B300 rows were not carried over (no clock lock, no preserved jsonl) | filled 2026-09-15: Families A / B / C dense and both grouped kernel tables, rendered from `gemm_sm100_qwen3_4b.jsonl`, `gemm_sm100_qwen3_30a3_dense.jsonl`, `gemm_sm100_qwen3_35a3_dense.jsonl`, `perf_moe_qwen3_30a3_b300.jsonl` and `perf_moe_qwen3_35a3_b300.jsonl`, provenance at the top of the file; the rows are natural-clock (`unlocked`) numbers and carry the caveats of `docs/perf/README.md` §8 |
| `docs/perf/layer/sm90.md` | Family A MLP-forward and Family B/C MoE-layer baselines (moved out of `gemm/sm90.md`), new Family C routed + shared-expert block run, same-day sglang triton / deep_gemm comparators | filled 2026-09-04; every table rendered from `tests/baselines/`, provenance at the top of the file |
| `docs/perf/layer/sm120.md` | same for RTX 5090 (cuBLAS `scaled_mm` comparators for A from the MLP jsonl, sglang triton comparators for B / C) | filled 2026-09-04; every table rendered from `tests/baselines/`, provenance at the top of the file |
| `docs/perf/layer/sm100.md` | same for the B300 (cuBLAS `scaled_mm` comparators for A from `gemm_sm100_qwen3_4b_mlp_fwd.jsonl`; torch `scaled_grouped_mm` MXFP8 and `_grouped_mm` BF16 for B / C from `ref_moe_qwen3_30a3_b300.jsonl`, `ref_moe_qwen3_35a3_b300.jsonl` and `ref_moe_qwen3_35a3_shared_b300.jsonl`, since neither sglang nor deep_gemm is installed on that pod) | filled 2026-09-15; every table rendered from `gemm_sm100_qwen3_4b_mlp_fwd.jsonl`, `perf_moe_qwen3_30a3_b300.jsonl`, `perf_moe_qwen3_35a3_b300.jsonl` and `perf_moe_qwen3_35a3_shared_b300.jsonl` with their comparator files, provenance at the top of the file; Family C carries four comparator columns (routed and routed + shared) because both were measured |
| `docs/perf/attention/sm120.md` | old `perf.md` attention section | skeleton: no baseline in `tests/baselines/` and no generator |
| `docs/api/compat.md` (named `docs/api/gemm.md` until 2026-10-01), `docs/api/attention.md` | old api docs + grouped MoE ops added 2026-09 | **done 2026-09-13**: rewritten from the bindings and wrappers (both MoE layouts, per-arch scale layouts, constraints, graph contract, every env knob; attention: three kernels, plan/run APIs, current channel-scale contracts); the `torch.ops` schema lists of `compat.md` and `attention.md` are generated by `scripts/gen_op_schemas.py` from the `m.def` registrations and the Python `custom_op` registrations, every registered op, between `BEGIN GENERATED` / `END GENERATED` markers, and are regenerated (`--write`, checked with `--check`) rather than edited by hand. On 2026-10-01, for the 0.2.0 interface, `gemm.md` became `compat.md`, took over the MoE per-step section of `moe.md`, and `dense.md` was added for `fso.dense` |
| harness (kernel-optimization rulebook) | workspace CLAUDE.md harness invariants; global CLAUDE.md GPU benching; casebook SKILL.md; charters' known traps; perf_review Reproduce | **not in the repo** (stone, 2026-09-15): all eight sections were filled on 2026-09-13 as `docs/harness.md`, then moved out of the repo to the maintainer's casebook skill. The measurement caveats the tables cite were folded into `docs/perf/README.md` §8 and the acceptance rule into its §7; no in-repo file points at the harness |
| `docs/design/` | the seven design docs + memory engineering logs (H200 MoE) | restored to `docs/design/` unchanged on 2026-09-15 (gitignored); consolidation still open: the two MoE charters carry stale status and merge with the memory logs, `moe_m1_sm120_grouped.md` splits into contract vs tuning log with every number traced to an artifact, `attn_v16_rewrite.md` goes to `docs/design/archive/` with a summary, `perf_review_2026-06.md` is archived |
