/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SM100/SM103 MXFP8 (1×32, SFVecSize=32) dispatch and kernel launcher.
//
// Mirrors the sm_120 dispatcher's host-overhead discipline:
//   * E9-equivalent: a thread-local cache of *initialized*
//     GemmUniversalAdapter instances keyed by (ptrs, shape). On a hit the
//     call is just adapter.run(stream) — no cuTensorMapEncodeTiled, no
//     cudaFuncSetAttribute (the adapter does both only in initialize()).
//   * E8-equivalent: cudaFuncSetAttribute happens inside
//     Gemm::initialize(), i.e. only on cache miss. CUDA-graph capture is
//     therefore safe as long as callers warm up eagerly first — the same
//     contract test_cuda_graph.py enforces for sm_120.
//   * Workspace (issue #12, charter 7.5): the caller provides it. The op
//     bodies ask gemm_workspace_sm100_mxfp8 for the bytes the cascade's
//     choice needs (the FP32 partials of the two-kernel split-K; zero for
//     the plain tiles with the default scheduler), allocate them with the
//     torch caching allocator on the stream they launch on, and pass them
//     down (Sm100DenseWorkspace, dense_common.cuh), so no two streams ever
//     share scratch and a capture takes graph-private memory. Only the
//     sweep-only Stream-K force codes (ST 7, 8) still use the per-thread
//     Sm100WorkspacePool below.
//
// Cascade status: bring-up heuristic only (C1). The (M, tiles_n) rules
// below are placeholders pending the b300 tile sweep (FSO_FORCE_TILE grid
// + cudagraph-µs bench per docs/perf/README.md methodology). Retune before
// trusting any cell.
//
// Env overrides: FSO_FORCE_TILE=TM,TN,ST reuses the sm_120 wire format so
// the shared sweep harness drives this path unchanged. For sm_100 the
// third field selects the (SM-count, TileK, cluster, scheduler) variant
// instead of the stage count (stages are StageCountAutoCarveout-derived):
//   ST=1 → 1SM cluster(1,1) K128          ST=2 → 2SM cluster(2,1) K128
//   ST=3 → 1SM cluster(1,1) K256          ST=4 → 2SM cluster(2,1) K256
//   ST=5 → 2SM cluster(2,2) K128          ST=6 → 2SM cluster(2,2) K256
//   ST=7 → 1SM cluster(1,1) K128 StreamK  ST=8 → 1SM cluster(1,1) K256 StreamK
//   ST=9 → 1SM parallel split-K (two-kernel; splits from FSO_FORCE_KSPLIT,
//          else auto pick_splits) — (128,128) only
//   ST=10 → 1SM cluster(2,2) K128 (TMA multicast ×2 on both A and B — the
//           nvjet mid-band shape; ncu F3)      ST=11 → 2SM cluster(2,2) K256
//
// Narrow / 192-wide N tiles (added 2026-09-17). Codes 1..11 above bake a
// particular epilogue choice into each row, which is fine for the tiles the
// cascade already ships but useless for sweeping a new tile, where the
// epilogue is one of the things being decided. Codes 20..25 therefore name
// the (SM-count, TileK, cluster) variant AND the epilogue explicitly, and
// accept any TileN the kernel supports (64, 128, 192, 256):
//   ST=20 → 1SM cluster(1,1) K128, NoSmem (direct-store) epilogue
//   ST=21 → 1SM cluster(1,1) K128, TMA (smem-staged) epilogue
//   ST=22 → 1SM cluster(1,1) K256, NoSmem epilogue
//   ST=23 → 1SM cluster(1,1) K256, TMA epilogue
//   ST=24 → 2SM cluster(2,1) K128, NoSmem epilogue
//   ST=25 → 2SM cluster(2,1) K128, TMA epilogue
// Instantiated TileN per code (the set the 2026-09-17 sweep needed; adding
// one costs a full CUTLASS kernel's compile time, so the list is explicit):
//   ST=20/21: TileN 64, 128, 192, 256     ST=22/23: TileN 64
//   ST=24: TileN 64, 192, 256             ST=25: TileN 64, 128, 192, 256
// (TileN 256 on 20 / 21 / 24 and 128 / 256 on 25 were added 2026-10-28,
// issue #12: they reach tiles the cascade already ships — (128,256) direct-
// store and TMA, (256,256) direct-store and TMA, (256,128) TMA — so that every
// configuration the cascade's rules pick can be forced under its own name.)
//
// Dense epilogue (bias, erf-GELU, residual; issue #12, common/epilogue.cuh):
// served on every configuration the cascade's default rules pick (the list in
// dense_common.cuh) and on the two-kernel split-K (applied once, in the
// reduce), whether the cascade or FSO_FORCE_TILE chose it. A forced row whose
// configuration has no dense-epilogue instantiation refuses a non-identity
// epilogue with fso::DenseEpilogueNotImplemented naming the code, before
// anything is launched: the Stream-K rows 7 and 8 (in-kernel partial-tile
// reduction) and the sweep-only tiles of rows 3, 4 (TileN 256), 5, 6, 10, 11,
// 20 / 24 (TileN 64, 192), 22 and 23. Identity calls are unchanged on every row.
//
// Quantized output (fso.dense.linear_quantized, issue #12 S3b; epilogue.out_q /
// out_scale): the same configurations, through their quantized-output
// instantiation (launch_sm100_mxfp8_gemm_epi_q: bias and the activation, then
// RN_bf16, then the activation quantizer of fso/activation_contract.h, the
// codes as D and the UE8M0 words in the sm_100 contract layout), and the
// two-kernel split-K through its quantized reduce. It takes no residual; the
// dispatch refuses out_q with a residual, with a scale layout other than
// kDenseOutScaleMxSm100, with N % 128 != 0, and on the rows that refuse the
// bf16 epilogue, before anything is launched.
//
// FSO_PRINT_TILE_INFO=1 makes every instantiation print, once, the mainloop
// stage count StageCountAutoCarveout derived for it and its shared-memory
// footprint — the quantity that decides whether a narrower TileN bought
// pipeline depth or only CTAs.
//
// FSO_FORCE_TILE_K=<K> restricts the override to GEMMs with that K, so a
// layer-level A/B can move one projection's tile while the others keep their
// cascade picks (same contract as the sm_120 dispatchers).
//
// Unknown combos fall through to the cascade (same behavior as sm_120).

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "blockscale_gemm/common/epilogue.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/gemm_types.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/dense_common.cuh" // host knobs, workspace, dense-epilogue launchers
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh" // FSO_FORCE_TILE parsing (arch-agnostic)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace sm100_blockscaled_gemm
{

namespace detail
{

// read_force_swizzle, read_force_raster, read_print_tile_info, kDenseGdcCompiled, dense_gemm_pdl and
// apply_default_scheduler_knobs live in dense_common.cuh (shared with the dense-epilogue translation units).

// Multiprocessor count of the current device, read once. The rule below is
// written against the count the driver reports rather than the 148 of a whole
// B300, so a MIG slice gets its own answer.
inline int sm_count() noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    return s_sms;
}

// CTAs a (cluster row count, CTAs per cluster, TileN) choice launches for an
// (M, N) problem. A 1-SM tile covers 128 rows with one CTA; a 2-SM tile covers
// 256 rows with a two-CTA cluster.
inline int grid_ctas(int M, int N, int cluster_rows, int ctas_per_cluster, int tile_n) noexcept
{
    return ((M + cluster_rows - 1) / cluster_rows) * ((N + tile_n - 1) / tile_n) * ctas_per_cluster;
}

// N-tile chosen by wave arithmetic. Returns TileM * 1000 + TileN, or 0 when
// this (M, N) is not a cell the rule owns.
//
// Why the CTA count is the quantity that decides: the block-scaled mainloop
// stages 200-230 KB of shared memory per CTA (printed per instantiation by
// FSO_PRINT_TILE_INFO), more than half of the SM's shared memory, so exactly
// one CTA is resident per SM. The CTA count is therefore the number of SMs
// that do any work at all, and in the decode band -- where the problem is one
// or two 128-row M-tiles -- it is decided by the N tile alone. Halving TileN
// doubles the CTA count and costs about 1.47x in per-SM work rate (ncu fit,
// 2026-09-17), a net 1.36x, but only while SMs are idle: past one wave the
// extra CTAs queue behind the first and the narrower tile is pure loss. So:
// take the tile that puts the most CTAs on the machine without spilling into a
// second wave.
//
// Four bounds, each from a measured counter-example rather than from the model:
//   * M >= 64. Below it the A-operand tile is almost entirely out-of-bounds
//     fill and every extra CTA repeats that fill, so the same kernel gets
//     slower the more CTAs it launches (gate_up M = 1, isolated GEMM: 14.46 us
//     on 76 CTAs, 14.85 on 102, 24.63 on 304), and at M = 32 the wave-picked
//     tile is 25 % slower than tier 1 on wqkv and 18 % on gate_up.
//   * At least half the SMs must end up busy. Every measured win puts 80 to
//     144 CTAs on the machine; the very narrow Family C projections
//     (N = 1024, N = 2048) would get a 32- or 64-CTA grid, outside the range
//     this rule was calibrated on.
//   * TileN = 64 only while one 256-row cluster row covers the problem
//     (M <= 256). Above a single 128-row M-tile (128 < M <= 256) the 1-SM and
//     the 2-SM 64-wide tiles launch the same number of CTAs, and the equal-CTA
//     tie-break below always takes the 2-SM 256x64 form there; on the tables
//     this adds `wo` and `down` (N = 2560) at 128 < M <= 256. The two forms
//     differ in how often they stream the weight: the 1-SM tile makes one pass
//     per 128-row M-tile, and at M = 256 it tied with tier 1 on the isolated
//     GEMM (`wo` 8.27 against 8.26, `down` 14.39 against 14.40) and was 3.5 %
//     SLOWER on the Family A MLP layer cell, where `down` runs straight after
//     the SwiGLU quantize (2026-09-17, when this bound was M <= 128). The 2-SM
//     tile makes one pass, and with cold weights it beats tier 1, whose kernel
//     puts 40 CTAs on these shapes, at every M from 144 to 256: alone `wo` by
//     4 to 5 % and `down` by 5 to 6 % (MXFP8 and BSFP8 alike), and the Family A
//     MXFP8 MLP block by 1.5 to 2.7 % (run fso_0.2.1_20261010/P1b, e4_gemm64 and
//     e1_band2). Above M = 256 a 64-wide tile needs two cluster rows or three
//     128-row tiles, which this rule has no measurement for, and at M = 512
//     with N = 2048 it doubles the CTA count to 128 and is still 10 % slower
//     than tier 1, so the rate penalty of the narrow atom is not always repaid.
//     The 192-wide tile has no such counter-example up to M = 1024.
//   * The 2-SM 192-wide form is not shipped. The only cells where it is the
//     CTA-maximising pick are Family C `gdn.in_proj` (N = 12288) at M = 64 and
//     M = 128, and there it measured 1.3x SLOWER than tier 1 even though it
//     puts 128 CTAs on the machine against cuBLAS's 96; every 192-wide win in
//     the sweep came from the 1-SM form.
//
// Ties (the same CTA count from a 1-SM and a 2-SM tile) go to the 2-SM tile.
// At an equal CTA count the two forms keep the same number of SMs busy, but
// the 2-SM form covers 256 rows per tile instead of 128, so the kernel makes
// half as many passes over the B operand -- the weight matrix. Timed on their
// own the two forms agree to within the replay tick (`wo` and `down` at
// M = 1024: 12.40 / 22.57 us for 1-SM against 12.37 / 22.55 for 2-SM), because
// an isolated dense cell replays one GEMM 150 times and its whole working set
// (10 MB of activations plus a 25 MB weight at `down` M = 1024) stays resident
// in this part's 132 MB L2, which makes the extra passes free. In a fused
// layer they are not free: the Family A MLP block runs `down` straight after
// `gate_up`, whose 50 MB weight and 40 MB BF16 output have just swept L2, so
// `down` starts with a cold weight and every extra pass over it is HBM
// traffic. Measured at M = 1024, N = 2560 with nsys inside the captured block
// (per-launch median): the 1-SM 192-wide tile costs 21.38 us on its own but
// 25.44 us in the block, while the 2-SM 192-wide tile costs 20.90 us on its
// own and 21.10 us in the block (run b300_head_vs_wt_20260921/R2).
//
// Empty-pair guard (issue #12, T1b). Below M = 129 the 2-SM 64-wide pick
// leaves the second CTA of every pair with A rows 128-255, all of them out
// of bounds. That CTA still loads its A box, and ncu counts the box among
// the bytes the crossbar delivers to the SMs (fused q/k/v 3840 x 1280 at
// M = 78: 25.8 MB of TMA loads on the 2-SM tile against 15.4 MB on the 1-SM
// tile). When the activation is also smaller than 128 KiB (M * K < 131072
// bytes of fp8), the 2-SM form runs in a slow mode. On 3840 x 1280 at
// M = 65-102 it takes 5.05-5.98 us for the identity call and 5.27-6.25 us
// for the bias call, while the 1-SM 128 x 64 tile takes 4.13-4.33 and
// 4.06-4.35 us. From M = 103, where the activation reaches 128 KiB, the 2-SM
// tile takes 3.93-3.96 and 4.14-4.16 us. The edge follows M * K and nothing
// else: at K = 1536 the slow mode ends at M = 86, at K = 1024 at M = 128,
// and K = 2048 or 2560 never shows it at M >= 65. Placing the activation at
// 18 base offsets (M = 78 and 117) did not move the edge on this shape, the
// values of x did not matter, and the slow mode stays with PDL off and in an
// isolated ncu launch (8.35-8.45 us at M = 78 against 7.36-7.62 us at
// M = 103). With fewer clusters the slow mode appears only at some
// activation addresses: on 2048 x 1280 (64 CTAs) the identity call was slow
// at 9 and 7 of the 38 M values below 103 in the two passes, six of them in
// both, and the same M turned fast or slow when only the activation's base
// offset changed. When both CTAs of the pair hold real rows (128 < M <= 256)
// a small activation does not slow the 2-SM form (K = 512 and 768: it stays
// 2.5-4.5 % faster than the 1-SM tile on the identity call). Why 128 KiB
// is the edge was not established. So below M = 129, with an activation
// under 128 KiB and K >= 1024, the rule hands the cell the 1-SM 128 x 64
// tile of the same width: it still owns the cell, with half the CTAs. On the
// wave-owned short-K shapes N = 2432-4736 at K = 1024-1536 and every M
// measured below the edge, the 1-SM tile was faster in both passes for the
// identity, bias, bias + residual and bf16-x bias calls (0.64-0.99 of the
// 2-SM time, the same output bytes). Above the edge the 2-SM tile keeps the
// identity call 2-6 % faster, so the guard stops there. Below K = 1024 the
// guard is not applied: at K = 512 on N = 2432 and 2560 (76 and 80 CTAs) and
// at K = 768 on N = 2432 at M = 64 / 65 the identity call was up to 3 %
// slower on the 1-SM tile in at least one pass, although the bias,
// bias + residual and bf16-x bias calls of the same cells were 1-27 % faster
// on it. Measured on the B300 with cold weights (run
// fso_issue12_20261028/runs/T1b, sweep/, ab/ and ncu/).
inline constexpr long long kWaveEmptyPairMinActBytes = 128 * 1024;
inline constexpr int kWaveEmptyPairMinK = 1024;

inline int pick_wave_tile(int M, int N, int K) noexcept
{
    if (M < 64 || M > 1024)
        return 0;
    int const sms = sm_count();
    int const cand_n[4] = {256, 192, 128, 64};
    int best_m = 0, best_n = 0, best_ctas = 0;
    for (int i = 0; i < 4; ++i)
    {
        int const ctas = grid_ctas(M, N, 128, 1, cand_n[i]);
        if (ctas <= sms && ctas > best_ctas) { best_ctas = ctas; best_m = 128; best_n = cand_n[i]; }
    }
    for (int i = 0; i < 4; ++i)
    {
        int const ctas = grid_ctas(M, N, 256, 2, cand_n[i]);
        if (ctas <= sms && ctas > best_ctas) { best_ctas = ctas; best_m = 256; best_n = cand_n[i]; }
    }
    if (best_n != 64 && best_n != 192)
        return 0;                 // the pick is a tile the cascade already had
    if (2 * best_ctas < sms)
        return 0;                 // narrowing would still leave the machine half idle
    if (best_n == 64 && M > 256)
        return 0;                 // 64-wide only while one 256-row cluster row covers M
    if (best_m == 256 && best_n == 192)
        return 0;                 // measured counter-example at gdn.in_proj M = 64 / 128
    // Equal-CTA tie-break, applied after the guards above so that each of their
    // rejections keeps its exact meaning: when the 2-SM form of the SAME N
    // width launches the same number of CTAs, take it, for the halved number of
    // passes over the weight matrix explained in the comment above. The test is
    // an equality on CTA counts, so it can only fire above M = 128 (below it a
    // 2-SM tile always doubles the count) and it never changes which cells the
    // rule owns -- only which tile an already-owned cell gets. A 64-wide cell
    // with 128 < M <= 256 always ties here (one 256-row cluster row against two
    // 128-row tiles), so it always leaves with the 2-SM 256x64 tile, never with
    // the 1-SM 128x64 tile that lost at M = 256.
    if (best_m == 128 && grid_ctas(M, N, 256, 2, best_n) == best_ctas)
        best_m = 256;
    // Empty-pair guard (see the comment above): below M = 129 with an activation under 128 KiB and K >= 1024 the
    // 2-SM 64-wide pick becomes the 1-SM 128 x 64 tile. It changes only the tile of a cell the rule already owns.
    if (best_m == 256 && best_n == 64 && M <= 128 && K >= kWaveEmptyPairMinK
        && static_cast<long long>(M) * K < kWaveEmptyPairMinActBytes)
        best_m = 128;
    return best_m * 1000 + best_n;
}

// Short-K tile rule (issue #12, T1): the K bound and the M range of pick_short_k_tile. Which of its cells tier 1
// (cuBLAS scaled_mm) declines, so that their bias calls reach this rule too, is decided in Python
// (`_sm100_smm.short_k_owns`: N <= 2048, 1024 <= K <= kShortKMaxK, 64 < M <= 256), whose K bound must stay equal.
inline constexpr int kShortKMaxK = 2560;
inline constexpr int kShortKMinM = 65;
inline constexpr int kShortKMaxM = 2048;

// N-tile and SM count of a short-K GEMM (K <= kShortKMaxK, 65 <= M <= kShortKMaxM), chosen by wave arithmetic over
// the TileK 128 tiles with a TMA epilogue. Returns TileM * 1000 + TileN, or 0 when no tile of the rule fits one wave
// (the cell then falls through to pick_wave_tile and the band rules below, unchanged). `fused` is true when the call
// carries an activation or writes the quantized output (linear_quantized).
//
// Why short K needs its own rule. At K = 1280 a CTA's mainloop is 10 TileK-128 blocks, and the 128 x 64 and 2-SM
// 256 x 64 tiles stage 8 and 9 of them (FSO_PRINT_TILE_INFO), so the whole K is in flight after the prologue: a CTA's
// time is one pipeline fill, ten MMA blocks and one epilogue, and since each CTA owns one tile its epilogue is fully
// exposed. Two things therefore set the kernel time: how many SMs share the output, and how many outputs each CTA's
// epilogue processes (the erf-GELU epilogue costs about 0.8 us over the bias epilogue on 64-wide tiles, 1.6 us on
// 128-wide and 3.3 us on 256-wide, at any M while the grid is one wave). The band rules below were measured on long-K
// shapes and send these cells to the 2-SM 256 x 128 tile with TileK 256, whose 4 stages of 5 blocks start the first
// MMA only after a 256-deep stage lands, and which puts at most 32 CTAs on 148 SMs at N <= 2048 and M <= 256 (half of
// them on out-of-bounds rows at M <= 128). Measured on the B300
// with cold weights (run fso_issue12_20261028/runs/T1: every tile of the dense-epilogue lists forced, both entry
// paths, two passes, and fine-M probes):
//   * 64-wide while the 1-SM 128 x 64 grid fits one wave. 1280 x 1280 at M = 65-768 and 2048 x 1280 at M = 65-364:
//     bias + erf-GELU 4.6-4.8 us, bias + residual 4.1-4.35 us and linear_quantized(bias + erf-GELU) 4.9-5.1 us on the
//     1-SM 128 x 64 tile, against 6.7-6.8, 5.9-6.0 and 7.4-7.55 us on the 256 x 128 TileK 256 tile. A call with an
//     activation or the quantized output takes the 1-SM tile always (the 2-SM 256 x 64 form was 1-23 % slower at
//     every M measured). Any other call keeps pick_wave_tile's 64-wide pick where that rule owns the cell, and
//     elsewhere takes the 2-SM 256 x 64 form only while both CTAs of its one cluster row hold real rows
//     (128 < M <= 256): there it halves each CTA's share of the weight (identity 3.75-3.82 against 3.94-4.05 us on the
//     1-SM tile, bias 1-2 % faster; 1280 x 1280, 2048 x 1280 and 2048 x 2048 at ten M from 129 to 250). At M <= 128
//     the second CTA of each pair holds only out-of-bounds rows and the 2-SM form is erratic on narrow N (2048 x 1280:
//     4.6 us at M = 65 / 66 and 4.5 at M = 80 against 3.8 at M = 72 / 78 / 96; the 1-SM tile 3.9-4.0 us at all of
//     them), so the rule does not use it there (pick_wave_tile's empty-pair guard explains the erratic cells: an
//     activation under 128 KiB, and the same guard covers the cells the wave rule owns); above M = 256 its grid adds
//     an empty CTA row whenever ceil(M / 128) is odd.
//   * otherwise 128-wide while a TileK 128 tile fits one wave: the 2-SM 256 x 128 form when its grid fits (it streams
//     the weight once per 256 rows), else the 1-SM 128 x 128. 5120 x 1280 at M = 144-256 with bias + erf-GELU: 2-SM
//     256 x 128 TileK 128 (80 CTAs) 6.6 against 7.2 us on the TileK 256 form of the same tile; at M = 364 the 1-SM
//     128 x 128 (120 CTAs, one wave) 7.0 against 10.4 us on the TileK 256 tile, whose 160 CTAs run two waves;
//     1280 x 1280 at M = 1079-1792 and 2048 x 1280 at M = 533-768: 2-SM 256 x 128 TileK 128 6.4-6.65 against
//     6.85-7.1 us; 2048 x 1280 at M = 1079: 1-SM 128 x 128 (144 CTAs) 6.9 against 10.15 us; 1024 x 2048 (Family C
//     shared.gate_up) at M = 1280-2048: 2-SM 256 x 128 TileK 128 7.4 against 7.85-7.9 us, and bias + residual 6.3-6.5
//     against 7.3-7.4 us. kShortKMaxM caps the rule at that last measured row (on 1024 x 2048 the 2-SM tile would
//     still fit one wave up to M = 2304, unmeasured; 1280 x 1280 already falls through at M = 1920).
//   * otherwise 0. Where only a 192-wide tile fits one wave (5120 x 1280 at M = 533, the Family A / B / C wide-N
//     projections at M = 256-1024) pick_wave_tile's pick was the fastest measured. Where nothing fits one wave the
//     TileK 128 alternatives on 5120 x 1280 (M = 768-2048) ranged from 6 % faster to 2 % slower than the band rules'
//     TileK 256 tile, cell by cell and epilogue by epilogue, and on 2048 x 1280 the 64-wide tiles in two waves were
//     10 % faster at M = 1280 / 1536 but 7 % slower at M = 2048: that multi-wave regime keeps the band rules.
inline int pick_short_k_tile(int M, int N, int K, bool fused) noexcept
{
    if (K > kShortKMaxK || M < kShortKMinM || M > kShortKMaxM)
        return 0;
    int const sms = sm_count();
    if (grid_ctas(M, N, 128, 1, 64) <= sms)
    {
        if (!fused)
        {
            int const wave = pick_wave_tile(M, N, K); // a 64-wide pick wherever it owns the cell (see the comment above)
            if (wave != 0)
                return wave;
            if (M > 128 && M <= 256 && grid_ctas(M, N, 256, 2, 64) <= sms)
                return 256 * 1000 + 64;
        }
        return 128 * 1000 + 64;
    }
    if (grid_ctas(M, N, 256, 2, 128) <= sms)
        return 256 * 1000 + 128;
    if (grid_ctas(M, N, 128, 1, 128) <= sms)
        return 128 * 1000 + 128;
    return 0;
}

// Thread-local device workspace of the forced Stream-K variants (FSO_FORCE_TILE
// codes 7 and 8, sweep-only), which keep their reduction barriers here. Until
// 2026-10-28 it also held the FP32 partial sums of the two-kernel parallel
// split-K; those now come from the caller (Sm100DenseWorkspace, the torch
// caching allocator on the caller's stream: issue #12, charter 7.5), because a
// per-thread buffer is shared by every stream and every graph that thread
// launches. The plain tiles with the default scheduler ask for nothing.
//
// A captured CUDA graph keeps the buffer address it saw at capture time, so the
// pool never frees a buffer. When an eager call needs more than the pool holds,
// the current buffer is retired -- kept allocated for the life of the process,
// because a graph captured earlier may still read and write it -- and a new one
// of at least twice the size is allocated, so retirements stay few. A growth is
// refused while the launch stream is capturing, where cudaMalloc is not allowed:
// ensure() then returns cudaErrorStreamCaptureUnsupported, which the op turns
// into a RuntimeError (sm100_mxfp8_error_hint in mxfp8_sm100_kernel.cu). One
// eager call of the same shape on the capturing thread sizes the pool for it.
struct Sm100WorkspacePool
{
    void* ptr = nullptr;
    std::size_t bytes = 0;
    std::vector<void*> retired; // never freed: graphs captured earlier may hold them

    static Sm100WorkspacePool& instance()
    {
        static thread_local Sm100WorkspacePool p;
        return p;
    }

    // On success *out is a buffer of at least `needed` bytes (nullptr when
    // `needed` is 0) that stays valid for the life of the process.
    cudaError_t ensure(std::size_t needed, cudaStream_t stream, void** out)
    {
        *out = nullptr;
        if (needed == 0)
            return cudaSuccess;
        if (bytes >= needed)
        {
            *out = ptr;
            return cudaSuccess;
        }
        // Ask about the stream the launch is on: torch.cuda.graph captures on
        // a side stream, which a query of the legacy stream does not see.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return cudaErrorStreamCaptureUnsupported;
        }
        if (cap != cudaStreamCaptureStatusNone)
            return cudaErrorStreamCaptureUnsupported;
        constexpr std::size_t kGranule = std::size_t(1) << 20;
        std::size_t grow = std::max(needed, 2 * bytes);
        grow = (grow + kGranule - 1) / kGranule * kGranule;
        void* fresh = nullptr;
        if (cudaMalloc(&fresh, grow) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return cudaErrorMemoryAllocation;
        }
        if (ptr)
            retired.push_back(ptr);
        ptr = fresh;
        bytes = grow;
        if (read_print_tile_info())
            std::fprintf(stderr,
                "[fso sm100 workspace] grew to %zu bytes for a %zu-byte request; %zu earlier buffer(s) "
                "retired and kept allocated\n",
                bytes, needed, retired.size());
        *out = ptr;
        return cudaSuccess;
    }
};

} // namespace detail

// Launch one (TileM, TileN, ClusterM, ClusterN) instantiation. Returns
// cudaSuccess on success; kernel runtime errors surface on the next sync
// (post-launch cudaGetLastError check intentionally omitted — see the E8
// note in the sm_120 launcher).
//
// Capture-safety split (mirror of the sm_120 E8/E9 discipline):
//   * cudaFuncSetAttribute → per-instantiation static guard, fires on the
//     FIRST call only. Callers must warm up eagerly before capture.
//   * Params (TMA descriptors) → thread-local cache. A cache MISS during
//     capture is safe: to_underlying_arguments is host-only work, and
//     GemmUniversalAdapter's *static* run(params, stream) issues nothing
//     but the (cluster-)launch itself.
//
// Workspace: whatever CUTLASS asks for comes from `workspace` (caller-provided,
// see Sm100DenseWorkspace; zero bytes with the default scheduler), except for
// the Stream-K variants, which keep the per-thread Sm100WorkspacePool. With
// `workspace_query` non-null nothing is launched and *workspace_query receives
// the bytes of `workspace` the call needs.
template <int TileM, int TileN, int ClusterM, int ClusterN, int TileK = 128, bool UseStreamK = false,
    bool NoSmemEpi = false>
cudaError_t launch_sm100_mxfp8_gemm(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream, int swizzle,
    Sm100DenseWorkspace const& workspace, std::size_t* workspace_query)
{
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, ClusterM, ClusterN, TileK, UseStreamK, NoSmemEpi>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if (workspace_query == nullptr && detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] %dx%dx%d cluster=(%d,%d) streamk=%d nosmem=%d "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, TileK, ClusterM, ClusterN, static_cast<int>(UseStreamK),
                static_cast<int>(NoSmemEpi), Config::kStages, Config::kSmemBytes,
                Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);
    auto ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(mat_d);

    auto problem = cute::make_shape(M, N, K, 1);
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, 1});
    auto layout_SFA = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem);
    auto layout_SFB = Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem);

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, ptr_D, stride_D}};

    if constexpr (UseStreamK)
    {
        // splits > 1 bypasses the stream-K heuristic in favour of an
        // explicit split-K decomposition (the heuristic mode measured
        // WORSE than no split at all on down-class decode shapes —
        // 30.9 µs vs 17.5 µs at M=1, 2026-07-05 sweep). FSO_FORCE_KSPLIT
        // drives the sweep; the cascade sets nothing yet.
        int const ks = tensorrt_llm::kernels::blockscale_gemm::read_force_tile().ks;
        if (ks > 1)
            args.scheduler.splits = ks;
        // Probe knobs (2026-07-06, both read once into static caches):
        //   FSO_SK_NDET=1   → ReductionMode::Nondeterministic (atomic
        //                     accumulation; the 2026-07-05 losses were all
        //                     under the default Deterministic turnstile
        //                     lock, which serializes the fixup).
        //   FSO_SK_DECOMP   → 1=DataParallel 2=SplitK 3=StreamK (default
        //                     heuristic).
        static int const s_ndet = []() {
            char const* e = std::getenv("FSO_SK_NDET");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        static int const s_decomp = []() {
            char const* e = std::getenv("FSO_SK_DECOMP");
            return (e && *e) ? std::atoi(e) : 0;
        }();
        using cutlass::gemm::kernel::detail::ReductionMode;
        using cutlass::gemm::kernel::detail::DecompositionMode;
        if (s_ndet)
            args.scheduler.reduction_mode = ReductionMode::Nondeterministic;
        if (s_decomp == 1)
            args.scheduler.decomposition_mode = DecompositionMode::DataParallel;
        else if (s_decomp == 2)
            args.scheduler.decomposition_mode = DecompositionMode::SplitK;
        else if (s_decomp == 3)
            args.scheduler.decomposition_mode = DecompositionMode::StreamK;
    }
    else
    {
        // Cascade-provided swizzle; FSO_FORCE_SWIZZLE overrides it and
        // FSO_FORCE_RASTER sets the raster order (probe knobs, dense_common.cuh).
        detail::apply_default_scheduler_knobs(args.scheduler, swizzle);
    }

    // Workspace of the non-Stream-K variants: sized here, provided by the caller. The Stream-K variants use the
    // per-thread pool instead and ask the caller for nothing.
    std::size_t caller_ws_bytes = 0;
    void* caller_ws = nullptr;
    if constexpr (!UseStreamK)
        caller_ws_bytes = Gemm::get_workspace_size(args);
    if (workspace_query != nullptr)
    {
        *workspace_query = caller_ws_bytes;
        return cudaSuccess;
    }
    if constexpr (!UseStreamK)
    {
        if (caller_ws_bytes > workspace.bytes)
            return cudaErrorInvalidValue; // the op body sized `workspace` with the query above
        if (caller_ws_bytes > 0)
            caller_ws = workspace.ptr;
    }

    // E9-style cache: key on everything the initialized Params bake in.
    struct CacheKey
    {
        void const* p[6];
        int dims[4];  // M, N, K, swizzle
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;
    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB; key.p[4] = ptr_D;
    key.p[5] = caller_ws; // baked into the Params when CUTLASS asks for a workspace (never with the default scheduler)
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = swizzle;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
        if constexpr (UseStreamK)
        {
            // Stream-K reduction barriers must be clean before every launch;
            // re-enqueue the (stream-ordered, capture-legal) workspace init
            // even on a Params-cache hit. A captured init simply re-runs on
            // every replay — exactly the semantics the barriers need.
            std::size_t const ws_bytes = Gemm::get_workspace_size(args);
            void* ws = nullptr;
            cudaError_t const ws_err = detail::Sm100WorkspacePool::instance().ensure(ws_bytes, stream, &ws);
            if (ws_err != cudaSuccess)
                return ws_err;
            if (ws_bytes > 0
                && GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
    }
    else
    {
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return cudaErrorInvalidValue;
        std::size_t ws_bytes = caller_ws_bytes;
        void* ws = caller_ws;
        if constexpr (UseStreamK)
        {
            ws_bytes = Gemm::get_workspace_size(args);
            cudaError_t const ws_err = detail::Sm100WorkspacePool::instance().ensure(ws_bytes, stream, &ws);
            if (ws_err != cudaSuccess)
                return ws_err;
        }
        if (ws_bytes > 0)
        {
            // Stream-ordered workspace init (capture-legal; a captured init
            // simply re-runs on every replay, which is the semantics the
            // scheduler workspace wants anyway).
            if (GemmKernel::initialize_workspace(args, ws, stream) != cutlass::Status::kSuccess)
                return cudaErrorUnknown;
        }
        kernel_params = GemmKernel::to_underlying_arguments(args, ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    // E8-equivalent: set max dynamic smem once per instantiation, on the
    // first (eager) call. GemmUniversalAdapter::initialize would do this on
    // every cache miss — including mid-capture — so we do it ourselves.
    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    // Programmatic dependent launch when GDC is compiled in (detail::dense_gemm_pdl): the kernel waits on its
    // parent grid before its first global access, so it may start while the activation quantize in front of
    // it drains, and it releases its own dependent once its MMAs are issued.
    bool const pdl = !UseStreamK && detail::dense_gemm_pdl();
    return Gemm::run(kernel_params, stream, nullptr, pdl) == cutlass::Status::kSuccess ? cudaSuccess
                                                                                     : cudaErrorLaunchFailure;
}

// ---------------------------------------------------------------------------
// Parallel split-K (two-kernel) decode path.
//
// ncu 2026-07-06 (profile/ncu_sm100_2026-07-06/REPORT.md, F1): at down-class
// decode shapes (N=2560 → 20 tile-CTAs on 148 SMs) cuBLAS runs the SAME
// 128×128 tile as a data-parallel split-K×3 (grid 60) plus a trivial
// splitKreduce kernel, and wins ~0.75×. CUTLASS's StreamK scheduler is the
// wrong kind of split-K here (in-kernel semaphore fixup: heuristic 30.9 µs,
// splits=2 18.7 µs vs 16.7 plain). This path reproduces the cuBLAS scheme:
//   1. ONE batched GEMM launch (L = splits) writing FP32 partials to the
//      workspace pool — batch b covers K-slice [b*K/S, (b+1)*K/S). A/B are
//      addressed in-place via batch strides (K/S along K); the SF tensors
//      keep the full-K quantize layout via stride surgery (see below).
//   2. A float4 grid-stride reduce kernel: partials [S, M, N] → BF16 D.
// Both launches are capture-safe (workspace from the pool, eager-warmup
// contract unchanged).
//
// SF layout stride surgery: tile_atom_to_shape_SFA((M,N,K/S,S)) has the
// right SHAPE for a K-slice batch, but its strides describe S independent
// contiguous tensors. In the full-K quantize layout (K-minor blocks per
// 128-row block), the correct strides are:
//   * M-block-row stride = 512 * (K/128)      = default * S
//   * batch (L) stride   = 512 * (K/(128*S))  = the DEFAULT M-block-row stride
// so the fix is: stride(Mrest) *= S; stride(L) = old stride(Mrest).

// Programmatic dependent launch (PDL) for the two-kernel split-K pair.
//
// The reduce kernel below reads the FP32 partials the batched split-K GEMM
// writes, so it must not touch them before that GEMM has finished. Without
// PDL the driver enforces that by serialising the two launches outright: the
// reduce's CTAs are only created after the GEMM's last CTA retires, so the
// reduce's entire launch and prologue latency sits exposed between the two
// kernels. cuBLAS's own split-K pair does not pay that cost -- its reduce is
// observed starting while its GEMM is still running.
//
// With `cudaLaunchAttributeProgrammaticStreamSerialization` set on the reduce
// launch the driver may create and schedule the reduce's CTAs while the GEMM's
// tail is still draining, and `cudaGridDependencySynchronize()` inside the
// kernel then holds every CTA until the GEMM has actually completed. The
// ordering the data dependency requires is therefore unchanged; only the
// launch and the prologue overlap.
//
// The producer is the CUTLASS kernel. Since 2026-10-11 mxfp8_sm100_kernel.cu builds it with
// CUTLASS_ENABLE_GDC_FOR_SM100, so the `cutlass::arch::launch_dependent_grids()` call that sits between the
// CUTLASS mainloop and its epilogue stores releases the reduce before the partials have landed. That is
// safe for the same reason as above: the release only creates the reduce's CTAs, and
// `cudaGridDependencySynchronize()` holds every one of them until the GEMM has completed.
//
// The reduce calls cudaTriggerProgrammaticLaunchCompletion() right after its wait (2026-10-11). It writes the
// caller's D buffer throughout its own execution, but a trigger does not hand D over: a programmatic
// dependent still waits, in griddepcontrol.wait, for this grid to complete and its writes to be visible
// before it touches global memory. The trigger only lets the dependent's CTAs be created early.
//
// FSO_DISABLE_PDL=1 restores the plain serialised launch (the same variable
// the sm_120 MoE chain reads, so one knob disables PDL everywhere).
// sm100_pdl_wait (the grid-dependency wait both reduce kernels start with) is defined in dense_common.cuh.

static __global__ void sm100_mxfp8_splitk_reduce_kernel(
    float4 const* __restrict__ partials, ushort4* __restrict__ out, int splits, long mn4, bool pdl)
{
    sm100_pdl_wait(pdl);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
    // Release the next kernel's launch (2026-10-11). A programmatic dependent waits for this grid's COMPLETION,
    // memory included, before it touches global memory (griddepcontrol.wait), so releasing it here only lets
    // its CTAs be created and run their prologue while this reduce still writes D; it cannot read D early.
    if (pdl && threadIdx.x == 0)
        cudaTriggerProgrammaticLaunchCompletion();
#endif
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < mn4; i += stride)
    {
        float4 acc = partials[i];
        for (int s = 1; s < splits; ++s)
        {
            float4 const p = partials[s * mn4 + i];
            acc.x += p.x; acc.y += p.y; acc.z += p.z; acc.w += p.w;
        }
        ushort4 o;
        o.x = __bfloat16_as_ushort(__float2bfloat16(acc.x));
        o.y = __bfloat16_as_ushort(__float2bfloat16(acc.y));
        o.z = __bfloat16_as_ushort(__float2bfloat16(acc.z));
        o.w = __bfloat16_as_ushort(__float2bfloat16(acc.w));
        out[i] = o;
    }
}

// The reduce that applies the dense epilogue (sm100_mxfp8_splitk_reduce_epi_kernel) is defined in its own translation
// unit, ops/mxfp8_sm100_splitk_epi_kernel.cu, and reached through launch_sm100_mxfp8_splitk_reduce_epi
// (dense_common.cuh). It is deliberately not in this header: this header's device code is the identity kernels'
// cubin, and the driver's load-time finalisation of a kernel depends on that cubin's layout (issue #12, R1).

namespace detail
{

// Largest split count in [2, min(8, sm_count / tiles)] that divides the SF
// K-block count evenly (SF blocks are 128-K-wide and cannot straddle a
// slice). Returns 1 when no useful split exists — including when the tile
// grid already fills the machine, which self-limits the route to decode.
inline int pick_splits(int tiles, int kblks) noexcept
{
    static int s_sms = 0;
    if (s_sms == 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sms, cudaDevAttrMultiProcessorCount, dev);
        if (s_sms <= 0)
            s_sms = 148;
    }
    int const budget = s_sms / (tiles > 0 ? tiles : 1);
    for (int s = budget < 8 ? budget : 8; s >= 2; --s)
        if (kblks % s == 0)
            return s;
    return 1;
}

} // namespace detail

// Batched-slice GEMM into FP32 partials + reduce. Caller guarantees
// K % 128 == 0 and (K/128) % splits == 0 (pick_splits does).
//
// `epilogue` (issue #12): applied once, by the reduce, to the sum of the
// partials; the batched GEMM is the identity-epilogue FP32-partials kernel in
// every case. `workspace` holds the partials (and whatever CUTLASS asks for,
// zero with the default scheduler): it is the caller's, allocated per call with
// the torch caching allocator on `stream` (charter 7.5), so two streams or two
// graphs never share it. With `workspace_query` non-null nothing is launched
// and *workspace_query receives the bytes the call needs.
template <int TileM, int TileN>
cudaError_t launch_sm100_mxfp8_gemm_splitk(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, int splits, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue, Sm100DenseWorkspace const& workspace, std::size_t* workspace_query)
{
    // NoSmemEpi: decode band, partial stores are a sliver (see the M ≤ 32
    // NOSMEM rationale in the cascade). ElementD float → FP32 partials.
    using Config = Sm100MxFP8GemmConfig<TileM, TileN, 1, 1, 128, false, true, float>;
    using Gemm = typename Config::Gemm;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename Gemm::Arguments;
    using Params = typename Gemm::Params;

    if (workspace_query == nullptr && detail::read_print_tile_info())
    {
        static bool s_info_printed = false;
        if (!s_info_printed)
        {
            s_info_printed = true;
            std::fprintf(stderr,
                "[fso sm100 tile] splitk %dx%dx128 cluster=(1,1) fp32-partials "
                "stages=%d smem_bytes=%d epi_smem_bytes=%d\n",
                TileM, TileN, Config::kStages, Config::kSmemBytes, Config::kEpiSmemBytes);
        }
    }

    auto ptr_A = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_a);
    auto ptr_B = reinterpret_cast<cutlass::float_e4m3_t const*>(mat_b);
    auto ptr_SFA = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_a);
    auto ptr_SFB = reinterpret_cast<cutlass::float_ue8m0_t const*>(scales_b);

    int const kslice = K / splits;
    long const mn = static_cast<long>(M) * N;

    // Workspace: FP32 partials [splits, M, N], plus whatever CUTLASS asks
    // for (expected 0 with the default CLC scheduler) appended behind.
    std::size_t const partial_bytes = static_cast<std::size_t>(splits) * mn * sizeof(float);

    auto problem = cute::make_shape(M, N, kslice, splits);
    // In-place K-slices: leading strides stay full-K, batch stride = kslice.
    auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, kslice, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, kslice, 1});
    cute::get<0>(stride_A) = K;
    cute::get<2>(stride_A) = kslice;
    cute::get<0>(stride_B) = K;
    cute::get<2>(stride_B) = kslice;
    // Partials are genuinely batched [S, M, N] — packed stride.
    auto stride_D = cutlass::make_cute_packed_stride(typename GemmKernel::StrideD{}, {M, N, splits});

    // SF layouts: shape from the sliced problem, strides fixed up per the
    // header comment (Mrest *= S; L = old Mrest). In-place mutation of a
    // by-value copy keeps the exact LayoutSFA/LayoutSFB types.
    // LayoutSF{A,B} structure: shape (((32,4),Mrest), ((32,4),Krest), (1,L))
    // stride (((16,4),sMrest), ((0,1),512), (0,sL)) — the scalars to patch
    // are get<0,1> (M-block-row) and get<2,1> (batch).
    auto fix_sf = [splits](auto l) {
        auto const row = cute::get<0, 1>(l.stride());
        cute::get<0, 1>(l.stride()) = row * splits;
        cute::get<2, 1>(l.stride()) = row;
        return l;
    };
    typename Config::LayoutSFA layout_SFA
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem));
    typename Config::LayoutSFB layout_SFB
        = fix_sf(Config::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem));

    // Params cache (E9 discipline). The partials pointer is NOT part of the key:
    // the batched GEMM's direct-store epilogue holds ptr_D as a plain pointer in
    // its Params (no TMA descriptor), so a cached entry is reused with ptr_D
    // patched to this call's workspace. The caller's caching allocator may hand
    // out a different block on every call, which would otherwise rebuild the
    // Params (TMA descriptors of A, B, SFA, SFB) each time.
    struct CacheKey
    {
        void const* p[5];
        int dims[4];  // M, N, K, splits
        bool operator==(CacheKey const& o) const { return std::memcmp(this, &o, sizeof(o)) == 0; }
    };
    struct CacheHash
    {
        std::size_t operator()(CacheKey const& k) const noexcept
        {
            std::size_t h = 1469598103934665603ULL;
            auto const* p = reinterpret_cast<unsigned char const*>(&k);
            for (std::size_t i = 0; i < sizeof(k); ++i) { h ^= p[i]; h *= 1099511628211ULL; }
            return h;
        }
    };
    static thread_local std::unordered_map<CacheKey, Params, CacheHash> s_params_cache;

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, problem,
        {ptr_A, stride_A, ptr_B, stride_B, ptr_SFA, layout_SFA, ptr_SFB, layout_SFB},
        {{1.0f, 0.0f}, nullptr, stride_D, nullptr /* ptr_D patched below */, stride_D}};

    std::size_t const ws_bytes = Gemm::get_workspace_size(args);
    if (workspace_query != nullptr)
    {
        *workspace_query = partial_bytes + ws_bytes;
        return cudaSuccess;
    }
    if (workspace.ptr == nullptr || workspace.bytes < partial_bytes + ws_bytes)
        return cudaErrorInvalidValue; // the op body sized `workspace` with the query above
    float* partials = static_cast<float*>(workspace.ptr);
    void* cutlass_ws = static_cast<char*>(workspace.ptr) + partial_bytes;
    args.epilogue.ptr_D = partials;

    if (detail::read_print_tile_info())
    {
        // The workspace a capture bakes in, so that a test can tell two captured calls' workspaces apart.
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess)
            (void) cudaGetLastError(); // the query's own error only
        else if (cap != cudaStreamCaptureStatusNone)
            std::fprintf(stderr, "[fso sm100 splitk] captured on stream %p: workspace %p (%zu bytes)\n",
                static_cast<void*>(stream), workspace.ptr, partial_bytes + ws_bytes);
    }

    CacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.p[0] = ptr_A; key.p[1] = ptr_B; key.p[2] = ptr_SFA; key.p[3] = ptr_SFB;
    key.p[4] = ws_bytes > 0 ? cutlass_ws : nullptr; // a CUTLASS workspace would be baked into the Params
    key.dims[0] = M; key.dims[1] = N; key.dims[2] = K; key.dims[3] = splits;

    Params kernel_params;
    auto it = s_params_cache.find(key);
    if (it != s_params_cache.end())
    {
        kernel_params = it->second;
        kernel_params.epilogue.ptr_D = partials;
    }
    else
    {
        // NOTE: Gemm::can_implement is intentionally SKIPPED. The mainloop's
        // check literally recomputes tile_atom_to_shape_SFA(problem) and
        // compares strides, so the split-K stride surgery always "fails" it
        // even though to_underlying_arguments consumes args.layout_SFA/SFB
        // verbatim (verified: make_tensor(args.ptr_SFA, args.layout_SFA) →
        // TMA descriptor). The constraints it would enforce hold by
        // construction here: K % 128 == 0, kslice % 128 == 0 (16B TMA
        // alignment on every batch base), N % 4 == 0 for the FP32 partials.
        if (ws_bytes > 0
            && GemmKernel::initialize_workspace(args, cutlass_ws, stream) != cutlass::Status::kSuccess)
            return cudaErrorUnknown;
        kernel_params = GemmKernel::to_underlying_arguments(args, cutlass_ws); // host-only
        if (s_params_cache.size() < 64u)
            s_params_cache.emplace(key, kernel_params);
    }

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaError_t result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    // The batched split-K GEMM carries PDL too (detail::dense_gemm_pdl): it waits on its parent before it reads,
    // and releases the reduce after its MMAs, so the reduce's launch overlaps the partial-sum epilogue (the
    // reduce still waits for this grid's completion before it reads a partial).
    if (Gemm::run(kernel_params, stream, nullptr, detail::dense_gemm_pdl()) != cutlass::Status::kSuccess)
        return cudaErrorLaunchFailure;

    // Reduce: N is a multiple of 128 on this surface, so M*N % 4 == 0.
    long const mn4 = mn / 4;
    int const threads = 256;
    int const blocks = static_cast<int>((mn4 + threads - 1) / threads) < 1024
        ? static_cast<int>((mn4 + threads - 1) / threads)
        : 1024;
    // PDL on the reduce (see the comment above the reduce kernel): it becomes
    // the GEMM's programmatic dependent and waits for the GEMM's completion
    // before it reads a partial, a bias or a residual.
    bool const pdl = tensorrt_llm::kernels::blockscale_gemm::fso_pdl_enabled();
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = dim3(blocks);
    cfg.blockDim = dim3(threads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    if (epilogue.is_identity())
    {
        if (pdl)
        {
            cudaLaunchKernelEx(&cfg, sm100_mxfp8_splitk_reduce_kernel,
                reinterpret_cast<float4 const*>(partials), reinterpret_cast<ushort4*>(mat_d), splits, mn4, true);
        }
        else
        {
            sm100_mxfp8_splitk_reduce_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<float4 const*>(partials), reinterpret_cast<ushort4*>(mat_d), splits, mn4, false);
        }
        return cudaGetLastError();
    }
    // The quantized output (linear_quantized): the third reduce, defined in ops/mxfp8_sm100_epi_quant_kernel_nosmem.cu
    // (declared in dense_common.cuh), with the same PDL discipline as the other two.
    if (epilogue.out_q != nullptr)
        return launch_sm100_mxfp8_splitk_reduce_q(partials, splits, M, N, stream, epilogue, pdl);
    // The dense-epilogue reduce lives in its own translation unit (see the note above pick_splits).
    return launch_sm100_mxfp8_splitk_reduce_epi(partials, mat_d, splits, mn4, N / 4, epilogue.bias, epilogue.act,
        epilogue.residual, static_cast<long>(epilogue.ld_residual), pdl, blocks, threads, stream);
}

namespace detail
{

// One call of the dense cascade: its operands, its epilogue, its workspace, and whether it only asks for the
// workspace size (query != nullptr: every launcher writes the bytes it needs there and launches nothing). The
// cascade below makes its choice once, through run / pick / splitk, so the size the op body allocates is the size
// the launch then uses.
struct Sm100DenseCall
{
    __nv_fp8_e4m3* a;
    __nv_fp8_e4m3* b;
    __nv_bfloat16* d;
    int32_t* sfa;
    int32_t* sfb;
    int M, N, K;
    cudaStream_t stream;
    fso::DenseEpilogue const& epilogue;
    Sm100DenseWorkspace workspace;
    std::size_t* query;
    // The FSO_FORCE_TILE row being served (st = 0 on the cascade's own picks), for refusal messages.
    int forced_tm = 0, forced_tn = 0, forced_st = 0;

    // A forced row or a cascade pick: the identity kernel for an identity epilogue, the quantized-output kernel of
    // the same configuration when the epilogue writes the quantized form (linear_quantized; `d` is unused then),
    // the dense-epilogue kernel otherwise, and a refusal (before anything is launched) for a configuration that
    // has none.
    template <int TM, int TN, int CM, int CN, int TK, bool SK, bool NS>
    cudaError_t run(int swizzle)
    {
        if (epilogue.is_identity())
            return launch_sm100_mxfp8_gemm<TM, TN, CM, CN, TK, SK, NS>(
                a, b, d, sfa, sfb, M, N, K, stream, swizzle, workspace, query);
        if constexpr (!SK && Sm100DenseEpiInstantiated<TM, TN, CM, CN, TK, NS>::value)
        {
            if (epilogue.out_q != nullptr)
                return launch_sm100_mxfp8_gemm_epi_q<TM, TN, CM, CN, TK, NS>(
                    a, b, sfa, sfb, M, N, K, stream, epilogue, workspace, swizzle, query);
            if (epilogue.residual != nullptr)
                return launch_sm100_mxfp8_gemm_epi<TM, TN, CM, CN, TK, NS, true>(
                    a, b, d, sfa, sfb, M, N, K, stream, epilogue, workspace, swizzle, query);
            return launch_sm100_mxfp8_gemm_epi<TM, TN, CM, CN, TK, NS, false>(
                a, b, d, sfa, sfb, M, N, K, stream, epilogue, workspace, swizzle, query);
        }
        else
        {
            refuse(TM, TN, CM, CN, TK, SK, NS);
            return cudaErrorNotSupported;
        }
    }

    // A pick of the cascade's own rules: it must have a dense-epilogue instantiation (checked at compile time, so
    // a rule added without one fails the build instead of refusing at run time).
    template <int TM, int TN, int CM, int CN, int TK, bool SK, bool NS>
    cudaError_t pick(int swizzle)
    {
        static_assert(!SK && Sm100DenseEpiInstantiated<TM, TN, CM, CN, TK, NS>::value,
            "every configuration the sm_100 dense cascade's rules pick needs a dense-epilogue instantiation "
            "(FSO_SM100_DENSE_EPI_CONFIGS_* in dense_common.cuh)");
        return run<TM, TN, CM, CN, TK, SK, NS>(swizzle);
    }

    template <int TM, int TN>
    cudaError_t splitk(int splits)
    {
        return launch_sm100_mxfp8_gemm_splitk<TM, TN>(
            a, b, d, sfa, sfb, M, N, K, splits, stream, epilogue, workspace, query);
    }

    [[noreturn]] void refuse(int tm, int tn, int cm, int cn, int tk, bool sk, bool ns) const
    {
        char cfg[160];
        std::snprintf(cfg, sizeof(cfg), "(%d x %d x %d, cluster (%d, %d), %s)", tm, tn, tk, cm, cn,
            sk ? "Stream-K scheduler" : (ns ? "direct-store epilogue" : "TMA epilogue"));
        char forced[96];
        std::snprintf(forced, sizeof(forced), "FSO_FORCE_TILE=%d,%d,%d", forced_tm, forced_tn, forced_st);
        std::string msg = "sm_100/sm_103 dense MXFP8 GEMM (gemm_dispatch_sm100_mxfp8, the C++ CUTLASS cascade, "
                          "tier 3): ";
        msg += forced_st != 0 ? std::string(forced) + " selects the configuration " + cfg
                              : std::string("the configuration ") + cfg;
        if (sk)
            msg += ", whose Stream-K scheduler reduces partial tiles inside the kernel and has no dense-epilogue "
                   "instantiation (force codes 7 and 8 are sweep-only)";
        else
            msg += ", which has no dense-epilogue instantiation";
        msg += "; the dense epilogue (" + fso::dense_epilogue_field_names(epilogue.fields())
            + ") runs on the configurations the cascade's own rules pick (forced: codes 20 / 21 / 24 / 25, the "
              "1, 2 and 4 rows of those tiles, and 9 for the split-K). Unset FSO_FORCE_TILE or pick one of those";
        throw fso::DenseEpilogueNotImplemented(msg);
    }
};

// A residual the dense-epilogue kernels can read: 16-byte aligned rows (the TMA epilogue's C descriptor, the
// direct-store epilogue's 16-byte loads, the split-K reduce's 8-byte loads), at least N wide.
inline void require_sm100_residual_layout(fso::DenseEpilogue const& epilogue, int N, char const* where)
{
    if (epilogue.residual == nullptr)
        return;
    bool const aligned = (reinterpret_cast<std::uintptr_t>(epilogue.residual) % 16u) == 0u;
    bool const pitch_ok = epilogue.ld_residual >= N && (epilogue.ld_residual % 8) == 0;
    if (aligned && pitch_ok)
        return;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        ": the residual must start on a 16-byte boundary and have a row pitch that is a multiple of 8 elements and "
        "at least N = %d (got address %% 16 = %u, row pitch %lld)",
        N, static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(epilogue.residual) % 16u),
        static_cast<long long>(epilogue.ld_residual));
    throw fso::DenseEpilogueNotImplemented(std::string(where) + buf);
}

// A quantized output (linear_quantized) this route can write: no residual (the op has no such argument; the
// quantized-output kernels have no C source), the scale buffer in the sm_100 contract layout (kDenseOutScaleMxSm100,
// fso/activation_contract.h), N a multiple of 128 (the codes are the next GEMM's activation; the TMA store of the
// codes needs 16-byte rows), out_q on a 16-byte boundary and out_scale on a 4-byte one.
inline void require_sm100_quant_out(fso::DenseEpilogue const& epilogue, int N, char const* where)
{
    if (epilogue.out_q == nullptr)
        return;
    char const* why = nullptr;
    if (epilogue.residual != nullptr)
        why = ": the quantized output (linear_quantized) takes no residual";
    else if (epilogue.out_scale == nullptr || epilogue.out_scale_layout != fso::kDenseOutScaleMxSm100)
        why = ": the quantized output's scale must be the sm_100/sm_103 layout of activation contract v1 "
              "(kDenseOutScaleMxSm100: int32 words in the Sm1xx atom layout over align(M, 128) rows)";
    else if (N % 128 != 0)
        why = ": the quantized output needs N % 128 == 0 (it is the next GEMM's activation)";
    else if ((reinterpret_cast<std::uintptr_t>(epilogue.out_q) % 16u) != 0u
        || (reinterpret_cast<std::uintptr_t>(epilogue.out_scale) % 4u) != 0u)
        why = ": the quantized output must start on a 16-byte boundary and its scale on a 4-byte one";
    if (why != nullptr)
        throw fso::DenseEpilogueNotImplemented(std::string(where) + why);
}

// The cascade: FSO_FORCE_TILE rows first, then the rules. Bring-up history (C1):
//   * decode / small M          → (64, 128) 1SM
//   * mid M                     → (128, 128) 1SM;  wide N → (128, 256) 1SM
//   * large M (> 256)           → (256, 128) 2SM;  wide N → (256, 256) 2SM
// No Stream-K: the CLC-based persistent scheduler already load-balances
// tail waves; revisit only if the b300 sweep shows narrow-N long-K decode
// shapes losing to a K-split (compare against sm_120's K≥9728 gate).
inline cudaError_t sm100_dense_cascade(Sm100DenseCall& call)
{
    int const M = call.M, N = call.N, K = call.K;
    // Epilogue binding (ncu A/B 2026-07-06, see gemm_types.cuh): NOSMEM
    // (direct-store) on the latency-bound decode tiles and the L2-headroom
    // peak tile; TMA epilogue everywhere L2 is near-bound (mid-band
    // (256,128)K256) or unmeasured (StreamK, forced-only variants, and the
    // (256,256)K128 *mid-band* narrow-N route — same tile as peak but bound
    // TMA until probed, hence the separate NOSMEM/TMA call sites below).
    // FORCED_* serve a FSO_FORCE_TILE row (any instantiated configuration);
    // DISPATCH_* are the rules' own picks (dense-epilogue instantiation
    // required at compile time).
#define FORCED_SM100_MX(TM_, TN_, CM_, CN_, TK_, SK_) call.run<TM_, TN_, CM_, CN_, TK_, SK_, false>(0)
#define FORCED_SM100_MX_NOSMEM(TM_, TN_, CM_, CN_, TK_, SK_) call.run<TM_, TN_, CM_, CN_, TK_, SK_, true>(0)
#define DISPATCH_SM100_MX(TM_, TN_, CM_, CN_, TK_, SK_) call.pick<TM_, TN_, CM_, CN_, TK_, SK_, false>(0)
#define DISPATCH_SM100_MX_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) call.pick<TM_, TN_, CM_, CN_, TK_, SK_, false>(SW_)
#define DISPATCH_SM100_MX_NOSMEM(TM_, TN_, CM_, CN_, TK_, SK_) call.pick<TM_, TN_, CM_, CN_, TK_, SK_, true>(0)
#define DISPATCH_SM100_MX_NOSMEM_SW(TM_, TN_, CM_, CN_, TK_, SK_, SW_) \
    call.pick<TM_, TN_, CM_, CN_, TK_, SK_, true>(SW_)

    // FSO_FORCE_TILE=TM,TN,ST — ST picks the variant (see header comment).
    // Valid tiles: TileM ∈ {128, 256} (SF Blk_MN=128 granularity on the M
    // axis); TileN ∈ {64, 128, 192, 256} (CUTLASS pads the SF block up on the
    // N axis), but only the (TileM, TileN, ST) rows written out below are
    // actually instantiated.
    auto forced = tensorrt_llm::kernels::blockscale_gemm::read_force_tile();
    if (forced.active() && forced.tm > 0
        && tensorrt_llm::kernels::blockscale_gemm::force_tile_applies(
            forced, static_cast<uint32_t>(K)))
    {
        int const tm = forced.tm, tn = forced.tn, st = forced.st;
        call.forced_tm = tm;
        call.forced_tn = tn;
        call.forced_st = st;
        // Codes 20..25: explicit (SM count, TileK, cluster, epilogue) with a
        // free TileN — the sweep surface for the 64- and 192-wide tiles.
        if (st == 20 && tm == 128)
        {
            if (tn == 64) return FORCED_SM100_MX_NOSMEM(128, 64, 1, 1, 128, false);
            if (tn == 128) return FORCED_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
            if (tn == 192) return FORCED_SM100_MX_NOSMEM(128, 192, 1, 1, 128, false);
            if (tn == 256) return FORCED_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        }
        if (st == 21 && tm == 128)
        {
            if (tn == 64) return FORCED_SM100_MX(128, 64, 1, 1, 128, false);
            if (tn == 128) return FORCED_SM100_MX(128, 128, 1, 1, 128, false);
            if (tn == 192) return FORCED_SM100_MX(128, 192, 1, 1, 128, false);
            if (tn == 256) return FORCED_SM100_MX(128, 256, 1, 1, 128, false);
        }
        if (st == 22 && tm == 128 && tn == 64) return FORCED_SM100_MX_NOSMEM(128, 64, 1, 1, 256, false);
        if (st == 23 && tm == 128 && tn == 64) return FORCED_SM100_MX(128, 64, 1, 1, 256, false);
        if (st == 24 && tm == 256)
        {
            if (tn == 64) return FORCED_SM100_MX_NOSMEM(256, 64, 2, 1, 128, false);
            if (tn == 192) return FORCED_SM100_MX_NOSMEM(256, 192, 2, 1, 128, false);
            if (tn == 256) return FORCED_SM100_MX_NOSMEM(256, 256, 2, 1, 128, false);
        }
        if (st == 25 && tm == 256)
        {
            if (tn == 64) return FORCED_SM100_MX(256, 64, 2, 1, 128, false);
            if (tn == 128) return FORCED_SM100_MX(256, 128, 2, 1, 128, false);
            if (tn == 192) return FORCED_SM100_MX(256, 192, 2, 1, 128, false);
            if (tn == 256) return FORCED_SM100_MX(256, 256, 2, 1, 128, false);
        }
        // 1SM rows (TileM=128) — K128 rows carry the cascade's NOSMEM binding
        // so FSO_FORCE_TILE sweeps exercise the production instantiation.
        if (tm == 128 && tn == 128 && st == 1) return FORCED_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        if (tm == 128 && tn == 256 && st == 1) return FORCED_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        if (tm == 128 && tn == 128 && st == 3) return FORCED_SM100_MX(128, 128, 1, 1, 256, false);
        if (tm == 128 && tn == 256 && st == 3) return FORCED_SM100_MX(128, 256, 1, 1, 256, false);
        if (tm == 128 && tn == 128 && st == 7) return FORCED_SM100_MX(128, 128, 1, 1, 128, true);
        if (tm == 128 && tn == 128 && st == 8) return FORCED_SM100_MX(128, 128, 1, 1, 256, true);
        if (tm == 128 && tn == 128 && st == 9 && (K % 128) == 0)
        {
            int const tiles = ((M + 127) / 128) * ((N + 127) / 128);
            int const ks = forced.ks > 1 ? forced.ks : detail::pick_splits(tiles, K / 128);
            if (ks > 1 && (K / 128) % ks == 0)
                return call.splitk<128, 128>(ks);
            // No valid split → plain decode tile.
            return FORCED_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
        }
        // 1SM cluster(2,2) probes (ncu F3: nvjet's gate M=2048 config is
        // 128×256/CTA with a 4-CTA cluster → half our L2 traffic).
        // NEGATIVE RESULT (2026-07-06 c22_probe): all three lose across the
        // whole mid-band — (128,256)c22 +15..28% (gate M2048 52.6 vs 41.0),
        // (128,128)c22 +9..20%, (256,128)c22-K256 +1..8%. The L2-traffic
        // saving does not survive the CLC scheduler's coarser 4-CTA
        // allocation granularity + cluster barriers. nvjet's mid-band edge
        // is scheduler/pipeline-level, not reachable by cluster shape alone.
        // Kept as force codes for future sweeps; do NOT wire into the
        // cascade.
        if (tm == 128 && tn == 128 && st == 10) return FORCED_SM100_MX(128, 128, 2, 2, 128, false);
        if (tm == 128 && tn == 256 && st == 10) return FORCED_SM100_MX(128, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 128 && st == 11) return FORCED_SM100_MX(256, 128, 2, 2, 256, false);
        // 2SM rows (TileM=256)
        if (tm == 256 && tn == 128 && st == 2) return FORCED_SM100_MX(256, 128, 2, 1, 128, false);
        if (tm == 256 && tn == 256 && st == 2) return FORCED_SM100_MX_NOSMEM(256, 256, 2, 1, 128, false);
        if (tm == 256 && tn == 128 && st == 4) return FORCED_SM100_MX(256, 128, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 4) return FORCED_SM100_MX(256, 256, 2, 1, 256, false);
        if (tm == 256 && tn == 256 && st == 5) return FORCED_SM100_MX(256, 256, 2, 2, 128, false);
        if (tm == 256 && tn == 256 && st == 6) return FORCED_SM100_MX(256, 256, 2, 2, 256, false);
        // Unknown forced combo → fall through to the cascade (sm_120 contract).
        call.forced_tm = call.forced_tn = call.forced_st = 0;
    }

    // Cascade v2 — rules distilled from the 2026-07-05 b300 round-2 tile
    // sweep (runs/b300_tile_sweep_v2.jsonl: 12 variants × Qwen3 grid,
    // best-shot cudagraph µs). Key findings baked in:
    //   * (256,128) 2SM c(2,1) K256 is the mid-band workhorse
    //     (128 < M ≤ 2048: best-or-tie on wqkv/wo/gate/down).
    //   * StreamK in heuristic mode LOSES to plain tiles everywhere
    //     (down M=1: 30.9 vs 17.5 µs) — removed from the default route
    //     until explicit-splits (FSO_FORCE_KSPLIT) proves out.
    //   * c(2,2) and K256 do NOT help the peak band (cubic-8192:
    //     c22 410 µs / K256 435 µs vs c21-K128 370 µs).
    //   * down-class decode floor is (256,128)K256 ≈ 16.7 µs vs cuBLAS
    //     ~10.4 µs — the remaining gap needs split-K or a custom path,
    //     tracked separately.
    int const tiles_n = (N + 127) / 128;

    // Parallel split-K for down-class decode (ncu F1): narrow-N long-K
    // shapes whose tile grid can't fill the machine. pick_splits returns 1
    // (→ fall through) once the grid fills, so the route self-limits to
    // decode. Band from the 2026-07-06 sk_probe (best-shot µs):
    //   * M ≤ 32, K ≥ 4096: always wins — down 17.2→11.7 (BEATS cuBLAS
    //     12.3), wo 10.6→9.9. wqkv/gate K=2560 excluded (forced-SK loses,
    //     11.4 vs 10.1: split ×2 + reduce isn't amortized at short K).
    //   * 32 < M ≤ 128: only long-K — down K=9728 wins (16.4→12.9 M=64,
    //     15.7 M=128) but wo K=4096 loses (10.3→13.9) → K ≥ 8192.
    //   * Deeper splits lose (down M=1: S=19 14.9 vs S=4 12.9) — the ≤ 8
    //     cap plus largest-divisor pick in pick_splits is the sweet spot.
    //   * The long-K clause ends at M = 104 (2026-10-11; it ended at 128).
    //     Above it the wave rule's one-kernel 2-SM 256x64 tile (the same 80
    //     CTAs, full K, no FP32 partials, no reduce) beats the split: the
    //     partials (S * M * N * 4 bytes written and read back) and the reduce
    //     grow with M while the wave tile's cost does not. Measured on the
    //     Qwen3-4B `down` (N = 2560, K = 9728) with cold weights inside the MLP
    //     block, split-K against the wave tile: M = 96 -1.0 %, 104 -0.7 %,
    //     112 +1.0 %, 120 +2.2 %, 128 +2.9 % (split-K slower), and alone
    //     M = 96 -3.5 %, 128 +6.5 % (run fso_0.2.1_20261010/P1, ab5_dnband and
    //     ab6_gemmprobe). Only `down` among the dense table shapes has K >= 8192.
    if (tiles_n <= 32 && (K % 128) == 0
        && ((M <= 32 && K >= 4096) || (M <= 104 && K >= 8192)))
    {
        int const tiles = ((M + 127) / 128) * tiles_n;
        int const ks = detail::pick_splits(tiles, K / 128);
        if (ks > 1)
            return call.splitk<128, 128>(ks);
    }

    // --- Short-K tiles (issue #12, T1) ----------------------------------
    // detail::pick_short_k_tile carries the rule and its measurements. It only
    // answers K <= 2560 (never a split-K cell above) and returns 0 where none
    // of its tiles fits one wave, so those cells reach the wave rule and the
    // band rules below exactly as before.
    {
        bool const fused = call.epilogue.act != fso::kDenseActNone || call.epilogue.out_q != nullptr;
        int const short_k_tile = detail::pick_short_k_tile(M, N, K, fused);
        switch (short_k_tile)
        {
        case 128 * 1000 + 64:  return DISPATCH_SM100_MX(128, 64, 1, 1, 128, false);
        case 256 * 1000 + 64:  return DISPATCH_SM100_MX(256, 64, 2, 1, 128, false);
        case 128 * 1000 + 128: return DISPATCH_SM100_MX(128, 128, 1, 1, 128, false);
        case 256 * 1000 + 128: return DISPATCH_SM100_MX(256, 128, 2, 1, 128, false);
        default: break;
        }
    }

    // --- N-tile chosen by wave arithmetic (2026-09-17) -----------------
    // detail::pick_wave_tile carries the rule and its measured bounds; it
    // returns 0 for every cell this round did not measure a win on, and those
    // fall through to the rules distilled in b300_mlp_tune_20260915 unchanged.
    // The parallel split-K branch above keeps precedence, so the narrow-N
    // long-K decode cells it owns are untouched.
    {
        int const wave_tile = detail::pick_wave_tile(M, N, K);
        switch (wave_tile)
        {
        case 128 * 1000 + 64:  return DISPATCH_SM100_MX(128, 64, 1, 1, 128, false);
        case 128 * 1000 + 192: return DISPATCH_SM100_MX(128, 192, 1, 1, 128, false);
        case 256 * 1000 + 64:  return DISPATCH_SM100_MX(256, 64, 2, 1, 128, false);
        case 256 * 1000 + 192: return DISPATCH_SM100_MX(256, 192, 2, 1, 128, false);
        default: break;
        }
    }

    if (M <= 8)
    {
        // Extreme decode: 1SM (128,128); very wide N amortizes the per-CTA
        // prologue with a 256-wide tile (gate_up 15.7 vs 19.9, gate 12.5
        // vs 13.5 µs). NOSMEM: latency-bound, L2 SOL ≈ 12% — direct store
        // wins (wqkv M1 10.75→10.50, down M1 17.60→17.15).
        if (N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        return DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false);
    }
    if (M <= 128)
    {
        // NOSMEM only while the M-tile is mostly empty (M ≤ 32: stores are
        // a sliver, latency wins — gate M1-32 −9..−16%). At fuller M-tiles
        // with this band's short-K shapes the epilogue dominates and the
        // TMA bulk store wins big (v23 bench: gate M=128 8.2→10.3 µs,
        // gate_up M=128 12.3→14.4 µs under NOSMEM → reverted to TMA).
        bool const nosmem = (M <= 32);
        // gate_up-class stays 1SM 256-wide through M=128 (13.3 vs 15.6+).
        if (N >= 16384)
            return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false)
                          : DISPATCH_SM100_MX(128, 256, 1, 1, 128, false);
        // gate M ≤ 16 also prefers the 256-wide 1SM tile (12.4 vs 12.7).
        if (M <= 16 && N >= 9728)
            return DISPATCH_SM100_MX_NOSMEM(128, 256, 1, 1, 128, false);
        // One 128-row M-tile: 2SM (256,128)K256 wins when the wave still
        // fits (wqkv tn=48: 9.3, down tn=20: 16.7); wide N overflows with
        // 2× CTAs and loses (gate tn=76: 12.1 vs 9.4 1SM) → stay 1SM.
        // M ≤ 32 is still latency-dominated and prefers 1SM (wqkv M=32:
        // 8.3 vs 10.3 µs under the M>16 2SM rule — bench 2026-07-05).
        if (M > 32 && tiles_n <= 60)
            return DISPATCH_SM100_MX(256, 128, 2, 1, 256, false);
        return nosmem ? DISPATCH_SM100_MX_NOSMEM(128, 128, 1, 1, 128, false)
                      : DISPATCH_SM100_MX(128, 128, 1, 1, 128, false);
    }
    if (M <= 2048)
    {
        // Narrow N at M≈1024 halves the wave count with a 256-wide tile
        // (down/wo M=1024) — long-K only (cubic-1024/1536 regressed) and
        // not at M=2048 (wo regressed 18%).
        if (tiles_n <= 20 && M >= 1024 && M < 2048 && K >= 4096)
            return DISPATCH_SM100_MX(256, 256, 2, 1, 128, false);
        // Wide-N many-wave cells also benefit from raster swizzle
        // (gate_up M=2048 K256: sw8 75.3 vs sw0 79.3 µs p50, warm probe
        // 2026-07-05); narrow-N mid cells are flat → gate on tiles_n.
        int const mid_swizzle = (M >= 1024 && tiles_n >= 32) ? 8 : 0;
        return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 256, false, mid_swizzle);
    }
    // Peak band: 256-wide once N supplies enough tiles; K128 (K256 loses
    // ~6% at cubic-4096+, deeper pipeline wins over fewer iterations).
    // Raster swizzle 8 recovers L2 locality on many-wave problems
    // (cubic-8192: 370 → 349 µs +6%, cubic-12288/16384 +11/16%) — but
    // ONLY when N supplies at least a full swizzle group of tiles
    // (down M=4096, tiles_n=20: swizzle=8 cost 24%).
    // NOSMEM on the peak tile only when the mainloop is long enough to
    // amortize the slower scattered-STG store (K ≥ 4096: cubic-8192
    // 352→327 µs = 3360 TF ≈ cuBLAS parity, cubic-4096 −3.8%, L2 SOL 49%
    // has headroom). Short-K epilogue-heavy peak cells keep the TMA bulk
    // store (v23 bench: gate M=4096 K=2560 74.3→85.0 µs under NOSMEM).
    // Narrow-N (256,128) stays TMA (unmeasured).
    int const peak_swizzle = (tiles_n >= 32) ? 8 : 0;
    if (N >= 4096)
    {
        if (K >= 4096)
            return DISPATCH_SM100_MX_NOSMEM_SW(256, 256, 2, 1, 128, false, peak_swizzle);
        return DISPATCH_SM100_MX_SW(256, 256, 2, 1, 128, false, peak_swizzle);
    }
    return DISPATCH_SM100_MX_SW(256, 128, 2, 1, 128, false, peak_swizzle);

#undef FORCED_SM100_MX
#undef FORCED_SM100_MX_NOSMEM
#undef DISPATCH_SM100_MX
#undef DISPATCH_SM100_MX_SW
#undef DISPATCH_SM100_MX_NOSMEM
#undef DISPATCH_SM100_MX_NOSMEM_SW
}

} // namespace detail

// Top-level sm_100/sm_103 MXFP8 dispatcher.
//
// `epilogue`: the dense epilogue of interface v1 (common/epilogue.cuh). This route applies bias, erf-GELU and the
// residual (kDenseEpiServedSm100Cascade) on every configuration its rules pick and on its two-kernel split-K, and
// writes the quantized output of linear_quantized (epilogue.out_q / out_scale; bias and the activation, no
// residual) on the same configurations and on the split-K. It refuses, before it builds or launches anything, a
// residual it cannot read (alignment, require_sm100_residual_layout), a quantized output it cannot write
// (require_sm100_quant_out) and a FSO_FORCE_TILE row without a dense-epilogue instantiation. With out_q set,
// `mat_d` is not written (pass nullptr).
// `workspace`: the caller's device scratch for this call, at least gemm_workspace_sm100_mxfp8(...) bytes.
// `workspace_query`: when non-null, nothing is launched and *workspace_query receives the bytes this call needs
// (the refusals above are thrown in this mode too).
inline cudaError_t gemm_dispatch_sm100_mxfp8(__nv_fp8_e4m3* mat_a, __nv_fp8_e4m3* mat_b, __nv_bfloat16* mat_d,
    int32_t* scales_a, int32_t* scales_b, int M, int N, int K, cudaStream_t stream,
    fso::DenseEpilogue const& epilogue = fso::DenseEpilogue{}, Sm100DenseWorkspace const& workspace = {},
    std::size_t* workspace_query = nullptr)
{
    char const* const where
        = "sm_100/sm_103 dense MXFP8 GEMM (gemm_dispatch_sm100_mxfp8, the C++ CUTLASS cascade, tier 3)";
    fso::require_dense_epilogue(epilogue, fso::kDenseEpiServedSm100Cascade, where);
    detail::require_sm100_residual_layout(epilogue, N, where);
    detail::require_sm100_quant_out(epilogue, N, where);
    detail::Sm100DenseCall call{mat_a, mat_b, mat_d, scales_a, scales_b, M, N, K, stream, epilogue, workspace,
        workspace_query};
    return detail::sm100_dense_cascade(call);
}

// The bytes of caller workspace gemm_dispatch_sm100_mxfp8 needs for this call (host-only: the cascade's choice is
// made, nothing is launched). Throws fso::DenseEpilogueNotImplemented exactly when the dispatch would.
inline std::size_t gemm_workspace_sm100_mxfp8(int M, int N, int K, fso::DenseEpilogue const& epilogue)
{
    std::size_t bytes = 0;
    cudaError_t const err = gemm_dispatch_sm100_mxfp8(
        nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, nullptr, epilogue, Sm100DenseWorkspace{}, &bytes);
    return err == cudaSuccess ? bytes : 0; // a launch error is reported by the launch itself
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
