#!/usr/bin/env python3
"""Generate the sm_100 multi-warpgroup TMA epilogue of the dense cascade from CUTLASS's sm_100 TMA epilogue.

Which CUTLASS class is cloned
-----------------------------
`cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp` holds one class, the specialisation of
`CollectiveEpilogue` for `Sm100TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>`: the
smem-staged (TMA store) epilogue the dense cascade's dense-epilogue kernels (bias / erf-GELU, and the quantized output of
linear_quantized) run on their TMA tiles. Its subtile loop runs in four epilogue warps, one per SM sub-partition, each
warp reading its 32 TMEM lanes (rows) of every 128-row subtile.

What the clone changes
----------------------
The clone runs the same subtile loop in NumWG warpgroups of four warps (NumWG x 128 epilogue threads; the kernel takes
its epilogue thread count from the collective, so a CTA then has 4 + 4 NumWG warps). Warp w of every warpgroup reads the
same TMEM lanes 32 (w % 4) .. 32 (w % 4) + 31 the hardware assigns to its sub-partition, and warpgroup g takes the
subtiles l = g, g + NumWG, g + 2 NumWG, ... of the tile in CUTLASS's own (M-fastest) order. Every subtile is computed
exactly as before, by one warpgroup, with the same instructions on the same accumulator values: the per-element fp32
sequence of the epilogue (the fusion callbacks, bias / activation / quantizer) is untouched, so the output bytes are the
stock epilogue's. What changes is how many warps share the sub-partition's issue slots and pipes while the epilogue's
arithmetic runs: two warps per sub-partition instead of one, so one warp's dependency stalls are filled by the other's
instructions. The dense cascade uses the clone only for the erf-GELU calls of linear_quantized (the quantized output),
on its TMA tiles with two accumulator stages (dense_epilogue_gemm.cuh, kDenseQuantGeluEpiWarpgroups); every other call
keeps the stock class, for the reasons given there.

Per warpgroup: its own store buffers (StagesDWG stages of the stock subtile staging buffer, the warpgroup's own TMA store
issued by its first warp and committed / waited by that thread), its own named barrier (warpgroup 0 keeps CUTLASS's
EpilogueBarrier, warpgroup g > 0 takes the user barrier g - 1, i.e. hardware barrier 8 + g - 1), and its own copy of the
fusion callbacks' shared storage (the bias row the broadcast node stages per tile), so that one warpgroup staging tile
t + 1's bias never overwrites what the other still reads for tile t. Every epilogue thread waits on the accumulator before
its warpgroup's first subtile and releases it after its warpgroup's last TMEM load, so the accumulator pipeline sees the
NumWG x 128 arrivals per tile its consumer count (NumEpilogueThreads in the kernel) expects.

Each warpgroup waits for its store buffer right before it writes the buffer (CUTLASS waits right after it issues the
previous TMA store): with one buffer per warpgroup the stock order would hold the warpgroup at every subtile until its
own store had read the buffer, while the moved wait lets the store drain during the next subtile's TMEM load and
arithmetic. The waits, barriers and their count per subtile are CUTLASS's; only their place in the subtile changes.

Not served by the clone (static_asserts): a C source (the residual; its producer-load pipeline hands every subtile to all
epilogue threads), smem reuse of C, and the overlapping-accumulator path of the 256-wide tiles (ReuseTmem: a single
accumulator stage whose subtile order alternates per tile). Those configurations keep CUTLASS's own class. The second
store() overload of the CUTLASS class (register accumulators, for the emulated-FP32 kernels) is dropped.

Nothing in 3rdparty/cutlass is modified; the generated header is written into the build directory only. Each edit asserts
its exact match count, so a CUTLASS bump that moves any of them fails the build here, naming the edit.

usage: make_sm100_dense_epi_multiwg.py <output header path>
"""
from __future__ import annotations

import os
import sys

HEADER_REL = "include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp"
CLASS_NAME = "FsoSm100TmaEpilogueMultiWG"

CLASS_HEAD = """template <
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  class CtaTileShape_, // (CTA_M,CTA_N,CTA_K, optional: Tile_L)
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyOpR2R_
>
class CollectiveEpilogue<
    Sm100TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>,
    CtaTileShape_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpT2R_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyOpR2R_
> {"""

NEW_HEAD = """template <
  int NumWG_,     // fso: epilogue warpgroups (128 threads each)
  int StagesDWG_, // fso: store buffers per warpgroup
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  class CtaTileShape_, // (CTA_M,CTA_N,CTA_K, optional: Tile_L)
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyOpR2R_
>
class """ + CLASS_NAME + """ {"""


def _cutlass_dir() -> str:
    for key in ("CUTLASS_DIR", "BSGEMM_CUTLASS_DIR"):
        cand = os.environ.get(key)
        if cand and os.path.isdir(os.path.join(cand, "include", "cutlass")):
            return cand
    here = os.path.dirname(os.path.abspath(__file__))
    sub = os.path.normpath(os.path.join(here, "..", "..", "..", "3rdparty", "cutlass"))
    if os.path.isdir(os.path.join(sub, "include", "cutlass")):
        return sub
    raise SystemExit("[fso] multi-warpgroup dense epilogue: CUTLASS not found; set CUTLASS_DIR")


def _extract_class(text: str) -> str:
    cnt = text.count(CLASS_HEAD)
    assert cnt == 1, f"[1 extract] found {cnt} copies of the class head, expected 1"
    start = text.index(CLASS_HEAD)
    i = start + len(CLASS_HEAD)
    depth = 1
    while depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert text[i] == ";", "[1 extract] class body does not end in '};': " + repr(text[i - 2:i + 2])
    return text[start:i + 1]


def _edits() -> list[tuple[str, str, int, str]]:
    e: list[tuple[str, str, int, str]] = []
    e.append((CLASS_HEAD, NEW_HEAD, 1, "2 primary-template head"))
    e.append(("""  constexpr static int ThreadCount = 128;
""", """  // fso: NumWG warpgroups of four epilogue warps; the kernel sizes its epilogue role from ThreadCount.
  constexpr static int NumWG = NumWG_;
  constexpr static int WgThreadCount = 128;
  constexpr static int ThreadCount = WgThreadCount * NumWG;
  static_assert(NumWG >= 1 && NumWG <= 3, "fso multi-warpgroup epilogue: 1 to 3 warpgroups");
""", 1, "3 thread count"))
    e.append(("""  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
""", """  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesDWG_; // fso: store buffers per warpgroup (each warpgroup has its own)
""", 1, "4 per-warpgroup store stages"))
    e.append(("""  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
""", """  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  static_assert(not is_source_supported, "fso multi-warpgroup epilogue: no C source (the residual keeps CUTLASS's class)");
  static_assert(not ReuseSmemC, "fso multi-warpgroup epilogue: no smem reuse of C");
""", 1, "5 no C source"))
    # the D staging storage: NumWG copies of the per-warpgroup buffers (three unions / structs name smem_D)
    e.append(("""    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
""", """    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD> * NumWG> smem_D; // fso: per warpgroup
""", 2, "6 D storage per warpgroup"))
    e.append(("""      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
""", """      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread[NumWG]; // fso: one per warpgroup (the bias row is staged per warpgroup and tile)
""", 1, "7 fusion storage per warpgroup"))
    e.append(("""  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}
""", """  // fso: the warpgroup of this thread. The kernel places its four non-epilogue warps first, so the epilogue threads
  // are threadIdx.x in [128, 128 + ThreadCount); every other thread (which constructs the collective too, but never
  // calls store()) reads as warpgroup 0.
  CUTLASS_DEVICE static int
  fso_wg_of_thread() {
    int const g = static_cast<int>(threadIdx.x) / WgThreadCount - 1;
    return g < 0 ? 0 : (g >= NumWG ? NumWG - 1 : g);
  }

  // fso: the named barrier of warpgroup g: CUTLASS's EpilogueBarrier for g = 0, the user barrier g - 1 otherwise.
  CUTLASS_DEVICE static void
  fso_wg_sync(int g) {
    if (g == 0) {
      cutlass::arch::NamedBarrier::sync(WgThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }
    else {
      cutlass::arch::NamedBarrier::sync(WgThreadCount, static_cast<uint32_t>(g - 1));
    }
  }

  CUTLASS_DEVICE
  """ + CLASS_NAME + """(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread[fso_wg_of_thread()]) {}
""", 1, "8 constructor: the warpgroup's fusion storage"))
    # store(): indices, buffers, barrier
    e.append(("""    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(accumulators) == 3, "Accumulators must be MMA-partitioned: [MMA, MMA_M, MMA_N]");
""", """    static_assert(is_tmem<AccEngine>::value, "Accumulator must be TMEM resident.");
    static_assert(rank(accumulators) == 3, "Accumulators must be MMA-partitioned: [MMA, MMA_M, MMA_N]");
    static_assert(not ReuseTmem, "fso multi-warpgroup epilogue: not for the overlapping-accumulator (256-wide) tiles");
""", 1, "9 no tmem reuse"))
    e.append(("""    int thread_idx = threadIdx.x % ThreadCount;
    int warp_idx = thread_idx / NumThreadsPerWarp;
    [[maybe_unused]] int lane_idx = thread_idx % NumThreadsPerWarp;

    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).""", """    // fso: thread_idx and warp_idx are this thread's index within its warpgroup, which is what the TMEM, register and
    // smem partitions below are built for (each warpgroup partitions a subtile exactly as the stock four warps do).
    int const fso_wg = fso_wg_of_thread();
    int thread_idx = threadIdx.x % WgThreadCount;
    int warp_idx = thread_idx / NumThreadsPerWarp;
    [[maybe_unused]] int lane_idx = thread_idx % NumThreadsPerWarp;

    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).""", 1, "10 warpgroup-local thread index"))
    e.append(("""    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(""", """    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sD = shared_tensors.collective.smem_D.begin() + fso_wg * cosize_v<SmemLayoutD>; // fso: this warpgroup's
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(""", 1, "11 warpgroup store buffers"))
    e.append(("""    auto synchronize = [] () { cutlass::arch::NamedBarrier::sync(ThreadCount, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for sub-128 thread T2R tiled copy""", """    auto synchronize = [fso_wg] () { fso_wg_sync(fso_wg); }; // fso: the warpgroup's own barrier

    // Predication for sub-128 thread T2R tiled copy""", 1, "12 warpgroup barrier"))
    e.append(("""      constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(gD_epi));
      constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(gD_epi));
      #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesN : 1)
      for (int iter_n = 0; iter_n < NumEpiSubtilesN; ++iter_n) {
        #pragma unroll(UnrollEpiLoop ? NumEpiSubtilesM : 1)
        for (int iter_m = 0; iter_m < NumEpiSubtilesM; ++iter_m) {
          int epi_m = iter_m, epi_n = iter_n;
          bool is_first_iteration = iter_m == 0 && iter_n == 0;
          bool is_last_iteration = iter_m == size<2>(gD_epi)-1 && iter_n == size<3>(gD_epi)-1;
          bool do_acc_release = is_last_iteration;
""", """      constexpr int NumEpiSubtilesN = CUTE_STATIC_V(size<3>(gD_epi));
      constexpr int NumEpiSubtilesM = CUTE_STATIC_V(size<2>(gD_epi));
      // fso: warpgroup fso_wg takes the subtiles l = fso_wg, fso_wg + NumWG, ... of CUTLASS's order (M fastest).
      // "first" and "last" are this warpgroup's first and last subtile: its threads wait for the accumulator before
      // the first and release it after the TMEM load of the last.
      constexpr int NumEpiSubtiles = NumEpiSubtilesN * NumEpiSubtilesM;
      static_assert(NumEpiSubtiles >= NumWG, "fso multi-warpgroup epilogue: fewer subtiles than warpgroups");
      #pragma unroll(UnrollEpiLoop ? (NumEpiSubtiles + NumWG - 1) / NumWG : 1)
      for (int fso_l = fso_wg; fso_l < NumEpiSubtiles; fso_l += NumWG) {
        {
          int const iter_n = fso_l / NumEpiSubtilesM;
          int const iter_m = fso_l % NumEpiSubtilesM;
          int epi_m = iter_m, epi_n = iter_n;
          bool is_first_iteration = fso_l == fso_wg;
          bool is_last_iteration = fso_l + NumWG >= NumEpiSubtiles;
          bool do_acc_release = is_last_iteration;
""", 1, "13 subtile split across warpgroups"))
    # the store buffer's wait moves from the end of tma_store_fn to right before the buffer is written
    e.append(("""        // Wait for the next smem buffer to be available
        if (issue_tma_store) {
          store_pipeline.producer_acquire(store_pipe_producer_state);
        }
        synchronize();
""", """        // fso: the wait for the next smem buffer moved to right before the buffer is written (see below)
""", 1, "15a no buffer wait after the TMA store"))
    e.append(("""          // Smem reduction callback entry point using current store buffer for workspace
          Tensor reduction_buffer = make_tensor(""", """          // fso: wait for this warpgroup's store buffer here, right before the subtile is written to it, so that the
          // previous TMA store from it drains while this subtile's TMEM load and arithmetic run
          if (issue_tma_store) {
            store_pipeline.producer_acquire(store_pipe_producer_state);
          }
          synchronize();

          // Smem reduction callback entry point using current store buffer for workspace
          Tensor reduction_buffer = make_tensor(""", 1, "15b buffer wait before the buffer is written"))
    return e


def _drop_register_store(body: str, report: list[str]) -> str:
    """Remove the second store() overload (register-resident accumulators of the emulated-FP32 kernels)."""
    head = "  // API with Global Accumulator in registers for FastFP32 (emulated MMA) kernels."
    tail = "  template <class CtaTileMNK>\n  CUTLASS_DEVICE void\n  store_tail("
    assert body.count(head) == 1, f"[14 drop register store] head found {body.count(head)} times"
    assert body.count(tail) == 1, f"[14 drop register store] tail found {body.count(tail)} times"
    a, b = body.index(head), body.index(tail)
    assert a < b
    report.append(f"edit 14 drop the register-accumulator store(): {b - a} chars")
    return body[:a] + "  // fso: the register-accumulator store() overload of the CUTLASS class is not cloned.\n\n" + body[b:]


def generate(cutlass_dir: str) -> tuple[str, list[str], str]:
    src_path = os.path.join(cutlass_dir, HEADER_REL)
    text = open(src_path).read()
    report: list[str] = []
    body = _extract_class(text)
    report.append(f"edit 1 extract the Sm100TmaWarpSpecialized epilogue: {len(body)} chars")
    body = _drop_register_store(body, report)
    for old, new, n, what in _edits():
        cnt = body.count(old)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}"
        body = body.replace(old, new, n)
        report.append(f"edit {what}: {cnt} site(s)")
    return _render(body, src_path), report, src_path


def _render(body: str, src_path: str) -> str:
    return """// GENERATED by csrc/gemm/tools/make_sm100_dense_epi_multiwg.py -- do not edit by hand, and do not check the result
// in: the build writes it into the build directory only.
//
// Clone of the CUTLASS SM100 TMA (smem-staged) epilogue whose subtile loop runs in NumWG warpgroups of four warps
// (see the generator's docstring).
// Source: """ + src_path + """
#pragma once

#include "cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp"

namespace cutlass {
namespace epilogue {
namespace collective {

""" + body + """

} // namespace collective
} // namespace epilogue
} // namespace cutlass
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: make_sm100_dense_epi_multiwg.py <output header path>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    text, report, src_path = generate(_cutlass_dir())
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    print("[fso] sm_100 multi-warpgroup dense epilogue: " + "; ".join(report))
    print("[fso]   from " + src_path)
    print("[fso]   wrote " + os.path.abspath(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
