#!/usr/bin/env python3
"""Generate the sm_100 fused-SwiGLU epilogue of the SLOT (swap-orientation)
route from CUTLASS's DENSE NoSmem epilogue.

Origin: the sibling generator `make_sm100_fused_swiglu_epilogue.py` does the
same thing for the pointer-array route, and `make_sm100_slot_kernel.py` forks
the dense kernel this epilogue is bound into. All three are run by the same
build hook, all three read a CUTLASS header and write a separate, independently
named class into the build directory, and none of them modifies anything under
`3rdparty/`.

Which CUTLASS class is cloned, and why that one
----------------------------------------------
`cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp` holds two
specialisations of the DENSE direct-store epilogue, chosen by whether the fusion
operation is the default one; the EVT specialisation (the one guarded by
`not IsDefaultFusionOp<FusionCallbacks_>::value`) is the one cloned here,
because it is what `OpClassBlockScaledTensorOp` selects and because it divides
the CTA tile by an `EpilogueTile` the NoSmem builder pins to
`(TileM, min(64, TileN))`. At the slot route's 128x64x128 tile that is one
subtile of the whole CTA tile, so a thread's accumulator run is the complete
64-column token tile.

The DENSE file, not the pointer-array one: the slot kernel is a fork of
CUTLASS's dense sm_100 kernel (fso's masked expert slab is one contiguous tensor
per operand with a uniform per-expert stride, so one 3-D TMA descriptor already
addresses every expert through the batch coordinate). Its epilogue arguments
therefore carry a single `ElementD* ptr_D` where the pointer-array epilogue
carries `ElementD**`, and the two classes differ in that and in the store body.

What the clone changes
----------------------
The EVT visit loop and the bf16 D store are replaced by one call to
`fso_swiglu_slot::fused_swiglu_slot_mxfp8_store`
(csrc/gemm/include/blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_slot_store.cuh),
which pairs the interleaved gate/up rows across lanes, computes silu(gate)*up in
FP32, reduces the 32-wide amax across a warp pair, derives the UE8M0 byte and
writes the fp8 bytes and the scale bytes. Everything else — the load of the
accumulator out of TMEM, the subtile walk, the coordinate tensors, the
predication — is CUTLASS's own code, unmodified.

The one resource the clone adds is `kSlotAmaxFloats` floats (1 KiB) of epilogue
shared storage for the cross-warp half of the amax. That is charged to
`StageCountAutoCarveout` like any other epilogue storage, and
`grouped_slot_dispatch.cuh` prints the resulting mainloop stage count under
FSO_PRINT_TILE_INFO=1 so the cost is measured rather than assumed. It is paid
for many times over by the epilogue this route no longer needs: the stock slot
kernel runs the TMA-store epilogue, whose staged bf16 D tiles are about sixteen
times larger.

Each edit asserts its exact match count, so a CUTLASS bump that moves any of
them fails the build here, naming the edit, rather than producing a silently
different kernel.
"""
from __future__ import annotations

import os
import sys

HEADER_REL = "include/cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp"

# The head of the dense EVT specialisation. Used both as the extraction anchor
# (edit 1) and as the literal edit 2 rewrites.
EVT_CLASS_HEAD = """template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_
>
class CollectiveEpilogue<
    Sm100NoSmem,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpT2R_,
    AlignmentC_,
    AlignmentD_,
    cute::enable_if_t<not IsDefaultFusionOp<FusionCallbacks_>::value>
> {"""

PRIMARY_CLASS_HEAD = """template <
  class EpilogueTile_, // (EPI_TILE_M, EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpT2R_,
  class AlignmentC_,
  class AlignmentD_
>
class FsoFusedSwiGluSlotNoSmem {"""


def _cutlass_dir() -> str:
    for key in ("CUTLASS_DIR", "BSGEMM_CUTLASS_DIR"):
        cand = os.environ.get(key)
        if cand and os.path.isdir(os.path.join(cand, "include", "cutlass")):
            return cand
    here = os.path.dirname(os.path.abspath(__file__))
    sub = os.path.normpath(os.path.join(here, "..", "..", "..", "3rdparty", "cutlass"))
    if os.path.isdir(os.path.join(sub, "include", "cutlass")):
        return sub
    raise SystemExit("[fso] fused SwiGLU slot epilogue: CUTLASS not found; set CUTLASS_DIR")


def generate(cutlass_dir: str) -> tuple[str, list[str], str]:
    """Return (generated source, per-edit report lines, source header path)."""
    src_path = os.path.join(cutlass_dir, HEADER_REL)
    text = open(src_path).read()
    report: list[str] = []

    # --- edit 1: extract the dense EVT specialisation by brace matching -----
    # Brace matching rather than a regex: a regex over 17 kB of nested templates
    # is not auditable, and the class body contains braces only in code and in
    # comments.
    cnt = text.count(EVT_CLASS_HEAD)
    assert cnt == 1, f"[1 extract] found {cnt} copies of the dense EVT class head, expected 1"
    start = text.index(EVT_CLASS_HEAD)
    i = start + len(EVT_CLASS_HEAD)
    depth = 1
    while depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert text[i] == ";", "[1 extract] class body does not end in '};': " + repr(text[i - 2:i + 2])
    body = text[start:i + 1]
    report.append(f"edit 1 extract EVT Sm100NoSmem specialisation: {len(body)} chars")

    edits: list[tuple[str, str, int, str]] = []

    # --- edit 2: partial specialisation -> primary class template -----------
    # fso instantiates the clone explicitly (through `RebindFusedSwiGluSlot` in
    # grouped_slot_dispatch.cuh), so it must not compete with CUTLASS's own
    # specialisation of `CollectiveEpilogue`.
    edits.append((EVT_CLASS_HEAD, PRIMARY_CLASS_HEAD, 1, "2 primary-template head"))

    # --- edit 3: the shared buffer the cross-warp amax exchange needs -------
    # The only resource the clone adds. It is declared inside the epilogue's own
    # SharedStorage so that `StageCountAutoCarveout` charges for it exactly as
    # it charges for the stock epilogue's storage.
    edits.append((
        """  struct SharedStorage {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
    array_aligned<uint8_t, ImplicitSharedStorageSize> buffer;
  };""",
        """  struct SharedStorage {
    using FusionStorage = typename FusionCallbacks::SharedStorage;
    FusionStorage thread;
    array_aligned<uint8_t, ImplicitSharedStorageSize> buffer;
    // ------------------------------------------- fso fused FC1 slot epilogue
    // One row of per-token maxima per epilogue warp. The 32 outputs of one 1x32
    // scale block span two warps in this orientation, so the amax needs one
    // cross-warp step; this is that step's buffer. 1 KiB, charged to
    // StageCountAutoCarveout like any other epilogue storage.
    alignas(16) float fso_amax[fso_swiglu_slot::kSlotAmaxFloats];
  };""",
        1, "3 shared amax buffer"))

    # --- edit 4: constructor name, and capturing the new buffer -------------
    edits.append((
        """  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, SharedStorage& shared_tensors)
  : fusion_callbacks(params_.thread, shared_tensors.thread)
  , smem_buffer_ptr(shared_tensors.buffer.data())
  , params(params_) {};

protected:
  FusionCallbacks fusion_callbacks;
  uint8_t* smem_buffer_ptr;
  Params const& params;""",
        """  CUTLASS_DEVICE
  FsoFusedSwiGluSlotNoSmem(Params const& params_, SharedStorage& shared_tensors)
  : fusion_callbacks(params_.thread, shared_tensors.thread)
  , smem_buffer_ptr(shared_tensors.buffer.data())
  , fso_amax_ptr(shared_tensors.fso_amax)   // fso fused FC1 slot epilogue
  , params(params_) {};

protected:
  FusionCallbacks fusion_callbacks;
  uint8_t* smem_buffer_ptr;
  float* fso_amax_ptr;                      // fso fused FC1 slot epilogue
  Params const& params;""",
        1, "4 constructor name + amax buffer capture"))

    # --- edits 5/6/7: carry the fused destinations through Arguments/Params -
    edits.append((
        """  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
    // ------------------------------------------- fso fused FC1 slot epilogue
    // The fused destinations are plain base pointers plus per-group element
    // counts, not the pointer arrays a grouped epilogue would need, because the
    // MoE slabs are contiguous in the group index. See FusedSwiGluSlotArgs in
    // blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_slot_store.cuh.
    fso_swiglu_slot::FusedSwiGluSlotArgs fused{};
  };""",
        1, "5 fused Arguments field"))

    edits.append((
        """  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const* ptr_C = nullptr;
    StrideC dC = {};
    ElementD* ptr_D = nullptr;
    StrideD dD = {};
    fso_swiglu_slot::FusedSwiGluSlotArgs fused{};   // fso fused FC1 slot epilogue
  };""",
        1, "6 fused Params field"))

    edits.append((
        """    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      args.ptr_C,
      args.dC,
      args.ptr_D,
      args.dD
    };""",
        """    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      args.ptr_C,
      args.dC,
      args.ptr_D,
      args.dD,
      args.fused          // fso fused FC1 slot epilogue
    };""",
        1, "7 to_underlying_arguments carries the fused args"))

    # --- edit 8: ordinary locals for the values the store needs ------------
    # M and N arrive as STRUCTURED BINDINGS and C++17 forbids capturing one in a
    # lambda; the epilogue loop, which contains the store, is a lambda. Binding
    # them to plain locals first is what makes the store callable from inside it.
    sync_anchor = (
        "    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE "
        "{ cutlass::arch::NamedBarrier::sync(ThreadCount, "
        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };")
    edits.append((
        sync_anchor,
        """    // fso fused FC1 slot epilogue: ordinary locals for the fused store (M and
    // N are structured bindings, which C++17 does not allow a lambda to
    // capture, and the store is called from inside the epilogue loop lambda).
    // The batch coordinate is the EXPERT id here, not a slot index: the forked
    // slot kernel remaps it before it reaches the epilogue.
    int const fso_M = static_cast<int>(M);
    int const fso_N = static_cast<int>(N);
    int const fso_l = static_cast<int>(cute::get<3>(cta_coord_mnkl));

""" + sync_anchor,
        1, "8 plain locals for the fused store"))

    # --- edit 9: the fused store replaces the EVT visit loop ---------------
    edits.append((
        """          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tTR_rAcc_frg); ++epi_v) {
            tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
          }
""",
        """          // -------------------------------------- fso fused FC1 slot epilogue
          // Precondition: the FC1 weights were laid out with gate and up rows
          // INTERLEAVED, so accumulator ROW 2j of this tile holds gate_j and row
          // 2j+1 holds up_j of the same output element j. In this orientation
          // the rows are the expert's weight rows and the columns are routed
          // tokens, so the problem's M extent is 2*I and the output width is I.
          // The op that launches this kernel documents that it cannot detect the
          // wrong layout.
          //
          // What replaces the visit loop: this thread holds one accumulator row
          // and the whole 64-column token tile of it, so the store pairs the
          // gate/up rows across lane l and l^1, reduces the 32-wide amax with
          // one redux inside each warp and one shared-memory step across the
          // warp pair, and writes the fp8 bytes and the scale bytes with plain
          // global stores. Nothing is stored through ptr_D (edit 10 removes that
          // store).
          fso_swiglu_slot::fused_swiglu_slot_mxfp8_store(tTR_rAcc, tTR_cCD_mn, params.fused,
                                                        fso_M, fso_N, fso_l,
                                                        fso_amax_ptr, thread_idx, synchronize);
""",
        1, "9 fused store replaces the EVT visit loop"))

    # --- edit 10: drop the bf16 D store ------------------------------------
    edits.append((
        """          using VecType = uint_bit_t<VD * sizeof_bits_v<ElementD>>;
          if constexpr (!is_same_v<VecType, uint256_t>) {
            Tensor tTR_gD_frg = recast<VecType>(coalesce(tTR_gD(_,_,_,epi_m,epi_n)));
            Tensor tTR_rD_frg = recast<VecType>(coalesce(tTR_rD));
            Tensor tTR_pD_frg = tensor<1>(zipped_divide(coalesce(tTR_pCD_mn), mclD.compose(Int<VD>{})));
            copy_if(tTR_pD_frg, tTR_rD_frg, tTR_gD_frg);
          }
          else {
            auto tiled_r2g = make_tiled_copy_D(Copy_Atom<SM100_STORE_256bit_CACHE_NOALLOCATION, ElementD>{}, tiled_t2r);
            auto thr_r2g = tiled_r2g.get_slice(threadIdx.x);
            Tensor src = thr_r2g.retile_S(tTR_rD);
            Tensor dst = thr_r2g.retile_D(tTR_gD(_,_,_,epi_m,epi_n));
            Tensor prd = thr_r2g.retile_D(tTR_pCD_mn);
            copy_if(tiled_r2g, prd, src, dst);
          }""",
        """          // fso fused FC1 slot epilogue: the bf16 D store is gone -- the fused
          // store above already wrote both outputs. tTR_gD / tTR_rD are still
          // built further up because they are what CUTLASS sizes the register
          // fragments from, but nothing is written through them and the
          // compiler drops them.
          (void) sizeof(VD);""",
        1, "10 drop the bf16 D store"))

    for old, new, n, what in edits:
        cnt = body.count(old)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}"
        body = body.replace(old, new, n)
        report.append(f"edit {what}: {cnt} site(s)")

    out = """// GENERATED by csrc/gemm/tools/make_sm100_fused_swiglu_slot_epilogue.py -- do
// not edit by hand, and do not check the result in: the build writes it into
// the build directory only.
//
// Clone of the CUTLASS SM100 DENSE NoSmem EVT epilogue with the bf16 D store
// replaced by a fused SwiGLU + MXFP8 1x32 dual store in the SWAP orientation
// (expert weight rows on M, routed tokens on N).
// Source: """ + src_path + """
#pragma once

// `sm100_epilogue_nosmem.hpp` is not self-contained: CUTLASS only ever includes
// it from the collective builder, after the epilogue fusion headers have
// declared `cutlass::epilogue::fusion`. Pulling the builder in first is what
// makes this generated header includable from anywhere.
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp"

#include "blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_slot_store.cuh"

namespace cutlass {
namespace epilogue {
namespace collective {

""" + body + """

} // namespace collective
} // namespace epilogue
} // namespace cutlass
"""
    return out, report, src_path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: make_sm100_fused_swiglu_slot_epilogue.py <output header path>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    text, report, src_path = generate(_cutlass_dir())
    old = None
    if os.path.exists(dst):
        with open(dst) as f:
            old = f.read()
    if old != text:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
        state = "wrote"
    else:
        state = "unchanged"
    print("[fso] sm_100 fused SwiGLU slot epilogue: " + "; ".join(report))
    print("[fso]   from " + src_path)
    print(f"[fso]   {state} " + os.path.abspath(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
