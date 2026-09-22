#!/usr/bin/env python3
"""Generate the sm_100 fused-SwiGLU epilogue from CUTLASS's pointer-array
NoSmem epilogue.

Origin: the prototype of run b300_mxfp8_20260917/M-E1 (`gen_me1_epilogue.py`),
vendored into the library with fso naming. The sibling generator
`make_sm100_slot_kernel.py` works the same way and is run by the same build
hook; both read a CUTLASS header and write a separate, independently named
class into the build directory. Nothing in `3rdparty/cutlass` is modified, and
the generated header is never written into the source tree.

Which CUTLASS class is cloned, and why that one
----------------------------------------------
`cutlass/epilogue/collective/sm100_epilogue_array_nosmem.hpp` holds TWO
specialisations of the pointer-array NoSmem epilogue, chosen by whether the
fusion operation is the default one:

  * the DEFAULT-fusion specialisation (plain `thread::LinearCombination`) builds
    its TMEM-to-register copy over the WHOLE CTA tile, so one thread holds TileN
    accumulator floats;
  * the EVT specialisation divides the CTA tile by `EpilogueTile`, which the
    NoSmem builder pins to `(TileM, min(64, TileN))`, so one thread holds 64
    consecutive N columns of ONE row per subtile.

Sixty-four consecutive N columns of one row are, once the FC1 weight rows are
gate/up interleaved, exactly thirty-two gate/up pairs, i.e. exactly one 1x32
output scale block. The EVT specialisation is therefore the one whose fragment
shape matches the fusion, and it is the one cloned here. fso's grouped config
reaches it by passing `OpClassBlockScaledTensorOp` to the epilogue builder (see
`Sm100MxFP8GroupedSwiGluGemmConfig` in grouped_gemm_types.cuh); the unfused
grouped config keeps `OpClassTensorOp` and is untouched.

What the clone changes
----------------------
The EVT visit loop and the bf16 D store are replaced by one call to
`fso_swiglu::fused_swiglu_mxfp8_store` (csrc/gemm/include/blockscale_gemm/
arch/sm100/mxfp8/fused_swiglu_store.cuh), which computes silu(gate)*up in FP32,
takes the 32-wide amax, derives the UE8M0 byte and writes the fp8 bytes and the
scale byte. Everything else -- the load of the accumulator out of TMEM, the
subtile walk, the per-group pointer and stride selection, the shared-storage
declaration -- is CUTLASS's own code, unmodified. In particular the clone adds
NO shared memory, so `StageCountAutoCarveout` gives the mainloop the same number
of stages it gives the stock epilogue; grouped_gemm_types.cuh static_asserts
that rather than asserting it in prose.

Why a clone and not an EVT node. CUTLASS 4.4.2 has no gated-activation fusion
node on any architecture (the activations in `epilogue/thread/activation.h` are
single-operand functors plugged into `LinCombEltAct`), and nothing in the tree
halves the N extent in an epilogue. `Sm100BlockScaleFactorRowStore` can write
block-scaled output, but it is handed one epilogue-tile position at a time and
has no access to a second operand, and the gate has to sit BELOW the
scale-factor node so that the scale is computed on post-gate values -- which the
stock EVT tree does not offer. Writing the gate as a proper EVT node is the
cleaner long-term shape and is recorded as open work in the report of run
b300_mxfp8_20260917/M-E2.

Each edit asserts its exact match count, so a CUTLASS bump that moves any of
them fails the build here, naming the edit, rather than producing a silently
different kernel.
"""
from __future__ import annotations

import os
import sys

HEADER_REL = "include/cutlass/epilogue/collective/sm100_epilogue_array_nosmem.hpp"

# The head of the EVT specialisation. Used both as the extraction anchor (edit
# 1) and as the literal edit 2 rewrites.
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
    Sm100PtrArrayNoSmem,
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
class FsoFusedSwiGluPtrArrayNoSmem {"""


def _cutlass_dir() -> str:
    for key in ("CUTLASS_DIR", "BSGEMM_CUTLASS_DIR"):
        cand = os.environ.get(key)
        if cand and os.path.isdir(os.path.join(cand, "include", "cutlass")):
            return cand
    here = os.path.dirname(os.path.abspath(__file__))
    sub = os.path.normpath(os.path.join(here, "..", "..", "..", "3rdparty", "cutlass"))
    if os.path.isdir(os.path.join(sub, "include", "cutlass")):
        return sub
    raise SystemExit("[fso] fused SwiGLU epilogue: CUTLASS not found; set CUTLASS_DIR")


def generate(cutlass_dir: str) -> tuple[str, list[str], str]:
    """Return (generated source, per-edit report lines, source header path)."""
    src_path = os.path.join(cutlass_dir, HEADER_REL)
    text = open(src_path).read()
    report: list[str] = []

    # --- edit 1: extract the EVT specialisation by brace matching -----------
    # Brace matching rather than a regex: the class body contains braces inside
    # string literals only in comments, and a regex over 17 kB of nested
    # templates is not auditable.
    cnt = text.count(EVT_CLASS_HEAD)
    assert cnt == 1, f"[1 extract] found {cnt} copies of the EVT class head, expected 1"
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
    report.append(f"edit 1 extract EVT Sm100PtrArrayNoSmem specialisation: {len(body)} chars")

    edits: list[tuple[str, str, int, str]] = []

    # --- edit 2: partial specialisation -> primary class template -----------
    # fso instantiates the clone explicitly (through `RebindFusedSwiGlu` in
    # grouped_gemm_types.cuh), so it must not compete with CUTLASS's own
    # specialisation of `CollectiveEpilogue`.
    edits.append((EVT_CLASS_HEAD, PRIMARY_CLASS_HEAD, 1, "2 primary-template head"))

    # --- edit 3: constructor name follows the class name --------------------
    edits.append((
        """  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, SharedStorage& shared_tensors)""",
        """  CUTLASS_DEVICE
  FsoFusedSwiGluPtrArrayNoSmem(Params const& params_, SharedStorage& shared_tensors)""",
        1, "3 constructor name"))

    # --- edits 4/5/6: carry the fused destinations through Arguments/Params --
    edits.append((
        """  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC = {};
    ElementD** ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC = {};
    ElementD** ptr_D = nullptr;
    StrideD dD = {};
    // ------------------------------------------------- fso fused FC1 epilogue
    // The fused destinations are plain base pointers plus per-group element
    // counts, not the per-group pointer arrays CUTLASS uses for D, because the
    // MoE slabs are contiguous in the group index. See FusedSwiGluArgs in
    // blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_store.cuh.
    fso_swiglu::FusedSwiGluArgs fused{};
  };""",
        1, "4 fused Arguments field"))

    edits.append((
        """  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC = {};
    ElementD** ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """  // Device side epilogue params
  struct Params {
    typename FusionCallbacks::Params thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC = {};
    ElementD** ptr_D = nullptr;
    StrideD dD = {};
    fso_swiglu::FusedSwiGluArgs fused{};   // fso fused FC1 epilogue
  };""",
        1, "5 fused Params field"))

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
      args.fused          // fso fused FC1 epilogue
    };""",
        1, "6 to_underlying_arguments carries the fused args"))

    # --- edit 7: ordinary locals for the values the store needs -------------
    # M, N and l_coord arrive as STRUCTURED BINDINGS, and C++17 forbids
    # capturing a structured binding in a lambda -- `process_tile`, which
    # contains the store, is a lambda. Binding them to plain locals first is
    # what makes the store callable from inside it.
    sync_anchor = (
        "    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE "
        "{ cutlass::arch::NamedBarrier::sync(ThreadCount, "
        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };")
    edits.append((
        sync_anchor,
        """    // fso fused FC1 epilogue: ordinary locals for the fused store (M, N and
    // l_coord are structured bindings, which C++17 does not allow a lambda to
    // capture, and the store is called from inside the `process_tile` lambda).
    int const fso_M = static_cast<int>(M);
    int const fso_N = static_cast<int>(N);
    int const fso_l = static_cast<int>(l_coord);

""" + sync_anchor,
        1, "7 plain locals for the fused store"))

    # --- edit 8: the fused store replaces the EVT visit loop ----------------
    edits.append((
        """        CUTLASS_PRAGMA_UNROLL
        for (int epi_v = 0; epi_v < size(tTR_rAcc_frg); ++epi_v) {
          tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
        }
""",
        """        // ------------------------------------------------ fso fused FC1 epilogue
        // Precondition: the FC1 weights were laid out with gate and up rows
        // INTERLEAVED, so accumulator column 2j of this tile holds gate_j and
        // column 2j+1 holds up_j of the SAME output element j; the problem's N
        // extent is 2*I and the output extent is I. The op that launches this
        // kernel documents that it cannot detect the wrong layout.
        //
        // What replaces the visit loop: this thread already holds 64 consecutive
        // FP32 accumulator columns of one row, so the pairing, the SwiGLU, the
        // 32-element amax, the UE8M0 byte and the e4m3 conversion are all
        // register-local, and the two outputs are written with plain global
        // stores. Nothing is stored through ptr_D (edit 9 removes that store).
        fso_swiglu::fused_swiglu_mxfp8_store(tTR_rAcc, tTR_cCD_mn, params.fused,
                                             fso_M, fso_N, fso_l);
""",
        1, "8 fused store replaces the EVT visit loop"))

    # --- edit 9: drop the bf16 D store --------------------------------------
    edits.append((
        """        using VecType = uint_bit_t<VD * sizeof_bits_v<ElementD>>;
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
        """        // fso fused FC1 epilogue: the bf16 D store is gone -- the fused store
        // above already wrote both outputs. tTR_gD / tTR_rD are still built
        // further up because they size the register accumulator, but nothing is
        // written through them.
        (void) sizeof(VD);""",
        1, "9 drop the bf16 D store"))

    for old, new, n, what in edits:
        cnt = body.count(old)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}"
        body = body.replace(old, new, n)
        report.append(f"edit {what}: {cnt} site(s)")

    out = """// GENERATED by csrc/gemm/tools/make_sm100_fused_swiglu_epilogue.py -- do not
// edit by hand, and do not check the result in: the build writes it into the
// build directory only.
//
// Clone of the CUTLASS SM100 pointer-array NoSmem EVT epilogue with the bf16 D
// store replaced by a fused SwiGLU + MXFP8 1x32 dual store.
// Source: """ + src_path + """
#pragma once

#include "cutlass/epilogue/collective/sm100_epilogue_array_nosmem.hpp"

#include "blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_store.cuh"

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
        print("usage: make_sm100_fused_swiglu_epilogue.py <output header path>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    text, report, src_path = generate(_cutlass_dir())
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.write(text)
    print("[fso] sm_100 fused SwiGLU epilogue: " + "; ".join(report))
    print("[fso]   from " + src_path)
    print("[fso]   wrote " + os.path.abspath(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
