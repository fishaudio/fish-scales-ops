#!/usr/bin/env python3
"""Generate SM100 128 x 128 x 128 unified MoE FC1 & FC2 NoSmem epilogues from CUTLASS (v3).

v3: FC2 uses direct zero-SMEM, zero-barrier coalesced 64B-warp BF16 stores
    (eliminating 32 barriers/tile and cutting register pressure).
"""
from __future__ import annotations

import os
import sys

EPI_REL = "include/cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp"

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


def _cutlass_dir() -> str:
    for var in ("CUTLASS_DIR", "BSGEMM_CUTLASS_DIR"):
        v = os.environ.get(var)
        if v:
            return v
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "3rdparty", "cutlass"))


def _lit(s: str, old: str, new: str, n: int, what: str) -> str:
    cnt = s.count(old)
    assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}: {old[:80]!r}"
    return s.replace(old, new, n)


def _extract_evt_class(src_path: str) -> str:
    text = open(src_path).read()
    assert text.count(EVT_CLASS_HEAD) == 1, "EVT Sm100NoSmem class head not unique"
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
    assert text[i] == ";", "EVT class body does not end with '};'"
    return text[start:i + 1]


def _make_fc1(original: str) -> str:
    s = original
    s = _lit(
        s,
        EVT_CLASS_HEAD,
        """template <
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
class FsoUnifiedMoeFc1NoSmem128 {""",
        1,
        "fc1 head",
    )
    s = _lit(
        s,
        "    array_aligned<uint8_t, ImplicitSharedStorageSize> buffer;\n  };",
        "    array_aligned<uint8_t, ImplicitSharedStorageSize> buffer;\n"
        "    alignas(16) float fso_amax[fso_swiglu_slot::kSlotAmaxFloats];\n  };",
        1,
        "fc1 SharedStorage",
    )
    s = _lit(
        s,
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
  FsoUnifiedMoeFc1NoSmem128(Params const& params_, SharedStorage& shared_tensors)
  : fusion_callbacks(params_.thread, shared_tensors.thread)
  , smem_buffer_ptr(shared_tensors.buffer.data())
  , fso_amax_ptr(shared_tensors.fso_amax)
  , params(params_) {};

protected:
  FusionCallbacks fusion_callbacks;
  uint8_t* smem_buffer_ptr;
  float* fso_amax_ptr;
  Params const& params;""",
        1,
        "fc1 ctor",
    )
    s = _lit(
        s,
        """    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """    ElementD* ptr_D = nullptr;
    StrideD dD = {};
    fso_swiglu_slot::FusedSwiGluSlotArgs fused{};
  };""",
        2,
        "fc1 Arguments/Params",
    )
    s = _lit(
        s,
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
      args.fused
    };""",
        1,
        "fc1 to_underlying_arguments",
    )
    sync_anchor = (
        "    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE "
        "{ cutlass::arch::NamedBarrier::sync(ThreadCount, "
        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };"
    )
    s = _lit(
        s,
        sync_anchor,
        """    int const fso_M = static_cast<int>(M);
    int const fso_N = static_cast<int>(N);
    int const fso_l = static_cast<int>(cute::get<3>(cta_coord_mnkl));

""" + sync_anchor,
        1,
        "fc1 locals",
    )
    s = _lit(
        s,
        """          // Copy accumulator tile from tmem to register
          // The current tile in tmem
          Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);

          Tensor tTR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));

          copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);

          // After the last tmem load, signal that tmem buffer is consumed and empty
          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tTR_rAcc_frg); ++epi_v) {
            tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
          }""",
        """          bool const fso_sub_live = (epi_n * 64 < fso_N);
          if (fso_sub_live) {
            Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);
            copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);
          }

          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          if (fso_sub_live) {
            fso_swiglu_slot::fused_swiglu_slot_mxfp8_store_bounded(
              tTR_rAcc, tTR_cCD_mn, params.fused, fso_M, fso_N, fso_l,
              fso_amax_ptr, thread_idx, synchronize);
          }""",
        1,
        "fc1 live-guarded t2r + swiglu store",
    )
    s = _lit(
        s,
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
        "          (void) sizeof(VD);",
        1,
        "fc1 drop stock D store",
    )
    return s


def _make_fc2(original: str) -> str:
    s = original
    s = _lit(
        s,
        EVT_CLASS_HEAD,
        """template <
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
class FsoUnifiedMoeFc2NoSmem128 {""",
        1,
        "fc2 head",
    )
    s = _lit(
        s,
        """  CUTLASS_DEVICE
  CollectiveEpilogue(Params const& params_, SharedStorage& shared_tensors)""",
        """  CUTLASS_DEVICE
  FsoUnifiedMoeFc2NoSmem128(Params const& params_, SharedStorage& shared_tensors)""",
        1,
        "fc2 ctor",
    )
    s = _lit(
        s,
        """    ElementD* ptr_D = nullptr;
    StrideD dD = {};
  };""",
        """    ElementD* ptr_D = nullptr;
    StrideD dD = {};
    fso_plain_slot::PlainSlotArgs plain{};
  };""",
        2,
        "fc2 Arguments/Params",
    )
    s = _lit(
        s,
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
      args.plain
    };""",
        1,
        "fc2 to_underlying_arguments",
    )
    sync_anchor = (
        "    auto synchronize = [] () CUTLASS_LAMBDA_FUNC_INLINE "
        "{ cutlass::arch::NamedBarrier::sync(ThreadCount, "
        "cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };"
    )
    s = _lit(
        s,
        sync_anchor,
        """    int const fso_M = static_cast<int>(M);
    int const fso_N = static_cast<int>(N);
    int const fso_l = static_cast<int>(cute::get<3>(cta_coord_mnkl));

""" + sync_anchor,
        1,
        "fc2 locals",
    )
    s = _lit(
        s,
        """          // Copy accumulator tile from tmem to register
          // The current tile in tmem
          Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);

          Tensor tTR_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(coalesce(tTR_rAcc));

          copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);

          // After the last tmem load, signal that tmem buffer is consumed and empty
          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tTR_rAcc_frg); ++epi_v) {
            tTR_rD_frg(epi_v) = cst_callbacks.visit(tTR_rAcc_frg(epi_v), epi_v, epi_m, epi_n);
          }""",
        """          bool const fso_sub_live = (epi_n * 64 < fso_N);
          if (fso_sub_live) {
            Tensor tTR_tAcc_mn = tTR_tAcc(_,_,_,epi_m,epi_n);
            copy(tiled_t2r, tTR_tAcc_mn, tTR_rAcc);
          }

          if (do_acc_release) {
            cutlass::arch::fence_view_async_tmem_load();
            acc_pipeline.consumer_release(acc_pipe_consumer_state);
            ++acc_pipe_consumer_state;
          }

          if (fso_sub_live) {
            fso_unified_moe128::moe128_fc2_bf16_store(
              tTR_rAcc, tTR_cCD_mn, params.plain, fso_M, fso_N, fso_l);
          }""",
        1,
        "fc2 live-guarded t2r + zero-barrier bf16 store",
    )
    s = _lit(
        s,
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
        "          (void) sizeof(VD);",
        1,
        "fc2 drop stock D store",
    )
    return s


def generate(cutlass_dir: str) -> tuple[str, str]:
    src_path = os.path.join(cutlass_dir, EPI_REL)
    original = _extract_evt_class(src_path)
    fc1 = _make_fc1(original)
    fc2 = _make_fc2(original)
    out = (
        "// Generated by make_sm100_unified_moe128_epilogue.py (v3)\n"
        "#pragma once\n\n"
        "#include \"cutlass/epilogue/collective/collective_builder.hpp\"\n"
        "#include \"cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp\"\n"
        "#include \"blockscale_gemm/arch/sm100/mxfp8/fused_swiglu_slot_store.cuh\"\n"
        "#include \"blockscale_gemm/arch/sm100/mxfp8/unified_moe128_store.cuh\"\n\n"
        "namespace cutlass::epilogue::collective {\n\n"
        + fc1
        + "\n\n"
        + fc2
        + "\n\n} // namespace cutlass::epilogue::collective\n"
    )
    return out, src_path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: make_sm100_unified_moe128_epilogue.py <dst>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    text, src_path = generate(_cutlass_dir())
    old = open(dst).read() if os.path.exists(dst) else None
    if old != text:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
        state = "wrote"
    else:
        state = "unchanged"
    print(f"[fso] unified moe128 epilogue v3: {state} {dst} ({len(text)} bytes) from {src_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
