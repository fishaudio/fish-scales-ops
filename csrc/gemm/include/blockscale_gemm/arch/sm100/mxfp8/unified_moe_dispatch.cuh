/*
 * Unified SM100/SM103 128 x 128 x 128 persistent FC1 + FC2 MoE dispatcher (v2).
 *
 * ONE single persistent 128 x 128 x 128 (1-SM, cluster 1x1x1) kernel serves ALL
 * M = 1..8192+, padded CUDA Graph buckets, skewed routing, and optional fused
 * Shared Expert (expert E). Zero host-side `if M > X` / `if expected_m > X` dispatch:
 *   - 8/16/32/64/128 row-bounded TMA activation loading (selected on device)
 *   - Live-guarded epilogue subtile 1 (skips tcgen05.ld + store when rows <= 64)
 *   - Vectorized 2 KiB SMEM transpose -> STG.128 FC2 store when rows > 4
 *   - Block-list persistent scheduler with Shared Expert first, register list cache,
 *     device-gated L2 prefetch, and device-gated 2-panel grouping.
 */

#pragma once

#include "cutlass/arch/config.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "blockscale_gemm/arch/sm100/mxfp8/unified_moe_scheduler.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/unified_moe128_mainloop.cuh"
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_unified_moe128_epilogue.hpp"
#include "blockscale_gemm/arch/sm100/mxfp8/sm100_slot_moe_kernel.hpp"

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/float_subbyte.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace sm100_blockscaled_gemm
{

namespace unified_moe_detail
{

using namespace cute;

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class UnifiedMoe128Fc1Epilogue
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoUnifiedMoeFc1NoSmem128<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoUnifiedMoeFc1NoSmem128<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
class UnifiedMoe128Fc2Epilogue
    : public cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
          cutlass::epilogue::collective::FsoUnifiedMoeFc2NoSmem128<ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
public:
    using cutlass::epilogue::collective::detail::Sm100TmaWarpSpecializedAdapter<
        cutlass::epilogue::collective::FsoUnifiedMoeFc2NoSmem128<ET, EC, SC, ED, SD, FC, CT, AC,
            AD>>::Sm100TmaWarpSpecializedAdapter;
};

template <class T>
struct RebindUnifiedMoe128Epilogues;

template <class ET, class EC, class SC, class ED, class SD, class FC, class CT, class AC, class AD>
struct RebindUnifiedMoe128Epilogues<cutlass::epilogue::collective::CollectiveEpilogue<
    cutlass::epilogue::Sm100NoSmemWarpSpecialized, ET, EC, SC, ED, SD, FC, CT, AC, AD>>
{
    using Fc1 = UnifiedMoe128Fc1Epilogue<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
    using Fc2 = UnifiedMoe128Fc2Epilogue<ET, EC, SC, ED, SD, FC, CT, AC, AD>;
};

struct Sm100MxFP8UnifiedMoeConfig
{
    using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementSF = cutlass::float_ue8m0_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;
    using LayoutATag = cutlass::layout::RowMajor;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    using LayoutDEpiTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int kTileN = 128;

    using MmaTileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    using StockEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, void, LayoutDEpiTag,
        AlignmentD, ElementD, LayoutDEpiTag, AlignmentD, cutlass::epilogue::NoSmemWarpSpecialized1Sm>::CollectiveOp;
    using EpilogueFc1 = typename RebindUnifiedMoe128Epilogues<StockEpilogue>::Fc1;
    using EpilogueFc2 = typename RebindUnifiedMoe128Epilogues<StockEpilogue>::Fc2;

    static constexpr int kEpiCarveout1 = static_cast<int>(sizeof(typename EpilogueFc1::SharedStorage));
    static constexpr int kEpiCarveout2 = static_cast<int>(sizeof(typename EpilogueFc2::SharedStorage));
    static constexpr int kEpilogueCarveout = kEpiCarveout1 > kEpiCarveout2 ? kEpiCarveout1 : kEpiCarveout2;

    using StockMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassBlockScaledTensorOp, ElementAPair, LayoutATag, AlignmentAB, ElementBPair, LayoutBTag,
        AlignmentAB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<kEpilogueCarveout>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
    using CollectiveMainloop = unified_moe128_detail::FsoMoe128Mainloop<StockMainloop>;
    using ProblemShape = Shape<int, int, int, int>;
    using GemmKernel = cutlass::gemm::kernel::FsoSm100SlotGemmMoe<ProblemShape, CollectiveMainloop, EpilogueFc1,
        cutlass::gemm::FsoUnifiedMoeSchedulerTag, EpilogueFc2>;
    using StrideA = typename StockMainloop::StrideA;
    using StrideB = typename StockMainloop::StrideB;
    using StrideD = typename StockEpilogue::StrideD;
    using Sm1xxBlkScaledConfig = typename StockMainloop::Sm1xxBlkScaledConfig;
};

#if defined(FSO_SM100_MOE_GDC)
inline constexpr bool kUnifiedMoeGdcCompiled = true;
#else
inline constexpr bool kUnifiedMoeGdcCompiled = false;
#endif

inline bool unified_moe_pdl_enabled() noexcept
{
    static bool const v = []
    {
        char const* const e = std::getenv("FSO_DISABLE_PDL");
        return !(e != nullptr && e[0] == '1');
    }();
    return v;
}

inline bool unified_moe_prefetch_enabled() noexcept
{
    static bool const v = []
    {
        char const* const e = std::getenv("FSO_UNIFIED_MOE_NO_PREFETCH");
        return !(e != nullptr && e[0] == '1');
    }();
    return v;
}

inline int unified_moe_sm_count() noexcept
{
    static int s_cached = -1;
    if (s_cached < 0)
    {
        int dev = 0;
        cudaGetDevice(&dev);
        s_cached = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
    }
    return s_cached;
}

} // namespace unified_moe_detail

// Note: `slot_tiles` carries the compact v2 `block_list[blocks_cap + 1]`,
// `tiles_cap` = `blocks_cap` (= num_blocks). `chunk_list` / `chunks_cap` kept
// in signature for ABI/op compatibility and unused in v2.
inline cudaError_t gemm_dispatch_sm100_mxfp8_unified_moe(__nv_fp8_e4m3 const* a, int32_t const* sa,
    __nv_fp8_e4m3 const* w13, int32_t const* sw13, __nv_fp8_e4m3 const* w2, int32_t const* sw2, __nv_fp8_e4m3* dq,
    int32_t* sd, __nv_bfloat16* dn, int32_t const* masked_m, int32_t const* blk0, int const* slot_to_expert,
    int* counters, int const* block_list, int blocks_cap, int const* /*chunk_list*/, int /*chunks_cap*/, int groups,
    int b_alloc, int n1, int k1, int n2, int k2, cudaStream_t stream)
{
    using Config = unified_moe_detail::Sm100MxFP8UnifiedMoeConfig;
    using GemmKernel = typename Config::GemmKernel;
    using Args = typename GemmKernel::Arguments;
    using Params = typename GemmKernel::Params;
    using EA = typename Config::ElementA;
    using EB = typename Config::ElementB;
    using ESF = typename Config::ElementSF;
    using ED = typename Config::ElementD;
    using Sm1xxCfg = typename Config::Sm1xxBlkScaledConfig;

    (void) blk0;
    int const inter = n1 / 2;
    if (groups <= 0 || b_alloc <= 0 || blocks_cap <= 0 || (n1 % 128) != 0 || (k1 % 128) != 0 || (n2 % 128) != 0
        || (k2 % 128) != 0 || k2 != inter || (inter % 128) != 0 || counters == nullptr || block_list == nullptr)
        return cudaErrorInvalidValue;

    int const pn = 128;
    int const L = b_alloc > groups ? b_alloc : groups;

    int dev = 0;
    cudaGetDevice(&dev);
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = dev;
    hw_info.sm_count = unified_moe_detail::unified_moe_sm_count();
    hw_info.max_active_clusters = 0;
    hw_info.cluster_shape = dim3(1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    auto stride_a1 = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(n1, k1, L));
    auto stride_a2 = cutlass::make_cute_packed_stride(typename Config::StrideA{}, cute::make_shape(n2, k2, L));
    typename Config::StrideB stride_b1{static_cast<int64_t>(k1), cute::Int<1>{}, static_cast<int64_t>(128) * k1};
    typename Config::StrideB stride_b2{static_cast<int64_t>(k2), cute::Int<1>{}, static_cast<int64_t>(128) * k2};
    auto stride_d1 = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(n1, pn, L));
    auto stride_d2 = cutlass::make_cute_packed_stride(typename Config::StrideD{}, cute::make_shape(n2, pn, L));

    auto layout_sfa1 = Sm1xxCfg::tile_atom_to_shape_SFA(cute::make_shape(n1, pn, k1, groups));
    auto layout_sfa2 = Sm1xxCfg::tile_atom_to_shape_SFA(cute::make_shape(n2, pn, k2, groups));
    auto layout_sfb1 = Sm1xxCfg::tile_atom_to_shape_SFB(cute::make_shape(n1, pn, k1, b_alloc));
    auto layout_sfb2 = Sm1xxCfg::tile_atom_to_shape_SFB(cute::make_shape(n2, pn, k2, b_alloc));

    Args args{cutlass::gemm::GemmUniversalMode::kGemm, {n1, pn, k1, L},
        {reinterpret_cast<EA const*>(w13), stride_a1, reinterpret_cast<EB const*>(a), stride_b1,
            reinterpret_cast<ESF const*>(sw13), layout_sfa1, reinterpret_cast<ESF const*>(sa), layout_sfb1},
        {{}, nullptr, stride_d1, reinterpret_cast<ED*>(dq), stride_d1}, hw_info, {}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.epilogue.fused.ptr_h = dq;
    args.epilogue.fused.ptr_sfh = sd;
    args.epilogue.fused.h_group_elems = static_cast<long long>(128) * inter;
    args.epilogue.fused.sf_group_words = static_cast<long long>(128) * (inter / 128);
    args.problem_shape2 = cute::make_shape(n2, pn, k2, L);
    args.mainloop2 = {reinterpret_cast<EA const*>(w2), stride_a2, reinterpret_cast<EB const*>(dq), stride_b2,
        reinterpret_cast<ESF const*>(sw2), layout_sfa2, reinterpret_cast<ESF const*>(sd), layout_sfb2};
    args.epilogue2 = {{}, nullptr, stride_d2, reinterpret_cast<ED*>(dn), stride_d2};
    args.epilogue2.thread.alpha = 1.0f;
    args.epilogue2.thread.beta = 0.0f;
    args.epilogue2.plain.ptr_d = dn;
    args.epilogue2.plain.col_elems = static_cast<long long>(n2);
    args.epilogue2.plain.batch_elems = static_cast<long long>(128) * n2;
    args.fso_slot_to_expert = slot_to_expert;
    args.fso_masked_m = masked_m;
    args.fso_num_slots = blocks_cap;
    args.fso_num_groups = groups;

    args.scheduler.counter = counters;
    args.scheduler.block_list = block_list;
    args.scheduler.blocks_cap = blocks_cap;
    args.scheduler.tiles_m2 = (n2 + 127) / 128;
    args.scheduler.num_groups = groups;
    bool const prefetch = unified_moe_detail::unified_moe_prefetch_enabled()
        && reinterpret_cast<uintptr_t>(w13) % 16 == 0 && reinterpret_cast<uintptr_t>(sw13) % 16 == 0;
    args.scheduler.first_panel_w = prefetch ? static_cast<void const*>(w13) : nullptr;
    args.scheduler.first_panel_sf = sw13;
    args.scheduler.first_panel_k = k1;

    if (GemmKernel::get_workspace_size(args) != 0)
        return cudaErrorNotSupported;
    if (!GemmKernel::can_implement(args))
        return cudaErrorInvalidValue;
    Params const params = GemmKernel::to_underlying_arguments(args, nullptr);
    dim3 const grid = GemmKernel::get_grid_shape(params);
    dim3 const block = GemmKernel::get_block_shape();

    static bool s_smem_configured = false;
    if (!s_smem_configured)
    {
        if (GemmKernel::SharedStorageSize >= (48 << 10))
        {
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
            cudaError_t const ex = cudaThreadExchangeStreamCaptureMode(&mode);
            if (ex != cudaSuccess)
                return ex;
            cudaError_t const result = cudaFuncSetAttribute(cutlass::device_kernel<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, GemmKernel::SharedStorageSize);
            cudaThreadExchangeStreamCaptureMode(&mode);
            if (result != cudaSuccess)
                return result;
        }
        s_smem_configured = true;
    }

    bool const pdl = unified_moe_detail::unified_moe_pdl_enabled();
    if (unified_moe_detail::kUnifiedMoeGdcCompiled && pdl)
    {
        cudaLaunchConfig_t cfg{};
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = GemmKernel::SharedStorageSize;
        cfg.stream = stream;
        cfg.attrs = attrs;
        cfg.numAttrs = 1;
        cudaError_t const r = cudaLaunchKernelEx(&cfg, cutlass::device_kernel<GemmKernel>, params);
        if (r != cudaSuccess)
            return r;
        return cudaGetLastError();
    }
    cutlass::device_kernel<GemmKernel><<<grid, block, GemmKernel::SharedStorageSize, stream>>>(params);
    return cudaGetLastError();
}

} // namespace sm100_blockscaled_gemm

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
