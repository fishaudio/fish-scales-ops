// Vendored from sgl-project/DeepGEMM v0.2.0 (MIT License, Copyright (c) 2025 DeepSeek; see ../LICENSE.DeepGEMM).
// Modified by fish-scales-ops for NVRTC: the includes of layout/sym_buffer.cuh and layout/mega_moe.cuh and the
// multi-GPU nvlink_barrier that needs them are removed; the dense GEMM does not use them.
#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/ptx/ld_st.cuh>

namespace deep_gemm::comm {

// 300s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 300ll * 2000000000ll;

// Spin until `pred()` holds; on timeout, `print_timeout()` runs before the assertion
template <typename pred_t, typename print_timeout_t>
CUTLASS_DEVICE void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const auto start_clock = clock64();
    while (not pred()) {
        if (clock64() - start_clock >= kNumTimeoutCycles) {
            print_timeout();
            DG_DEVICE_ASSERT(false and "Timeout");
        }
    }
}

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0, typename WorkspaceT, typename sync_scope_t>
CUTLASS_DEVICE void grid_sync(const WorkspaceT& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.template get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000) && \
        !(defined(DG_NVLINK_BARRIER_VERBOSE_TIMEOUT) && DG_NVLINK_BARRIER_VERBOSE_TIMEOUT)
        do {
            new_value = ptx::ld_acq(count_ptr);
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
#else
        const auto start_clock = clock64();
        do {
            new_value = ptx::ld_acq(count_ptr);
            if (clock64() - start_clock >= kNumTimeoutCycles) {
                printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                       sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
                DG_DEVICE_ASSERT(false and "Grid sync timeout");
            }
        } while (((new_value ^ old_value) & kFinishSumTag) == 0);
#endif
    }
    sync_scope();
}

} // namespace deep_gemm::comm
