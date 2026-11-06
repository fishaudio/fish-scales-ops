/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <nvrtc.h>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

#include "fso_dense_splitk_layout.cuh"
#include "scheduler.cuh"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

// Helper function to check NVRTC errors. The call goes through the NVRTC function table (deep_gemm::jit::nvrtc(),
// below), and a failure throws a RuntimeError instead of ending the process.
#define CHECK_NVRTC(call)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        nvrtcResult result = call;                                                                                     \
        if (result != NVRTC_SUCCESS)                                                                                   \
        {                                                                                                              \
            TLLM_THROW("NVRTC error in %s: %s", #call, ::deep_gemm::jit::nvrtc().nvrtcGetErrorString(result));        \
        }                                                                                                              \
    } while (0)

// Helper function to check CUDA driver errors
#define CHECK_CUDA(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult result = call;                                                                                        \
        if (result != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            const char* error_string;                                                                                  \
            cuGetErrorString(result, &error_string);                                                                   \
            std::cerr << "CUDA error: " << error_string << std::endl;                                                  \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

namespace deep_gemm::jit
{

// The NVRTC library the sm_90 JIT compiles with. The extension does not link libnvrtc: it loads one library itself,
// privately (dlopen with RTLD_NOW | RTLD_LOCAL), and calls NVRTC only through this table of the nine functions the JIT
// uses, resolved with dlsym from that library. No call can therefore bind to the libnvrtc.so.13 that torch loads into
// the process, whose version depends on the torch build. <nvrtc.h> supplies the types and enums only: decltype does not
// reference the functions it names.
struct NvrtcApi
{
    decltype(&::nvrtcVersion) nvrtcVersion = nullptr;
    decltype(&::nvrtcCreateProgram) nvrtcCreateProgram = nullptr;
    decltype(&::nvrtcCompileProgram) nvrtcCompileProgram = nullptr;
    decltype(&::nvrtcGetProgramLogSize) nvrtcGetProgramLogSize = nullptr;
    decltype(&::nvrtcGetProgramLog) nvrtcGetProgramLog = nullptr;
    decltype(&::nvrtcDestroyProgram) nvrtcDestroyProgram = nullptr;
    decltype(&::nvrtcGetCUBINSize) nvrtcGetCUBINSize = nullptr;
    decltype(&::nvrtcGetCUBIN) nvrtcGetCUBIN = nullptr;
    decltype(&::nvrtcGetErrorString) nvrtcGetErrorString = nullptr;
    std::string path; // the absolute path of the library the table was resolved from
    int major = 0;    // its nvrtcVersion
    int minor = 0;
};

// The table of this process, loaded on first use and kept for the life of the process (defined in compiler.cuh, next
// to the knobs it reads).
inline NvrtcApi const& nvrtc();

// What to do when the NVRTC library cannot be loaded; part of every loader error.
inline constexpr char const* kNvrtcRemedy
    = "Run `python scripts/vendor_nvrtc.py` in the fish-scales-ops source tree (scripts/build.sh does it for an sm_90 "
      "build), or set FSO_JIT_NVRTC_LIB to a CUDA 13.0 libnvrtc.so.13.";

// The builtins library that ships next to the bundled libnvrtc.so.13 (scripts/vendor_nvrtc.py unpacks both).
inline constexpr char const* kBundledNvrtcBuiltins = "libnvrtc-builtins.so.13.0";

// A function with internal linkage, so its address always lies in the shared object this translation unit is linked
// into (the extension); dladdr on it names that file.
static void extensionLocatorAnchor() {}

// The directory of the loaded extension .so, from dladdr on a function of its own; an empty path when dladdr cannot
// name the file. The files the package ships next to the extension (the bundled NVRTC, the packaged JIT include tree)
// are found from here rather than from a path baked in at build time, so they move with the package.
inline std::filesystem::path extensionDirectory()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void const*>(&extensionLocatorAnchor), &info) == 0 || info.dli_fname == nullptr
        || info.dli_fname[0] == '\0')
        return {};
    return std::filesystem::path(info.dli_fname).parent_path();
}

// Where the bundled NVRTC lives: <directory of the extension .so>/_nvrtc/libnvrtc.so.13, which scripts/vendor_nvrtc.py
// fills.
inline std::filesystem::path bundledNvrtcPath()
{
    std::filesystem::path const dir = extensionDirectory();
    if (dir.empty())
    {
        TLLM_THROW("sm_90 JIT: cannot locate the fish_scales_ops extension (dladdr failed), so the NVRTC bundled next "
                   "to it cannot be found. %s",
            kNvrtcRemedy);
    }
    return dir / "_nvrtc" / "libnvrtc.so.13";
}

// Where the packaged JIT include tree lives: <directory of the extension .so>/_jit_include. A wheel built by
// scripts/build_wheel.sh carries every header the JIT compiles with there, in one include root (deep_gemm/, CUTLASS
// and the CUDA headers); an in-place build of the source tree has no such directory. Empty when the extension cannot
// be located.
inline std::filesystem::path bundledJitIncludePath()
{
    std::filesystem::path const dir = extensionDirectory();
    return dir.empty() ? dir : dir / "_jit_include";
}

// Loads the NVRTC library and resolves the table; nvrtc() calls it once per process, with debugLog = FSO_JIT_DEBUG.
// The library is, in this order: the file FSO_JIT_NVRTC_LIB names, when that is set and non-empty; the copy bundled
// next to the extension. Nothing else is tried. In particular a failure never falls back to the libnvrtc.so.13 already
// in the process or to $CUDA_HOME: it throws a RuntimeError with the path, the dlerror() text and the remedy. A version
// other than 13.0 is the caller's explicit choice (an override, or a replaced bundled file): it is used, with one line
// on stderr.
inline NvrtcApi loadNvrtcApi(bool debugLog)
{
    char const* envPath = std::getenv("FSO_JIT_NVRTC_LIB");
    bool const fromEnv = envPath != nullptr && envPath[0] != '\0';
    char const* const origin = fromEnv ? "set by FSO_JIT_NVRTC_LIB" : "the copy bundled with fish_scales_ops";
    std::filesystem::path path = fromEnv ? std::filesystem::path(envPath) : bundledNvrtcPath();
    // An absolute path, so that dlopen opens exactly this file. A bare name such as "libnvrtc.so.13" would make it
    // search the library path, and it would return the copy torch already loaded under that soname.
    std::error_code ec;
    std::filesystem::path const absolute = std::filesystem::absolute(path, ec);
    if (!ec)
        path = absolute;

    NvrtcApi api;
    api.path = path.string();
    if (!fromEnv)
    {
        // NVIDIA's libnvrtc.so.13 13.0.88 has no RUNPATH: it opens its builtins library by soname, and an object with
        // that soname already in the process satisfies the lookup before any directory is searched. Loading the
        // bundled builtins by absolute path first makes that lookup find a 13.0.88 copy: torch 2.13's own when torch
        // has already loaded it (torch preloads its NVRTC at import; same version), else this one. A builtins library
        // of another toolkit on the library path is never used.
        std::filesystem::path const builtins = path.parent_path() / kBundledNvrtcBuiltins;
        dlerror();
        if (dlopen(builtins.string().c_str(), RTLD_NOW | RTLD_LOCAL) == nullptr)
        {
            char const* err = dlerror();
            TLLM_THROW("sm_90 JIT: cannot load the bundled NVRTC builtins library %s: %s. %s", builtins.string().c_str(),
                err != nullptr ? err : "unknown error", kNvrtcRemedy);
        }
    }
    dlerror();
    void* handle = dlopen(api.path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
    {
        char const* err = dlerror();
        TLLM_THROW("sm_90 JIT: cannot load the NVRTC library %s (%s): %s. %s", api.path.c_str(), origin,
            err != nullptr ? err : "unknown error", kNvrtcRemedy);
    }
    auto resolve = [&](char const* name, auto& fn)
    {
        dlerror();
        void* sym = dlsym(handle, name);
        if (sym == nullptr)
        {
            char const* err = dlerror();
            std::string const reason = err != nullptr ? err : "the symbol is null";
            dlclose(handle);
            TLLM_THROW("sm_90 JIT: the NVRTC library %s (%s) has no %s: %s. %s", api.path.c_str(), origin, name,
                reason.c_str(), kNvrtcRemedy);
        }
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(sym);
    };
    resolve("nvrtcVersion", api.nvrtcVersion);
    resolve("nvrtcCreateProgram", api.nvrtcCreateProgram);
    resolve("nvrtcCompileProgram", api.nvrtcCompileProgram);
    resolve("nvrtcGetProgramLogSize", api.nvrtcGetProgramLogSize);
    resolve("nvrtcGetProgramLog", api.nvrtcGetProgramLog);
    resolve("nvrtcDestroyProgram", api.nvrtcDestroyProgram);
    resolve("nvrtcGetCUBINSize", api.nvrtcGetCUBINSize);
    resolve("nvrtcGetCUBIN", api.nvrtcGetCUBIN);
    resolve("nvrtcGetErrorString", api.nvrtcGetErrorString);

    // Not CHECK_NVRTC: that goes through nvrtc(), whose initialisation is this call.
    nvrtcResult const versionResult = api.nvrtcVersion(&api.major, &api.minor);
    if (versionResult != NVRTC_SUCCESS)
        TLLM_THROW("sm_90 JIT: nvrtcVersion failed in the NVRTC library %s (%s): %s", api.path.c_str(), origin,
            api.nvrtcGetErrorString(versionResult));
    if (api.major != 13 || api.minor != 0)
        std::fprintf(stderr,
            "[fish_scales_ops] sm_90 JIT: compiling with NVRTC %d.%d from %s (%s); the sm_90 kernels are validated "
            "and measured with NVRTC 13.0\n",
            api.major, api.minor, api.path.c_str(), origin);
    if (debugLog)
        TLLM_LOG_INFO("sm_90 JIT compiler: NVRTC %d.%d from %s (%s)", api.major, api.minor, api.path.c_str(), origin);
    return api;
}

using GemmConfig
    = std::tuple<int, int, int, int, int>; // block_m, block_n, num_stages, num_tma_multicast, best_smem_size

std::string gemm_type_to_string(deep_gemm::GemmType gemm_type);

int div_up(int a, int b);
int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k, bool swap_ab);
bool is_tma_multicast_legal(int n, int block_n, int num_tma_multicast, int num_sms);
GemmConfig get_best_gemm_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_groups,
    int num_device_sms, bool is_grouped_contiguous, bool swap_ab);
} // namespace deep_gemm::jit

namespace deep_gemm::jit
{

std::string gemm_type_to_string(deep_gemm::GemmType gemm_type)
{
    switch (gemm_type)
    {
    case deep_gemm::GemmType::Normal: return std::string("Normal");
    case deep_gemm::GemmType::GroupedContiguous: return std::string("GroupedContiguous");
    case deep_gemm::GemmType::GroupedMasked: return std::string("GroupedMasked");
    case deep_gemm::GemmType::GroupedWithOffset: return std::string("GroupedWithOffset");
    case deep_gemm::GemmType::StridedBatched: return std::string("StridedBatched");
    // Add other GEMM types as needed
    default: return std::string("Unknown");
    }
}

int div_up(int a, int b)
{
    return (a + b - 1) / b;
}

int get_smem_size(int num_stages, int k, int block_m, int block_n, int block_k = 128, bool swap_ab = false)
{
    if (!swap_ab)
    {
        int smem_d = block_m * block_n * 2;
        int smem_a_per_stage = block_m * block_k;
        int smem_scales_a_per_stage = block_m * 4;
        int smem_b_per_stage = block_n * block_k;
        int smem_scales_b = div_up(k, block_k) * 4;
        int smem_barrier = num_stages * 8 * 2;

        int smem_size = 0;
        smem_size += smem_d;
        smem_size += num_stages * smem_a_per_stage;
        smem_size += num_stages * smem_scales_a_per_stage;
        smem_size += num_stages * smem_b_per_stage;
        smem_size += div_up(smem_scales_b * (block_k % block_n == 0 ? 1 : 2), 8) * 8;
        smem_size += smem_barrier;

        return smem_size;
    }
    else
    {
        int smem_d = block_n * block_m * 2;
        int smem_a_per_stage = block_m * block_k;             // weight
        int smem_scales_a_per_stage = div_up(k, block_k) * 4; // weight scales
        int smem_b_per_stage = block_n * block_k;             // act
        int smem_scales_b = div_up(block_n * 4, 128) * 128;   // act scales,tma 128B alignment
        int smem_barrier = num_stages * 8 * 2;

        int smem_size = 0;
        smem_size += smem_d;
        smem_size += num_stages * smem_a_per_stage;
        smem_size += num_stages * smem_scales_b;
        smem_size += num_stages * smem_b_per_stage;
        smem_size += div_up(smem_scales_a_per_stage, 8) * 8;
        smem_size += smem_barrier;

        return smem_size;
    }
}

bool is_tma_multicast_legal(int n, int block_n, int num_tma_multicast, int num_sms)
{
    if (num_tma_multicast == 1)
    {
        return true;
    }
    return (n % (block_n * num_tma_multicast) == 0) && num_sms % num_tma_multicast == 0;
}

GemmConfig get_best_gemm_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_groups,
    int num_device_sms, bool is_grouped_contiguous = false, bool swap_ab = false)
{
    // Choose candidate block sizes
    std::vector<int> block_ms;
    block_ms.push_back((!is_grouped_contiguous && shape_m <= 64) ? 64 : 128);

    // Candidate block sizes for N dimension
    std::vector<int> block_ns;
    for (int i = 16; i <= 128; i += 8)
    {
        block_ns.push_back(i);
    }

    // Lambda functions for calculating waves and utilization
    auto fix_wave_saturate = [num_device_sms](int x) -> int { return x == 0 ? num_device_sms : x; };

    auto get_num_waves = [shape_m, shape_n, num_groups, num_device_sms](int block_m, int block_n) -> int
    { return div_up(div_up(shape_m, block_m) * div_up(shape_n, block_n) * num_groups, num_device_sms); };

    auto get_last_wave_util
        = [shape_m, shape_n, num_groups, num_device_sms, &fix_wave_saturate](int block_m, int block_n) -> int
    { return fix_wave_saturate((div_up(shape_m, block_m) * div_up(shape_n, block_n) * num_groups) % num_device_sms); };

    // Find best block sizes
    int best_block_m = 0;
    int best_block_n = 0;
    for (int block_m : block_ms)
    {
        for (int block_n : block_ns)
        {
            bool success = false;
            int num_waves = get_num_waves(block_m, block_n);
            int best_num_waves = best_block_m == 0 ? INT_MAX : get_num_waves(best_block_m, best_block_n);

            if (best_block_m == 0 || best_block_n == 0)
            {
                success = true;
            }
            else if (num_waves < best_num_waves)
            {
                success = true;
            }
            else if (num_waves == best_num_waves)
            {
                // Check last wave utilization
                int util = get_last_wave_util(block_m, block_n);
                int best_util = get_last_wave_util(best_block_m, best_block_n);
                // FSO (2026-06): in the high-wave regime the last-wave makes up
                // a negligible fraction of total work, so the upstream
                // util tie-break wrongly prefers an awkward non-128 block_n
                // (e.g. 120/112) whose WGMMA-N tiling is less efficient and
                // which forbids num_stages >= 7 (128 % block_n != 0). Prefer a
                // clean block_n (128 % block_n == 0) when waves >= 8, or
                // waves >= 3 with large K (the per-CTA efficiency win then
                // dominates the small fill difference). Validated on the
                // Qwen3-4B prefill shapes: gate_up M=1024 +4.5%, gate M=2048
                // +3.8%, down M=2048 +3.3%; no regression on the few-wave
                // fill-bound shapes (down/wo M<=1024 keep the smaller bn).
                bool const prefer_clean
                    = !swap_ab && (num_waves >= 8 || (num_waves >= 3 && shape_k >= 8192u));
                bool const cand_clean = (128 % block_n == 0);
                bool const best_clean = (128 % best_block_n == 0);
                if (prefer_clean && cand_clean != best_clean)
                {
                    success = cand_clean;
                }
                else
                {
                    success = util > best_util
                        || (util == best_util
                            && (block_m > best_block_m || (block_m == best_block_m && block_n < best_block_n)));
                }
            }

            if (success)
            {
                best_block_m = block_m;
                best_block_n = block_n;
            }
        }
    }

    // Find best number of stages
    int best_num_stages = 0;
    int best_smem_size = 0;
    constexpr int sm90_capacity = 232448;

    std::vector<int> stage_candidates;
    if (128 % best_block_n != 0)
    {
        stage_candidates = {6, 5, 4};
    }
    else
    {
        stage_candidates = {8, 7, 6, 5, 4};
    }

    for (int num_stages : stage_candidates)
    {
        int smem_size = get_smem_size(num_stages, shape_k, best_block_m, best_block_n, 128, swap_ab);
        if (smem_size <= sm90_capacity)
        {
            best_num_stages = num_stages;
            best_smem_size = smem_size;
            break;
        }
    }

    // Determine TMA multicast settings
    int best_num_tma_multicast = 1;

    if (!swap_ab)
    {
        if (shape_m >= 1024 && is_tma_multicast_legal(shape_n, best_block_n, 2, num_device_sms) && num_groups == 1)
        {
            best_num_tma_multicast = 2;
        }
    }
    else
    {
        if (shape_n >= 1024 && is_tma_multicast_legal(shape_m, best_block_m, 2, num_device_sms) && num_groups == 1)
        {
            best_num_tma_multicast = 2;
        }
    }

    return std::make_tuple(best_block_m, best_block_n, best_num_stages, best_num_tma_multicast, best_smem_size);
}

// Tile choice of the dense GEMM (gemm_dispatch_sm90: GemmType::Normal, one problem). It starts from
// get_best_gemm_config's pick, which every grouped (MoE) path keeps using unchanged, and applies the four rules below,
// measured on the H200 (132 SMs) against the previous pick (rules 1-3 on 2026-10-05, rule 4 on 2026-10-11), with
// outputs bitwise identical for every tile (each output element is accumulated by one CTA in the same order whatever
// the tile). The dispatcher may still hand the GEMM to DeepGEMM v2's kernel instead (use_dg2_dense, below):
//
// 1. Main path. A 128-row tile 104, 112 or 120 wide straddles the 128-row weight-scale blocks, so every promotion
//    step selects between two weight scales per accumulator group, and it keeps the store whose completion the CTA
//    waits for before its next tile; a 128-wide tile has one weight scale per k-block and fp8_gemm_kernel's deferred
//    swizzled store (block_n a multiple of 64). When the pick is such a 104- to 120-wide tile and a 128-wide tile
//    needs the same number of waves, the 128-wide tile is taken: it does at most 128 / 104 = 1.23 times the MMA work
//    of the picked tile per wave, and it took 0.85-0.96 of the 0.2.0 release's time on the nine cells of the dense
//    family tables and the cubic sweep where the rule applies (M = 256-3072). Narrower picks (block_n <= 96, 1.33
//    times the work or more) keep their tile; 128 measured 1.01-1.25 there.
// 2. Swap-AB path (small M). The weight is the swap kernel's A operand, tiled by block_m rows, and at these M every
//    CTA streams a block_m x K weight slab, so the number of CTAs with work sets the achieved weight bandwidth. When
//    128-row weight tiles give work to at most a quarter of the SMs (ceil(N / 128) * 4 <= SMs, i.e. N <= 4224 on
//    132 SMs), 64-row tiles are taken: twice the CTAs, still at most half the SMs per activation tile (0.82-0.91 of
//    the 0.2.0 release's time on all 30 family cells where the rule applies, N = 1024-2560 at M <= 32). Wider
//    weights keep 128 rows here: 64-row tiles measured mixed at N = 5120-6144 and 3-34 % slower from N = 9216, where
//    they exceed one wave. The swap-AB route itself takes 64-row tiles at 8 <= M <= 32 up to N = 8448 since FH1 of the
//    issue-12 round (get_dense_swap_ab_config); this pick stays the weight tile the band rules below reason about.
//
// 3. Main path, M <= 128 with a narrow pick. A 128-row tile covers all of M in one M-block, and when the picker
//    then chose block_n <= 32 (the narrow outputs, N up to about 4096) every CTA streams 128 rows of A for at most
//    32 rows of B and runs WGMMAs at most 32 wide. Two 64-row tiles of twice the width keep the CTA count and the
//    output area per CTA, cut the rows each CTA loads per k-block from 128 + block_n to 64 + 2 x block_n and double
//    the WGMMA width (0.95-0.97 of the time with the 128-row tile on the five family cells where the rule applies,
//    M = 128; multicast stays off at these M).
// 4. Main path, 64-row tiles with a 24-wide pick (M <= 64 on the narrow outputs, N about 2100-3200 on 132 SMs). A
//    24-wide tile straddles the 128-row weight-scale blocks and caps the pipeline at 6 stages; when a 32-wide tile
//    needs the same number of waves it is taken: one weight scale per k-block, up to 8 stages, and a quarter fewer
//    CTAs re-reading the same activation rows (0.93-0.95 of the time with the 24-wide tile on Family A `wo` and
//    `down` at M = 64, P2 of the 0.2.1 round; `down`, K = 9728, now runs DeepGEMM v2's 64 x 32 tile instead, see
//    use_dg2_dense).
//
// The stage count is recomputed for the new tile with get_best_gemm_config's rule (the deepest of 8..4, or 6..4
// when block_n does not divide 128, that fits the shared memory) and so is the multicast of rule 1 (2 when
// M >= 1024 and N splits into pairs of 128-wide tiles). `swap_ab` selects the shape convention of the swap-AB call:
// (shape_m, shape_n) are then (N, M), as gemm_dispatch_sm90 passes them to get_best_gemm_config.
inline GemmConfig get_dense_gemm_config(
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_device_sms, bool swap_ab)
{
    auto [block_m, block_n, num_stages, num_tma_multicast, smem_size]
        = get_best_gemm_config(shape_m, shape_n, shape_k, 1, num_device_sms, false, swap_ab);

    auto restage = [&]()
    {
        constexpr int sm90_capacity = 232448;
        std::vector<int> const stage_candidates
            = (128 % block_n != 0) ? std::vector<int>{6, 5, 4} : std::vector<int>{8, 7, 6, 5, 4};
        for (int stages : stage_candidates)
        {
            int const smem = get_smem_size(stages, shape_k, block_m, block_n, 128, swap_ab);
            if (smem <= sm90_capacity)
            {
                num_stages = stages;
                smem_size = smem;
                return;
            }
        }
    };

    if (!swap_ab)
    {
        auto num_waves = [&](int bm, int bn)
        { return div_up(div_up(shape_m, bm) * div_up(shape_n, bn), num_device_sms); };
        if (block_m == 128 && block_n > 96 && block_n < 128 && num_waves(128, 128) == num_waves(128, block_n))
        {
            block_n = 128;
            restage();
            num_tma_multicast
                = (shape_m >= 1024 && is_tma_multicast_legal(shape_n, block_n, 2, num_device_sms)) ? 2 : 1;
        }
        if (block_m == 128 && shape_m <= 128 && block_n <= 32)
        {
            block_m = 64;
            block_n *= 2;
            restage();
        }
        if (block_m == 64 && block_n < 32 && 128 % block_n != 0 && num_waves(64, 32) == num_waves(64, block_n))
        {
            block_n = 32;
            restage();
        }
    }
    else
    {
        // shape_m is the weight's N here.
        if (block_m == 128 && div_up(shape_m, 128) * 4 <= num_device_sms)
        {
            block_m = 64;
            restage();
        }
    }
    return std::make_tuple(block_m, block_n, num_stages, num_tma_multicast, smem_size);
}

// The stage count of a dense tile, by get_best_gemm_config's rule: the deepest of 8..4 (6..4 when block_n does not
// divide 128) whose shared memory fits; num_stages and smem_size are left as they are when none fits.
inline void dense_pick_stages(int block_m, int block_n, uint32_t shape_k, bool swap_ab, int& num_stages, int& smem_size)
{
    constexpr int sm90_capacity = 232448;
    std::vector<int> const stage_candidates
        = (128 % block_n != 0) ? std::vector<int>{6, 5, 4} : std::vector<int>{8, 7, 6, 5, 4};
    for (int stages : stage_candidates)
    {
        int const smem = get_smem_size(stages, static_cast<int>(shape_k), block_m, block_n, 128, swap_ab);
        if (smem <= sm90_capacity)
        {
            num_stages = stages;
            smem_size = smem;
            return;
        }
    }
}

// Swap-AB above M = 32 (fso issue 12, 2026-11-02; /data/bench-runs/fso_issue12_20261028/runs/H2). For 32 < M < 64 the
// main path runs a 64-row tile, so 64 - M rows of every activation box it loads lie beyond M. Those boxes move their
// full size whatever the fill (ncu, 2048 x 4096 at 64 x 16: the same L2-to-SM bytes at M = 33, 40, 48 and 64) and the
// out-of-bounds rows are not free: the kernel takes longer the emptier its tile, with the math warp groups waiting
// longer on every stage. Swap-AB puts the tokens on the narrow side of the tile instead (a token tile of 16 to 40
// rows), streams the weight once per token tile, and runs at about the rate of a full main-path tile while its launch
// gives between half and all of the SMs one tile each. It therefore wins on the emptiest 64-row tiles, and how empty
// depends on the weight tile the swap-AB picker takes (get_dense_gemm_config's rule 2):
//   - 64-row weight tiles (ceil(N / 128) * 4 <= SMs, N <= 4224 on 132 SMs; the main tile is then 64 x 16 or 64 x 32,
//     whose traffic is mostly the activation box): a 16-token tile, from 18 empty rows (M <= 46);
//   - 128-row weight tiles (three warp groups per CTA, 72-96 CTAs on the measured shapes; the main tile is 64 x 40 to
//     64 x 96, whose traffic is mostly the weight box): from 28 empty rows (M <= 36) and K >= 2048, with the token
//     tile that covers M in two tiles when two tiles per weight tile fit one wave (5120 x 2048, 6144 x 2560: a
//     24-token tile), else in one (9216 / 9728 / 12288 x 2048-2560: 40 tokens). A token tile of 16 there re-streams
//     each weight tile three times and was slower than 24 on every measured cell. At M = 37 / 38, and with K = 1280
//     (5120 x 1280 at M = 33 / 34), the route measured level or faster alone but slower behind the activation
//     quantize it is chained to.
// The launch floor keeps the main path where swap-AB has too few tiles to beat it: N <= 1280 with 64-row weight tiles
// (20 tiles of 64 rows x 3 token tiles = 60 CTAs) ran at the main path's full-tile rate but not below it, so it lost
// wherever the main tile happened to be fast (1280 x 7680 at M = 40 / 46). A launch that would need a second wave keeps
// the main path as well (19456 x 2560, 152 weight tiles of 128 rows). Below the thresholds swap-AB no longer wins both
// alone and in the chain (M = 48 on the narrow shapes; M = 37 to 44 on the wide ones). M <= 32 is unchanged:
// swap-AB below 32, and at 32 for N < 16384, at get_dense_gemm_config's swap-AB tile. Both routes compute each output
// element in one CTA in K order; the outputs of every re-routed cell of the measurement were byte-identical to the
// main path's.
//
// Swap-AB at 64 < M <= 96 (fso issue 12, 2026-11-02; /data/bench-runs/fso_issue12_20261028/runs/H3). There the main
// path's last M block (two 64-row blocks, or one 128-row block) leaves 128 - M rows empty, from 63 at M = 65 to 32 at
// M = 96, and its boxes again move their full size whatever the fill (ncu, 2048 x 4096 at 64 x 32: the same L2-to-SM
// bytes at M = 96 and 112, with fewer load stalls at 112), while the swap-AB token tiles cover M with less padding: at
// M = 72 the swap-AB 24-token tile moved a third fewer bytes and took 8.9 against 11.4 us (DeepGEMM v2's 64 x 32 tile
// of rule 5), isolated with the caches flushed. It wins, alone and behind the activation quantize, wherever the main
// tile leaves at least 32 rows empty, with the weight tile of rule 2 deciding the token tile and the band:
//   - 64-row weight tiles (N <= 4224 on 132 SMs): the smallest token tile, a multiple of 8 from 16, whose launch fits
//     one wave: 16 for N = 1024 / 1280 (80-120 CTAs), 24 for N = 2048 (96-128), 24 or 32 for N = 2560 (120), 40 or
//     48 for N = 4096 (128). Every measured cell from M = 65 to 96 won (0.66-0.97 of the trunk's time alone,
//     0.67-0.98 in the chain; 10 shapes, two passes); at M = 100 (28 empty rows) 2048 x 512 and 1280 x 5120 lost.
//   - 128-row weight tiles (K >= 2048; the token tile covers M in two tiles when two per weight tile fit one wave,
//     else in one): at M <= 72 (56 empty rows; the main path is DeepGEMM v2's tile there, rule 5); and at
//     80 < M <= 96, where the main path is fp8_gemm_kernel's 128-row tile, when two token tiles per weight tile fit
//     (5120 x 2048, 6144 x 2560, 7168 x 5120). Between them rule 5's DeepGEMM v2 tiles run nearly as fast as swap-AB
//     (12288 x 2048 at M = 76 was level in the chain, 6144 x 2560 at M = 80 1 % slower alone); with one token tile
//     per weight tile the CTAs are fewer than the main path's and each carries a 128 x M tile, which lost from
//     M = 84, 92 or 96 (9216 x 2048, 9728 x 2560, 12288 x 2048; at M = 96 9728 x 2560's 76 CTAs of 128 x 96 took 13.9
//     against 12.5 us for the main path's 122 CTAs of 128 x 80, with a third fewer bytes); at M = 100 5120 x 2048
//     lost in the chain. The re-routed cells measured 0.75-0.98 of the trunk's time alone and 0.79-0.98 in the chain.
// Rule 5 decides the main path wherever this route does not take the cell; the cells it takes at 64 < M <= 80 leave
// the v2 kernel for swap-AB.
//
// The token tile of the swap-AB route above M = 32 (32 < M < 64, and 64 < M <= 96 below) for a weight tile of
// `weight_rows` rows, or 0 when no tile fits one wave: on 64-row weight tiles a 16-token tile below M = 64 and, above
// it, the smallest token tile (a multiple of 8 from 16) whose launch fits one wave; on 128-row weight tiles the token
// tile that covers M in two tiles when two per weight tile fit one wave, else in one.
inline int dense_swap_ab_token_tile(uint32_t shape_m, uint32_t shape_n, int weight_rows, int num_sms)
{
    int const weight_tiles = div_up(static_cast<int>(shape_n), weight_rows);
    int const m = static_cast<int>(shape_m);
    if (weight_rows == 64)
    {
        if (shape_m < 64u)
            return weight_tiles * div_up(m, 16) <= num_sms ? 16 : 0;
        for (int token_tile = 16; token_tile <= 128; token_tile += 8)
            if (weight_tiles * div_up(m, token_tile) <= num_sms)
                return token_tile;
        return 0;
    }
    if (2 * weight_tiles <= num_sms)
        return div_up(div_up(m, 2), 8) * 8;
    if (weight_tiles <= num_sms)
        return div_up(m, 8) * 8;
    return 0;
}

// Swap-AB above M = 96 (fso issue 12, 2026-11-03; /data/bench-runs/fso_issue12_20261028/runs/H4, measured on the
// trunk with the stage-release fence of f041cc8, which makes swap-AB launches of more than one wave correct). From
// M = 97 to 128 the main path is fp8_gemm_kernel, whose last M block is up to 31 rows short; from M = 129 to 191 it is
// DeepGEMM v2 at DeepGEMM's configuration (use_dg2_dense_classes), on these shapes a 64-row tile whose third M block
// holds only M - 128 rows. The CTAs on that block load full boxes for a mostly empty activation tile and are a third of
// the launch, and the v2 kernel loses more behind the activation quantize than alone. Swap-AB pads M only to its token
// tile (0-31 rows over all its token tiles here, against 48-63 rows of the v2 tile's third block at M = 129-144), and
// it won, alone and in the quantize -> GEMM chain in both passes, where:
//   - 64-row weight tiles (N <= 4224 on 132 SMs) with K >= 4096, at 128 < M <= 144 (at least 48 of the third block's
//     64 rows empty): every such cell of the measurement (2048 x 4096, 2560 x 4096 / 9728, 1280 x 5120 / 7680,
//     4096 x 4096; 0.77-0.97 of the trunk's time alone, 0.73-0.87 in the chain). At M = 152 four of the six lost alone
//     (2048 x 4096, 2560 x 4096, 1280 x 5120, 4096 x 4096). With K = 2048 the 2048-wide weight lost alone at M = 144
//     and the 2560-wide one at M = 136 / 144 (also with PDL off, although the isolated kernel, ncu with the caches
//     flushed, was 4 % faster at 2560 x 2048 and M = 136), and 2048 x 512 was level at M = 136, so K < 4096 keeps the
//     main path, also on the shapes that won there by 1-7 % (1024 x 2048, 1280 x 1280, 2048 x 1280).
//   - the same tiles at 96 < M <= 100 with N >= 2048 (the main path's last block 28-31 rows short): 2048 x 4096,
//     2560 x 4096 / 9728, 4096 x 4096 (0.83-0.96 alone, 0.87-0.99 in the chain). At M = 104 2560 x 4096 lost alone; at
//     N = 1280 swap-AB's 100 CTAs lost alone to the main path's 80 at M = 97 and 100 (1280 x 5120) or won by 1 %
//     (1280 x 7680).
//   - 128-row weight tiles at 128 < M <= 144 when the swap-AB launch has fewer CTAs than the main path
//     (dense_main_path_tiles), and up to M = 160 when it has at most three quarters of them: 6144 x 2560, 9216 x 2048,
//     9728 x 2560, 12288 x 2048, 7168 x 5120 (0.64-0.95 alone, 0.65-0.97 in the chain). With as many CTAs as the main
//     path the swap-AB launch lost alone in the replayed graph (5120 x 2048 and 5120 x 1280: 120 CTAs of 128 x 48
//     against 120 of v2's 64 x 128, 2-5 % slower, also with PDL off) although the isolated kernel was 4 % faster with
//     9 % fewer L2-to-SM bytes (the rule follows the bench); with more CTAs (19456 x 2560: 304 against 244) it won
//     by 1-3 % at M = 132 and lost from M = 144; 9216 x 2048 and 9728 x 2560 (216 / 228 against 249 / 261 CTAs) were
//     level at M = 152 and lost in the chain at M = 160; 6144 x 2560 and 12288 x 2048 (96 / 192 against 129 / 258)
//     were level or 3 % faster at M = 176 in two measurements of the same tile and lost at M = 182.
// The token tile is get_best_gemm_config's own criterion with the real weight rows (dense_swap_ab_token_tile_above_96):
// on the wide weights it takes two waves (216 / 228 tiles of 128 x 48 or 192 of 128 x 72-80 on 132 SMs). M = 101-128
// keeps the main path, and so does M = 161-288 outside the two bands below; the cells that still won there (by 1-4 %
// on most shapes, more on 2560 x 4096 / 9728 at M = 208 / 224 and on 7168 x 5120 up to M = 192) are not a class this
// rule separates from the losing neighbours (runs/H4 open items). Rule 5 and the v2 tile classes decide the main path
// wherever this route does not take the cell. Both routes compute each output element in one CTA in K order; the
// outputs of all 3,988 swap-AB arm-cells of the measurement were byte-identical to the trunk route's.
//
// Two more swap-AB bands (fso issue 12, 2026-11-05; /data/bench-runs/fso_issue12_20261028/runs/FH1). In both the main
// path's last M block is mostly empty, and three token tiles of this criterion cover M with at most 23 padded rows:
//   - Narrow outputs (64-row weight tiles) at 144 < M <= 216, where the last 64-row block of the main path holds at
//     most 24 rows (M = 145-152 and 193-216) and the launch, three token tiles per weight tile, gives at least three
//     quarters of the SMs a tile in one wave (2048 < N <= 2816 on 132 SMs): with K >= 8192 against the v2 kernel's
//     64 x 64 / 64 x 80 tiles (2560 x 9216 at M = 152-208: 0.97-1.03 of the route's bare GEMM, 0.77-0.83 of its served
//     call and 0.98 with PDL off), and with K >= 4096 against fp8_gemm_kernel's 128 x 40 tile, whose second block is
//     40 rows short at M = 209-216 (2560 x 4096 and 2560 x 9216 at M = 216: 0.76-0.85 bare, 0.78-0.88 served). At
//     K = 4096 against the v2 tiles (2560 x 4096 at M = 152-208) the swap-AB launch was level or slower on the bare
//     GEMM and won only the served call with PDL on, where the v2 grid loses 1.3-4 us behind the activation quantize
//     (runs/FH1 n1_nsys and its held list): those cells keep the main path.
//   - Wide outputs (128-row weight tiles) with K >= 2560 at 256 < M <= 288, when the launch needs no more waves than
//     the main path. The main path's 128-row tiles then hold 0-32 rows in their third block, so they waste 96-120 of
//     every 384 rows they compute, while three token tiles of 88-96 tokens waste at most 16 (288 = 3 x 96). Measured on
//     10240 x 2560 at M = 264-288: 0.73-0.84 of the route's bare GEMM, 0.78-0.85 of its served call. The larger token
//     tiles do not pay: 104-128 tokens ran at 0.98-1.01 of the route at M = 304 / 320, 192 tokens at twice its time.
//     A wide output whose v2 tile of use_dense_v2_override needs fewer waves (12352 x 2560, 18432 x 2560) takes that
//     tile instead.
//
// The token tile of the swap-AB route above M = 96 for a weight tile of `weight_rows` rows: get_best_gemm_config's own
// criterion (the fewest waves on num_sms, then the fullest last wave, then the smaller tile), applied with the weight
// tile rule 2 picks. On 128-row weight tiles this is the picker's own token tile; on 64-row weight tiles the picker
// counts 128-row tiles and may pick a tile that needs two waves once the rows are halved, where this criterion takes
// the smallest token tile that fits one wave.
inline int dense_swap_ab_token_tile_above_96(uint32_t shape_m, uint32_t shape_n, int weight_rows, int num_sms)
{
    int const weight_tiles = div_up(static_cast<int>(shape_n), weight_rows);
    int best = 0, best_waves = 0, best_last = 0;
    for (int token_tile = 16; token_tile <= 128; token_tile += 8)
    {
        int const tiles = weight_tiles * div_up(static_cast<int>(shape_m), token_tile);
        int const waves = div_up(tiles, num_sms);
        int const last = tiles - (waves - 1) * num_sms; // tiles of the last wave
        if (best == 0 || waves < best_waves || (waves == best_waves && last > best_last))
        {
            best = token_tile;
            best_waves = waves;
            best_last = last;
        }
    }
    return best;
}

// The number of CTAs the main path launches for a dense GEMM (defined after use_dg2_dense below).
inline int dense_main_path_tiles(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms);

// Whether the main path runs a dense GEMM on DeepGEMM v2's kernel (use_dg2_dense; defined after it below).
inline bool dense_main_path_is_v2(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms);

// Whether gemm_dispatch_sm90 runs a dense GEMM with M > 96 through swap-AB (see above).
inline bool use_dense_swap_ab_above_96(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    bool const low_band = shape_m <= 100u;                      // 96 < M <= 100
    bool const dg2_band = shape_m > 128u && shape_m <= 160u;    // the third 64-row block of M holds 1 to 32 rows
    bool const narrow_band = shape_m > 144u && shape_m <= 216u; // the last 64-row block holds 1 to 24 rows (below)
    bool const wide_band = shape_m > 256u && shape_m <= 288u;   // the third 128-row block holds 1 to 32 rows
    if (!low_band && !dg2_band && !narrow_band && !wide_band)
        return false;
    int const weight_rows = std::get<0>(get_dense_gemm_config(shape_n, shape_m, shape_k, num_sms, true));
    int const token_tile = dense_swap_ab_token_tile_above_96(shape_m, shape_n, weight_rows, num_sms);
    int const tiles = div_up(static_cast<int>(shape_n), weight_rows) * div_up(static_cast<int>(shape_m), token_tile);
    if (2 * tiles < num_sms) // fewer than half the SMs would get a tile
        return false;
    if (narrow_band && weight_rows == 64)
    {
        // Narrow outputs (the band below): three token tiles in one wave on at least three quarters of the SMs, with
        // K >= 8192 or against fp8_gemm_kernel's tile.
        int const last = static_cast<int>(shape_m % 64u); // rows of the main path's last 64-row block
        if (shape_k < 4096u || last < 1 || last > 24 || div_up(static_cast<int>(shape_m), token_tile) != 3
            || tiles > num_sms || 4 * tiles < 3 * num_sms)
            return false;
        return shape_k >= 8192u || !dense_main_path_is_v2(shape_m, shape_n, shape_k, num_sms);
    }
    if (wide_band)
    {
        // Wide outputs (the band below): three token tiles of at most 96 tokens, no more waves than the main path.
        return weight_rows == 128 && shape_k >= 2560u && div_up(static_cast<int>(shape_m), token_tile) == 3
            && token_tile <= 96
            && div_up(tiles, num_sms) <= div_up(dense_main_path_tiles(shape_m, shape_n, shape_k, num_sms), num_sms);
    }
    if (!low_band && !dg2_band)
        return false;
    if (weight_rows == 64)
        return shape_k >= 4096u && (low_band ? shape_n >= 2048u : shape_m <= 144u);
    if (!dg2_band)
        return false;
    int const main_tiles = dense_main_path_tiles(shape_m, shape_n, shape_k, num_sms);
    return (shape_m <= 144u && tiles < main_tiles) || 4 * tiles <= 3 * main_tiles;
}

// Whether gemm_dispatch_sm90 runs the dense GEMM through swap-AB (see above).
inline bool use_dense_swap_ab(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    if (shape_m < 32u || (shape_m == 32u && shape_n < 16384u))
        return true;
    if (shape_m > 96u)
        return use_dense_swap_ab_above_96(shape_m, shape_n, shape_k, num_sms);
    if (shape_m <= 32u || shape_m == 64u)
        return false;
    int const weight_rows = std::get<0>(get_dense_gemm_config(shape_n, shape_m, shape_k, num_sms, true));
    int const token_tile = dense_swap_ab_token_tile(shape_m, shape_n, weight_rows, num_sms);
    if (token_tile == 0)
        return false;
    int const weight_tiles = div_up(static_cast<int>(shape_n), weight_rows);
    int const tiles = weight_tiles * div_up(static_cast<int>(shape_m), token_tile);
    if (2 * tiles < num_sms) // fewer than half the SMs would get a tile
        return false;
    if (shape_m < 64u)
    {
        uint32_t const empty = 64u - shape_m; // rows of the main path's 64-row tile beyond M
        if (weight_rows == 64)
            return empty >= 18u;
        return empty >= 28u && shape_k >= 2048u;
    }
    // 64 < M <= 96: the main path's last M block (two 64-row blocks or one 128-row block) leaves 128 - M >= 32 rows
    // empty.
    uint32_t const empty = 128u - shape_m;
    if (weight_rows == 64)
        return true;
    if (shape_k < 2048u)
        return false;
    if (empty >= 56u) // M <= 72
        return true;
    return empty < 48u && 2 * weight_tiles <= num_sms; // 80 < M <= 96 with two token tiles per weight tile
}

// The swap-AB tile of the dense GEMM (shape convention of gemm_dispatch_sm90: M tokens, N weight rows):
// get_dense_gemm_config's swap-AB pick, with the token tile of dense_swap_ab_token_tile at 32 < M < 64 and 64 < M <= 96
// (where it gives one) and of dense_swap_ab_token_tile_above_96 at 96 < M <= 288, and the stage count recomputed.
//
// At 8 <= M <= 32 a 128-row weight tile of rule 2 (N > 4224 on 132 SMs) is replaced by a 64-row one with a single token
// tile that covers M (16 tokens up to M = 16, else M rounded up to 8) when the 64-row launch fits one wave (N <= 8448
// on 132 SMs). The pick it replaces streams each weight tile in one CTA per 16 tokens and gives work to at most half
// the SMs (40-66 CTAs of 128 rows); the 64-row tiles stream every weight row once, in twice the CTAs, so the weight,
// which is nearly all the traffic at these M, arrives at the bandwidth of up to every SM (fso issue 12, FH3's lever,
// measured by FH1, runs/FH1/runs/fh3_lever, fh3_lever2 and rf_ab): 0.80-0.96 of the bare GEMM and 0.83-0.96 of the
// served call (0.84-0.97 with PDL off) on every measured cell at M = 8-32 (5120 x 1280 / 2048 / 2560, 6144 x 2048 /
// 2560, 7168 x 2560, 8192 x 2048). Below M = 8 the served call measured mixed (6144 x 2560: 0.96 at M = 1 / 2, 1.03 at
// M = 3, 1.08 at M = 4), so those M keep the 128-row tile. Every output element is still accumulated by one CTA in K
// order, so the bytes are those of the 128-row tile.
inline GemmConfig get_dense_swap_ab_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    auto [block_m, block_n, num_stages, num_tma_multicast, smem_size]
        = get_dense_gemm_config(shape_n, shape_m, shape_k, num_sms, true);
    int token_tile = 0;
    if (shape_m > 32u && shape_m != 64u && shape_m <= 96u)
        token_tile = dense_swap_ab_token_tile(shape_m, shape_n, block_m, num_sms);
    else if (shape_m > 96u && shape_m <= 288u)
        token_tile = dense_swap_ab_token_tile_above_96(shape_m, shape_n, block_m, num_sms);
    else if (shape_m >= 8u && shape_m <= 32u && block_m == 128 && div_up(static_cast<int>(shape_n), 64) <= num_sms)
    {
        block_m = 64; // 64-row weight tiles, one token tile covering M
        token_tile = std::max(16, div_up(static_cast<int>(shape_m), 8) * 8);
    }
    if (token_tile != 0)
    {
        block_n = token_tile;
        dense_pick_stages(block_m, block_n, shape_k, true, num_stages, smem_size);
    }
    return std::make_tuple(block_m, block_n, num_stages, num_tma_multicast, smem_size);
}

// ------------------------------------------------------------------------------------------------------------------
// The in-cluster split-K route of the dense GEMM (fp8_gemm_kernel_swapAB_splitk, fso_dense_splitk.cuh): one 64-row
// weight tile per cluster of `splits` CTAs that divide K between them, one token tile covering all of M.
struct DenseSplitKConfig
{
    uint32_t block_t = 0; // token tile: max(16, M rounded up to 8), at most 64
    uint32_t splits = 0;  // CTAs per cluster, each taking one slice of K; 0 when the route does not apply
    uint32_t stages = 0;  // pipeline stages per CTA
    uint32_t smem = 0;    // dynamic shared memory per CTA in bytes
    uint32_t tiles = 0;   // 64-row weight tiles = clusters

    [[nodiscard]] bool valid() const
    {
        return splits != 0;
    }

    [[nodiscard]] uint32_t ctas() const
    {
        return tiles * splits;
    }
};

// The dynamic shared memory from which a CTA leaves no room for a second one on an sm_90 SM (two need 2 x (smem + 1 KB
// reserved) <= 228 KB). The split-K kernel requests at least this much, so that its CTAs never share an SM: launched
// early behind a programmatic predecessor (the activation quantize ahead of a bf16-x call), the CTAs of a kernel that
// would fit twice are stacked two to an SM on the SMs that are free at that moment, and a stacked CTA streams its slice
// at half the rate (FH5 trace, runs/FH5/data/hold_t2.jsonl: down 2560 x 9216 at M = 24 ran 80 CTAs on 44 SMs behind
// the quantize, its K loop 7.9 against 6.2 us on 80 SMs after a GEMM). A predecessor without shared memory can still
// share the SM.
constexpr uint32_t kDenseSplitKOneCtaPerSmBytes = 114u * 1024u;

// The split-K launch of an M x N x K dense GEMM with `splits` K slices, whatever the route rule says: the token tile
// covering M, 8 stages (fewer when a slice has fewer K blocks) and the shared memory of fso_dense_splitk_layout.cuh,
// raised to kDenseSplitKOneCtaPerSmBytes when `one_cta_per_sm`. Invalid when the kernel cannot serve the shape:
// M = 0 or above 64, K not a multiple of 128, N not a multiple of 8 (the D tensor map's row stride), or a slice count
// outside 1 .. 4 or above K / 128.
inline DenseSplitKConfig make_dense_splitk_config(
    uint32_t m, uint32_t n, uint32_t k, uint32_t splits, bool one_cta_per_sm = true)
{
    namespace L = deep_gemm::fso_dense_splitk_smem;
    DenseSplitKConfig c;
    if (m == 0 || m > 64 || n == 0 || k == 0 || k % 128 != 0 || n % 8 != 0 || splits < 1 || splits > L::kNumSlabs
        || splits > k / 128)
        return c;
    constexpr uint32_t kCapacity = 232448; // sm_90 dynamic shared memory per block
    uint32_t const kb_slice = (k / 128 + splits - 1) / splits;
    c.block_t = std::max(16u, (m + 7) / 8 * 8);
    c.stages = std::min(8u, kb_slice);
    c.smem = L::total_bytes(c.block_t, splits, c.stages, kb_slice);
    if (one_cta_per_sm)
        c.smem = std::max(c.smem, kDenseSplitKOneCtaPerSmBytes);
    c.tiles = (n + L::kBlockW - 1) / L::kBlockW;
    if (c.smem > kCapacity)
        return DenseSplitKConfig{};
    c.splits = splits;
    return c;
}

// The slice count the split-K route prefers for an N x K weight on `num_sms` SMs, 0 when it does not take the shape
// (issue #12, FH5; /data/bench-runs/fso_issue12_20261028/runs/FH5, holds B, E and G). With tiles = ceil(N / 64)
// clusters, one per 64-row weight tile: the largest S of 4 and 3 whose S * tiles CTAs take at most three quarters of
// the SMs and whose slices hold at least 2 K blocks each; else 3 when K > 4096 and 3 * tiles CTAs fit the SMs; else 2
// when 2 * tiles CTAs fit them and the slices hold at least 2 K blocks. More slices stream a weight faster until about
// three quarters of the SMs work; past that they add reduction work without a faster stream, except on long K: at
// K <= 4096, S = 3 on 40 tiles (2560 x 4096) tied S = 2 at M <= 8 and lost 3.5 % at M = 16, while on 32 tiles
// (2048 x 4096) it won 4-5 % over S = 2 and S = 4 lost; at K = 9216, S = 3 on 40 tiles won 2-6 % at M <= 24 (served
// call); with 4 K blocks (2048 x 512) S = 2 won over S = 3, whose slices hold 2 / 1 / 1. The dispatcher lowers the
// count where one cluster per tile does not fit at once (sm90_dense_splitk_route in dispatch.cuh).
inline uint32_t dense_splitk_preferred_slices(uint32_t n, uint32_t k, int num_sms)
{
    if (num_sms <= 0 || n == 0 || k == 0 || k % 128 != 0)
        return 0;
    uint32_t const tiles = (n + 63) / 64;
    uint32_t const kb = k / 128;
    uint32_t const sms = static_cast<uint32_t>(num_sms);
    for (uint32_t s = 4; s >= 3; --s)
    {
        if (4 * s * tiles <= 3 * sms && 2 * s <= kb)
            return s;
    }
    if (k > 4096u && 3 * tiles <= sms && 6 <= kb)
        return 3;
    if (2 * tiles <= sms && 4 <= kb)
        return 2;
    return 0;
}

// The largest M the split-K route takes at `splits` slices and K = `k`, by K class: M <= 64 at K >= 8192; M <= 56 at
// 4096 <= K < 8192; at 2048 <= K < 4096, M <= 40 with 3 or 4 slices and M <= 24 with 2; M <= 24 below K = 2048. Every
// call of the band won on the served bf16-x call and on the 2b call against the routes it replaces (FH5 holds C, E, G
// and BAND, two passes): down 2560 x 9216 -5 % at M = 64; at M = 52 / 56 every N of 1024 .. 4224 at K = 4096, 4608,
// 5120, 6144 and 7680, in both passes (2560 x 5120 at M = 56 by the least, -0.8 % and -4.3 %; o 2560 x 4096 -6 % and
// -2 %, llm_o 2048 x 4096 -15 % and -4 % at M = 52 and 56); 2048 x 2048 -7 % and 1024 x 2048 -15 % at M = 40,
// 2560 x 2048 -3 % and 2048 x 512 -5 % at M = 24. Past the edges the split route lost or tied: at M = 60 / 64 (token
// tile 64) from K = 4096 to 6144 on most N (at K = 4096 o +5 % / +6 %, llm_o +0.4 % / +7 %; it still won on every N
// at K = 7680), 2560 x 2048 at M = 32 (+0.5 %), 2048 x 2048 and 1024 x 2048 at M = 64 (+16 %, +19 %), 2048 x 512 at
// M = 32 (-1 %).
inline uint32_t dense_splitk_max_m(uint32_t splits, uint32_t k)
{
    if (k >= 8192u)
        return 64u;
    if (k >= 4096u)
        return 56u;
    if (k >= 2048u && splits >= 3)
        return 40u;
    return 24u;
}

// The split-K route's configuration for an M x N x K dense GEMM at `splits` slices under the route's band
// (dense_splitk_max_m), one CTA per SM; invalid outside the band.
inline DenseSplitKConfig get_dense_splitk_config(uint32_t m, uint32_t n, uint32_t k, uint32_t splits)
{
    if (splits < 2 || m > dense_splitk_max_m(splits, k))
        return DenseSplitKConfig{};
    return make_dense_splitk_config(m, n, k, splits, /*one_cta_per_sm=*/true);
}

// ------------------------------------------------------------------------------------------------------------------
// DeepGEMM v2's dense block-FP8 kernel (deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh, vendored from sgl-project/DeepGEMM
// v0.2.0, MIT): its configuration for one dense GEMM, and the rule that decides when the dense dispatcher runs it
// instead of fp8_gemm_kernel.
//
// The two kernels compute every output element in one CTA, accumulate each 128-wide K block of WGMMA results in FP32
// and promote it with the same A-scale x B-scale product in the same K order, so their outputs are bitwise identical
// whatever the tile (checked on every dense family cell, 2026-10-11); choosing between them is a tile choice. The v2
// kernel adds what fp8_gemm_kernel cannot run: 256-row tiles (two 128-row WGMMA waves per k-block), 144- to 192-wide
// tiles (a compile-time dispatch of the weight-scale split point), multicast on either operand with a per-block
// fallback for an odd tail, a stage ring that runs on across k-blocks and tiles (any stage count up to 16, no padded
// rounds), and the swizzled D store for every width.
struct Dg2DenseConfig
{
    uint32_t block_m = 0;
    uint32_t block_n = 0;
    uint32_t num_stages = 0;
    uint32_t cluster_m = 1; // DeepGEMM's layout: cluster_m > 1 pairs two M blocks, which share B (multicast on B)
    uint32_t cluster_n = 1; // cluster_n > 1 pairs two N blocks, which share A (multicast on A)
    uint32_t swizzle_d = 0; // the D staging swizzle in bytes (128, 64, 32, or 16 = none)
    uint32_t smem_size = 0;
    uint32_t shape_n = 0;   // N compiled into the kernel (DeepGEMM's specialize_n), or 0 for a runtime N
    uint32_t num_sms = 0;
    int num_waves = 0;
    int64_t num_cycles = 0;

    uint32_t cluster_size() const
    {
        return cluster_m * cluster_n;
    }

    bool multicast_on_a() const
    {
        return cluster_n > 1;
    }

    uint32_t num_math_threads() const
    {
        return block_m <= 64 ? 128 : 256;
    }
};

// heuristics/utils.hpp get_swizzle_mode: the widest swizzle (128, 64, 32 or 16 bytes, 16 meaning none) that divides
// one row of the block.
inline uint32_t dg2_swizzle_mode(uint32_t block_size, uint32_t elem_size)
{
    for (uint32_t mode : {128u, 64u, 32u, 16u})
        if ((block_size * elem_size) % mode == 0)
            return mode;
    return 16u;
}

// heuristics/sm90.hpp get_pipeline_config for the 1D2D kernel (FP8 A/B, BF16 D, no accumulation): the shared memory
// left after D, the barriers and the B scales holds as many stages as fit, at most 16. Returns the stage count and
// writes the total shared memory.
inline int dg2_num_stages(uint32_t block_m, uint32_t block_n, uint32_t shape_k, uint32_t& smem_size)
{
    constexpr int kCapacity = 232448;
    constexpr int kMaxStages = 16;
    auto align = [](int x, int a) { return (x + a - 1) / a * a; };
    int const smem_cd = align(static_cast<int>(block_m * block_n * 2), 1024);
    int const smem_barriers = kMaxStages * 8 * 2;
    int const smem_a = static_cast<int>(block_m) * 128;
    int const smem_b = static_cast<int>(block_n) * 128;
    int const smem_sfa = align(static_cast<int>(block_m) * 4, 128);
    int const uniform_sfb = (128 % block_n == 0) ? 1 : 2;
    int const smem_extra_sfb = align(div_up(static_cast<int>(shape_k), 128) * 4 * uniform_sfb, 8);
    int const extra = smem_cd + smem_barriers + smem_extra_sfb;
    int const per_stage = smem_a + smem_b + smem_sfa;
    int const stages = std::min((kCapacity - extra) / per_stage, kMaxStages);
    smem_size = static_cast<uint32_t>(extra + stages * per_stage);
    return stages;
}

// DeepGEMM v2's configuration for a dense GEMM (heuristics/sm90.hpp, SM90ArchSpec: get_layout_candidates,
// get_layout_info and compare, in DeepGEMM's enumeration order, ties keeping the earlier candidate), so that the
// vendored kernel runs exactly the tile, cluster, swizzle and stage count sglang's DeepGEMM runs for the shape.
// Candidates: block_m 64 and 128 (plus 16 at M <= 16 and 32 at M <= 32) and 256; block_n 16 to 192 in steps of 16
// that satisfy the 1D2D kernel's weight-scale split (block_n <= 128, or block_n - 128 dividing block_n or 128); not
// both block sizes above 128; clusters 1x1, 1x2 (multicast on A) and 2x1 (multicast on B); at least 3 stages, 4 for
// tiles smaller than 128x192. Cost: L1 and L2 bytes per block over the SMs' bandwidth, divided by the wave
// efficiency; a cluster with only one wave is never chosen.
inline Dg2DenseConfig get_dg2_dense_config(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    std::vector<int> block_ms = {64, 128};
    if (shape_m <= 16)
        block_ms.push_back(16);
    if (shape_m <= 32)
        block_ms.push_back(32);
    block_ms.push_back(256);
    std::vector<int> block_ns;
    for (int bn = 16; bn <= 192; bn += 16)
        block_ns.push_back(bn);
    constexpr int block_k = 128;

    Dg2DenseConfig best;
    bool have = false;
    for (int cluster_m = 1; cluster_m <= 2; ++cluster_m)
    {
        for (int cluster_n = 1; cluster_n <= 2; ++cluster_n)
        {
            if (cluster_m * cluster_n > 2 || num_sms % (cluster_m * cluster_n) != 0)
                continue;
            for (int bm : block_ms)
            {
                for (int bn : block_ns)
                {
                    if (bn > block_k && (bn % (bn - block_k) != 0 && block_k % (bn - block_k) != 0))
                        continue;
                    if (bm > 128 && bn > 128)
                        continue;
                    uint32_t smem = 0;
                    int const stages = dg2_num_stages(bm, bn, shape_k, smem);
                    if (stages < 3 || (bm * bn < 128 * 192 && stages < 4))
                        continue;

                    // get_layout_info
                    int64_t const num_blocks = static_cast<int64_t>(div_up(static_cast<int>(shape_m), bm))
                        * div_up(static_cast<int>(shape_n), bn);
                    int64_t const num_waves = (num_blocks + num_sms - 1) / num_sms;
                    int64_t const l2_bw = static_cast<int64_t>(std::min(64. * num_sms, 8e6 / 1.3e3));
                    int64_t const l1_bw = 128 * static_cast<int64_t>(num_sms);
                    int64_t const k = shape_k;
                    int64_t const l2_ab = k * (bm / cluster_n + bn / cluster_m);
                    int64_t const l1_ab = k * (bm + bn);
                    int64_t const l1_tc = k * (std::max(64, bm) + bn) + static_cast<int64_t>(bm) * bn * 2;
                    int64_t const cd = static_cast<int64_t>(bm) * bn * 2;
                    int64_t const l2_cycles = (l2_ab + cd) * num_blocks / l2_bw;
                    int64_t const l1_cycles = (l1_ab + l1_tc + cd) * num_blocks / l1_bw;
                    float const wave_eff = static_cast<float>(num_blocks) / static_cast<float>(num_waves * num_sms);
                    int64_t cycles = static_cast<int64_t>(static_cast<float>(std::max(l1_cycles, l2_cycles)) / wave_eff);
                    if (cluster_m * cluster_n > 1 && num_waves <= 1)
                        cycles = std::numeric_limits<int64_t>::max();

                    if (!have || cycles < best.num_cycles)
                    {
                        have = true;
                        best.block_m = bm;
                        best.block_n = bn;
                        best.num_stages = stages;
                        best.cluster_m = cluster_m;
                        best.cluster_n = cluster_n;
                        best.smem_size = smem;
                        best.num_waves = static_cast<int>(num_waves);
                        best.num_cycles = cycles;
                    }
                }
            }
        }
    }
    best.swizzle_d = dg2_swizzle_mode(best.block_n, 2);
    best.num_sms = static_cast<uint32_t>(num_sms);
    // sm90_fp8_gemm_1d2d.hpp: "A known N keeps wide single-CTA SM90 tiles free of register spills."
    best.shape_n = (best.block_m == 128 && best.block_n > 128 && best.cluster_size() == 1) ? shape_n : 0u;
    return best;
}

// get_dg2_dense_config behind a small per-thread direct-mapped cache keyed by the shape, so that an eager call pays
// one lookup instead of the ~180-candidate enumeration.
inline Dg2DenseConfig get_dg2_dense_config_cached(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    struct Entry
    {
        uint32_t m = 0, n = 0, k = 0;
        int sms = -1;
        Dg2DenseConfig cfg;
    };
    thread_local Entry table[256];
    uint64_t const h = (static_cast<uint64_t>(shape_m) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(shape_n) << 20)
        ^ (static_cast<uint64_t>(shape_k) << 40) ^ static_cast<uint64_t>(num_sms);
    Entry& e = table[(h ^ (h >> 29)) & 255u];
    if (e.sms != num_sms || e.m != shape_m || e.n != shape_n || e.k != shape_k)
    {
        e.cfg = get_dg2_dense_config(shape_m, shape_n, shape_k, num_sms);
        e.m = shape_m;
        e.n = shape_n;
        e.k = shape_k;
        e.sms = num_sms;
    }
    return e.cfg;
}

// Whether the dense dispatcher runs DeepGEMM v2's kernel at its configuration (`dg2`) instead of fp8_gemm_kernel at
// get_dense_gemm_config's pick, on the main (non-swap-AB) path, by the tile classes below (use_dg2_dense_classes) or
// by rule 5 further down (a mostly empty last M block, use_dg2_dense_partial_block). The v2 kernel is taken for the
// tile classes
// fp8_gemm_kernel cannot run and that measured faster on the H200 against fso's pick wherever DeepGEMM chose them
// (H1, /data/bench-runs/sm90_dg2_dense_20261005/phase2/runs/remaining/report.md; P2 of the 0.2.1 round,
// /data/bench-runs/fso_0.2.1_20261010/P2):
//   - 256 x 128 with multicast on A: 0.86-0.99 of fso's time on 23 cells (M >= 1024 on the wide outputs, M >= 4096
//     on the narrow ones);
//   - 128 x 160 and 128 x 192, with or without multicast on B: 0.70-0.96 on 16 cells (M = 128-2048);
// and only from M = 512 on, where the MLP block gains 4-11 % from M = 1024. At M = 128 and M = 256 these tiles do not
// pay inside the block: at M = 128 the 128 x 160 gate_up runs 1.2 us longer inside the quantize -> GEMM -> SwiGLU ->
// quantize -> GEMM chain than fso's 128 x 80 (20.4 against 19.2 us, two traced runs) although it is 1 us faster in a
// graph of back-to-back GEMMs, and at M = 256 the block is level within noise.
// Two decode-band classes are taken as well (the serving buckets M = 32-192 weigh most, stone 2026-10-12):
//   - every DeepGEMM pick at 128 < M < 256 except its 80-wide tiles. There fp8_gemm_kernel's pick is a 128-row tile
//     whose second M block is only partly filled, and it runs far slower than at M = 256 with the same tiles: at
//     M = 160 the MLP block's gate_up takes 38.6 us and its down 36.0 us against 28.0 and 19.7 us at M = 256.
//     DeepGEMM's picks there (128 x 160 with multicast on B for gate_up; 64 x 32 to 64 x 144 and 128 x 144 to 128 x
//     192 for the others) take every dense family GEMM to 0.53-0.87 of its time at M = 144-192 and the Family A block
//     from 81.5 to 55.4 us at M = 160 and from 72.3 to 51.9 us at M = 192. At M = 208-240 DeepGEMM picks 80-wide
//     tiles for the narrow outputs (64 x 80, 128 x 80), which are 8-14 % slower than fso's pick there (its kernel is
//     weak at 80-wide tiles, H1), so those keep fp8_gemm_kernel;
//   - DeepGEMM's 64-row tiles at M <= 128 when K >= 8192 and M fills whole 64-row blocks. Each CTA then owns one
//     output tile and streams at least 64 k-blocks; fp8_gemm_kernel restarts its stage ring per round and caps a tile
//     whose width does not divide 128 at 6 stages, where v2 keeps up to 16 in flight. In the MLP block (the down
//     projection, K = 9728) 64 x 48 with 15 stages takes 14.9 us against fp8_gemm_kernel's 16.2 us at M = 128 (the
//     block 3.8 % less), and 64 x 32 with 16 stages gains 0.3 us at M = 64. With a partly filled last 64-row block
//     (M = 96) it is 0.4 us slower than fp8_gemm_kernel, and at K = 2048-4096 the same tiles gain nothing or lose
//     (1.0-1.06 timed alone), so those keep fp8_gemm_kernel.
// Every other v2 pick keeps fso's kernel: 256-row tiles narrower than 128 (256 x 112: 1.04-1.29), 256 x 128 without
// multicast (one wave: 1.9-2.3), 128 x 144 (mixed), the 64-row mid-band tiles (1.03-1.22 on most cells) and the
// 16- and 32-row small-M tiles, where fso's swap-AB kernel is faster.
inline bool use_dg2_dense_classes(Dg2DenseConfig const& dg2, uint32_t shape_m, uint32_t shape_k)
{
    if (shape_m > 128 && shape_m < 256)
        return dg2.block_n != 80;
    if (shape_m < 512)
        return dg2.block_m == 64 && shape_m % 64 == 0 && shape_m <= 128 && shape_k >= 8192 && dg2.block_n >= 32;
    if (dg2.block_m == 256)
        return dg2.block_n == 128 && dg2.cluster_size() == 2 && dg2.multicast_on_a();
    if (dg2.block_m == 128)
        return dg2.block_n == 160 || dg2.block_n == 192;
    return false;
}

// Rule 5 (fso issue 12, 2026-10-28; /data/bench-runs/fso_issue12_20261028/runs/H1): a mostly empty last M block.
// When fso's tile (get_dense_gemm_config) leaves at least 48 rows of its last M block beyond M, the CTAs on that block
// load TMA boxes that are mostly out of bounds, and the zero-filled rows are not free: the boxes still move their full
// size (ncu, 1280 x 7680 at 128 x 40: 169.1 MB from the L2 to the SMs at M = 400 and at M = 512 alike, and 14 % more
// L2 read sectors at M = 400), and the math warp groups wait longer for every stage. A one-wave GEMM then takes up to
// 1.5 times as long as with that block full (tower fc2 1280 x 5120 at 128 x 40: 17.1 against 11.3 us at M = 400 and
// 512). DeepGEMM v2's 128-row tiles pay it as well (1.25-1.45 on that shape), its 64-row tiles stay within 3 %, and
// DeepGEMM's cost model, which counts the blocks of the padded M, picks those or a fuller wave on such shapes. Below
// M = 1024 the shape therefore runs the v2 kernel:
//   - at DeepGEMM's configuration when that is a single-CTA one and fso's tile fills one wave or needs more waves than
//     it (qkv 5120 x 2048 and o 2048 x 4096 at M = 80, 400-464 and o at 272-336, the tower projections at M = 400-576:
//     0.55-0.98 of fso's time alone, 0.66-0.99 in the fso_chain arm of bench_issue12_dense.py, the quantize ->
//     GEMM chain a serving step runs; exceptions below);
//   - at fso's own tile when DeepGEMM picks a two-CTA cluster, which needs two waves there, and fso's tile fills one
//     (qkv at M = 272-320: 0.79-0.88 alone, 0.90-0.94 in the chain);
//   - otherwise it keeps fp8_gemm_kernel (tower fc1 5120 x 1280 at M = 533 / 576: DeepGEMM's pick clustered and fso's
//     tile two waves; no tile of either kernel fills one wave there).
// Up to 32 empty rows (M = 96 on every published family shape, M = 112, 352, 480, 496) keep the previous choice: the
// cost is small there and the v2 kernel loses its gain back in the chain. From M = 1024 the classes above decide alone
// (at M = 1079 the v2 kernel takes the tower conv_out 0.89 of its time alone but 1.10 in the chain, fc2 1.02 / 1.07).
// The v2 configuration must also give at least three quarters of the SMs a tile: with fewer, its time depends on the
// SMs its CTAs land on, and fp8_gemm_kernel's does not (tower fc2 / conv_out at M = 78, 80 tiles of 64 x 32 on the v2
// kernel: rounds alternate between 8.4 and 10.3 us and between 11.3 and 13.7 us, against a steady 8.65 / 11.5 us;
// runs/H1/val_i12). Measured exceptions inside the rule: o at M = 336 (0.81 alone, 1.07 in the chain), the 1280 x 1280
// projection at M = 448 (0.98 / 1.02) and proj2 2048 x 1280 at M = 576 (1.01 / 1.02). Both kernels compute the same
// bits.
inline bool use_dg2_dense_partial_block(
    Dg2DenseConfig& dg2, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    if (shape_m >= 1024)
        return false;
    auto [bm, bn, ns, mc, smem] = get_dense_gemm_config(shape_m, shape_n, shape_k, num_sms, false);
    (void) ns;
    (void) smem;
    uint32_t const ubm = static_cast<uint32_t>(bm);
    uint32_t const empty = (ubm - shape_m % ubm) % ubm; // rows of the last M block beyond M
    if (empty < 48 || mc != 1)
        return false;
    int const fso_tiles = div_up(static_cast<int>(shape_m), bm) * div_up(static_cast<int>(shape_n), bn);
    int const fso_waves = div_up(fso_tiles, num_sms);
    // The v2 configuration must give at least three quarters of the SMs a tile (see above).
    auto fills = [&](int tiles) { return 4 * tiles >= 3 * num_sms; };
    if (dg2.cluster_size() == 1)
    {
        int const dg2_tiles = div_up(static_cast<int>(shape_m), static_cast<int>(dg2.block_m))
            * div_up(static_cast<int>(shape_n), static_cast<int>(dg2.block_n));
        return (fso_waves == 1 || dg2.num_waves < fso_waves) && fills(dg2_tiles);
    }
    if (fso_waves != 1 || bn % 16 != 0 || !fills(fso_tiles))
        return false;
    Dg2DenseConfig c;
    c.block_m = static_cast<uint32_t>(bm);
    c.block_n = static_cast<uint32_t>(bn);
    c.num_stages = static_cast<uint32_t>(dg2_num_stages(c.block_m, c.block_n, shape_k, c.smem_size));
    if (c.num_stages < 4)
        return false;
    c.swizzle_d = dg2_swizzle_mode(c.block_n, 2);
    c.num_sms = static_cast<uint32_t>(num_sms);
    c.num_waves = fso_waves;
    c.shape_n = 0u;
    dg2 = c;
    return true;
}

// `dg2` is DeepGEMM's configuration on entry; rule 5 may replace it with the configuration the v2 kernel runs.
inline bool use_dg2_dense(Dg2DenseConfig& dg2, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    if (use_dg2_dense_classes(dg2, shape_m, shape_k))
        return true;
    return use_dg2_dense_partial_block(dg2, shape_m, shape_n, shape_k, num_sms);
}

// The number of CTAs the main (non-swap-AB) path launches for a dense GEMM in the default FSO_SM90_DENSE_DG2 mode: the
// tiles of DeepGEMM v2's configuration where use_dg2_dense takes it (a two-CTA cluster counts both CTAs), else of
// fp8_gemm_kernel's get_dense_gemm_config tile. use_dense_swap_ab_above_96 compares the swap-AB launch with it.
inline int dense_main_path_tiles(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    Dg2DenseConfig dg2 = get_dg2_dense_config_cached(shape_m, shape_n, shape_k, num_sms);
    if (use_dg2_dense(dg2, shape_m, shape_n, shape_k, num_sms))
        return div_up(static_cast<int>(shape_m), static_cast<int>(dg2.block_m))
            * div_up(static_cast<int>(shape_n), static_cast<int>(dg2.block_n));
    auto [bm, bn, ns, mc, smem] = get_dense_gemm_config(shape_m, shape_n, shape_k, num_sms, false);
    (void) ns;
    (void) mc;
    (void) smem;
    return div_up(static_cast<int>(shape_m), bm) * div_up(static_cast<int>(shape_n), bn);
}

inline bool dense_main_path_is_v2(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    Dg2DenseConfig dg2 = get_dg2_dense_config_cached(shape_m, shape_n, shape_k, num_sms);
    return use_dg2_dense(dg2, shape_m, shape_n, shape_k, num_sms);
}

// The number of CTAs of the route gemm_dispatch_sm90 takes before use_dense_v2_override: swap-AB's tiles where
// use_dense_swap_ab takes the call, else the main path's (dense_main_path_tiles).
inline int dense_route_tiles(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms)
{
    if (use_dense_swap_ab(shape_m, shape_n, shape_k, num_sms))
    {
        auto const [bm, bn, ns, mc, smem] = get_dense_swap_ab_config(shape_m, shape_n, shape_k, num_sms);
        (void) ns;
        (void) mc;
        (void) smem;
        return div_up(static_cast<int>(shape_n), bm) * div_up(static_cast<int>(shape_m), bn);
    }
    return dense_main_path_tiles(shape_m, shape_n, shape_k, num_sms);
}

// DeepGEMM v2's configuration at a given tile and cluster (cluster_m pairs two M blocks, which share B; cluster_n two N
// blocks, which share A), with get_pipeline_config's stage count (dg2_num_stages) and swizzle, and N compiled into the
// kernel for a single-CTA 128-row tile wider than 128 (as get_dg2_dense_config does); false when fewer than 3 stages,
// or 4 below 128 x 192, fit.
inline bool dg2_tile_config(int block_m, int block_n, int cluster_m, int cluster_n, uint32_t shape_m, uint32_t shape_n,
    uint32_t shape_k, int num_sms, Dg2DenseConfig& out)
{
    uint32_t smem = 0;
    int const stages = dg2_num_stages(block_m, block_n, shape_k, smem);
    if (stages < 3 || (block_m * block_n < 128 * 192 && stages < 4))
        return false;
    Dg2DenseConfig c;
    c.block_m = static_cast<uint32_t>(block_m);
    c.block_n = static_cast<uint32_t>(block_n);
    c.cluster_m = static_cast<uint32_t>(cluster_m);
    c.cluster_n = static_cast<uint32_t>(cluster_n);
    c.num_stages = static_cast<uint32_t>(stages);
    c.smem_size = smem;
    c.swizzle_d = dg2_swizzle_mode(c.block_n, 2);
    c.num_sms = static_cast<uint32_t>(num_sms);
    c.num_waves = div_up(
        div_up(static_cast<int>(shape_m), block_m) * div_up(static_cast<int>(shape_n), block_n), num_sms);
    c.shape_n = (c.block_m == 128 && c.block_n > 128 && c.cluster_size() == 1) ? shape_n : 0u;
    out = c;
    return true;
}

// Whether DeepGEMM v2's 1D2D kernel keeps its main loop compact for a tile `block_n` wide. A width that divides 128
// (one weight scale per k-block), or whose offsets into the 128-row weight-scale blocks take more than four values
// (128 / gcd(128, width) > 4: 48, 80, 112, 144, ...), runs one loop with the split point as a variable. The others get
// one copy of the loop per split point (kShouldOptimize in impls/sm90_fp8_gemm_1d2d.cuh): five for 96 and 160, three for
// 192. Unrolled eight times, as DeepGEMM does, a 160-wide kernel held about 7,400 SASS instructions against 2,700 for a
// 144-wide one (runs/FH1/runs/dg2_sass). Since issue #12 (FH6) the five copies of the 160-wide loop unroll four times
// (kWideFiveCopies), about 4,500 instructions, which counts as compact here; the 96-wide tile keeps DeepGEMM's unroll
// (shorter ones were slower at long K) and the 192-wide one holds about 8,000, so neither does.
inline bool dg2_compact_main_loop(int block_n)
{
    int a = 128, b = block_n;
    while (b != 0)
    {
        int const t = a % b;
        a = b;
        b = t;
    }
    return 128 % block_n == 0 || 128 / a > 4 || (128 / a == 4 && block_n > 128);
}

// The single-CTA v2 tile with the fewest rows (16 at M <= 16, 32 at M <= 32, then 64, then 128) and, for those rows,
// the narrowest width (16 to 192 in steps of 16, with the 1D2D kernel's weight-scale split: a width above 128 must
// have (width - 128) dividing the width or 128) whose launch runs the whole GEMM in one wave on num_sms. Up to
// M = 128 only widths with a compact main loop (dg2_compact_main_loop) qualify, which since FH6 includes the 160-wide
// tile. There the one-wave tile saves the route's partly filled second wave, at most 2 us per call (19456 x 2560,
// served), while behind another GEMM the copies of the main loop cost a 160-wide kernel 3-4 us per call: in the
// Qwen3-4B MLP block its gate_up ran 3.4 us longer than alone, next to any other GEMM (fso's swap-AB, fp8_gemm_kernel,
// another v2 tile, cuBLAS) but not next to a memcpy, the quantize or the SwiGLU kernel (runs/FH1/runs/mlp_decomp), so
// the block got 4-8 % slower with five copies unrolled eight times; unrolled four times (FH6) the same 160-wide gate_up
// left the block level with the route at M = 64 and made it 2-3 % faster at M = 1 and 128 (runs/FH6/runs/d3). The
// 144-wide tile of 18432 x 2560 made its block 1-6 % faster. Above M = 128 the tile replaces a full second wave
// (2.2-4.8 us served on 12352 x 2560).
inline bool dg2_one_wave_tile(uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms, Dg2DenseConfig& out)
{
    int const block_ms[] = {16, 32, 64, 128};
    for (int bm : block_ms)
    {
        if ((bm == 16 && shape_m > 16u) || (bm == 32 && shape_m > 32u))
            continue;
        for (int bn = 16; bn <= 192; bn += 16)
        {
            if (bn > 128 && (bn % (bn - 128) != 0 && 128 % (bn - 128) != 0))
                continue;
            if (shape_m <= 128u && !dg2_compact_main_loop(bn))
                continue;
            if (div_up(static_cast<int>(shape_m), bm) * div_up(static_cast<int>(shape_n), bn) > num_sms)
                continue;
            if (dg2_tile_config(bm, bn, 1, 1, shape_m, shape_n, shape_k, num_sms, out))
                return true;
        }
    }
    return false;
}

// Three overrides of the route by DeepGEMM v2 tiles (fso issue 12, 2026-11-05; /data/bench-runs/fso_issue12_20261028/
// runs/FH1). gemm_dispatch_sm90 asks this first, in its default mode; `route` below is what it would run otherwise
// (dense_route_tiles), and every tile computes each output element in one CTA in K order, so the bytes are those of
// the route (identical on every one of the 1,600 configuration-cells of the measurement):
//
// 1. One wave (M <= 256). When the route needs two or more waves and a single-CTA v2 tile (dg2_one_wave_tile) runs the
//    whole GEMM in one, run that tile. The route's last wave is then mostly empty: swap-AB's 128-row weight tiles give
//    ceil(N / 128) tiles, 144 on 18432 x 2560, 152 on 19456 x 2560, a second wave of 12-20 tiles; the 64- and 128-row
//    main-path tiles of gate_up (64 x 72, 128 x 72-80) and of qkv / in_proj at M = 129-256 (64 x 128 / 64 x 144 with
//    a cluster, 128 x 80 / 96 at M = 256) need 240-261 CTAs. With two 128-row M blocks of which the second holds at
//    most half its rows (128 < M <= 192), the two blocks run as a 2-CTA cluster sharing each weight tile (multicast on
//    B): measured as fast on the bare GEMM as the single-CTA form within 1-3 %, and 6-21 % faster than it in the
//    served call with PDL on, where the single-CTA v2 grid loses 1-7 us behind the activation quantize. Measured
//    against 0281c5f's route (runs/FH1/runs/final3_ab), bare GEMM / served call: 18432 x 2560 at M = 8-128 0.78-0.98 /
//    0.83-0.97 (144 wide); 10240 x 2560 and 12352 x 2560 at M = 136-192 (paired) 0.76-0.95 / 0.78-0.95, at M = 256
//    0.80-0.84 / 0.83-0.87; 9728 x 2560, 9216 x 2048 and 12288 x 2048 at M = 144-256 0.77-0.94 / 0.79-0.94.
//    19456 x 2560 takes its 160-wide one-wave tile at M = 1-128 as well, now that the kernel keeps that width's main
//    loop compact (see dg2_one_wave_tile).
// 2. Fewer waves for the wide outputs (256 < M <= 672). When the 128-row v2 tile 160 wide, or 192 wide where that needs
//    fewer waves, needs at least a quarter fewer waves than the route and still two or more (DeepGEMM never clusters a
//    one-wave launch, and none was measured), run it as a 1 x 2 cluster (two N blocks share each activation tile,
//    multicast on A). Above M = 256 the route's 128-row tiles (fp8_gemm_kernel's 128 x 96 / 128 x 128, three swap-AB
//    token tiles) run 3-7 waves on 12352 x 2560 and 18432 x 2560 with a last M block that is mostly empty; the
//    160-wide tile covers the same M in 2-5 waves. The single-CTA form of the same tile is 4-12 % faster on the bare
//    GEMM but loses 3-11 us per call in the served call with PDL on (the CTA placement of an early-launched
//    single-CTA grid, runs/FH3); the cluster form keeps that time. Measured at M = 264-672: 12352 x 2560 0.70-0.98 of
//    the route's bare GEMM and 0.71-0.97 of its served call, 18432 x 2560 0.63-0.99 and 0.67-0.98, 10240 x 2560
//    (M = 392-504) 0.72-0.91 and 0.74-0.90.
// 3. Paired 64-row tiles for the narrow outputs (256 < M <= 384, K >= 4096). On a narrow output (64-row swap-AB weight
//    tiles), when the 128-row tiling's last M block would hold at most 96 rows and the 64 x 128 tile runs the GEMM in
//    one wave on at least three quarters of the SMs (2112 <= N on 132 SMs at five 64-row blocks), run it as a 2 x 1
//    cluster (two M blocks share each weight tile, multicast on B). The route there is DeepGEMM's 64 x 112 pick (rule
//    5, M = 257-320), whose five 64-row blocks read every weight five times on a width that stages D with a 32-byte
//    swizzle, or fp8_gemm_kernel's 128 x 64 tile with its third block mostly empty (M = 337-352); the paired 64 x 128
//    tile reads the weights three times. Measured on 2560 x 4096 and 2560 x 9216 at M = 264-320 and 344-352: 0.82-0.92
//    of the route's bare GEMM and 0.83-0.91 of its served call. Where the route already is DeepGEMM's 64 x 128 pick
//    (M = 321-336) the pair was level on the bare GEMM and won only the served call with PDL on (held): no override.
inline bool use_dense_v2_override_uncached(
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms, Dg2DenseConfig& cfg)
{
    int const m = static_cast<int>(shape_m);
    int const n = static_cast<int>(shape_n);
    if (shape_m <= 256u)
    {
        if (div_up(dense_route_tiles(shape_m, shape_n, shape_k, num_sms), num_sms) < 2
            || !dg2_one_wave_tile(shape_m, shape_n, shape_k, num_sms, cfg))
            return false;
        if (cfg.block_m == 128 && shape_m > 128u && shape_m <= 192u)
        {
            cfg.cluster_m = 2;
            cfg.shape_n = 0u;
        }
        return true;
    }
    if (shape_m > 672u)
        return false;
    int const route_waves = div_up(dense_route_tiles(shape_m, shape_n, shape_k, num_sms), num_sms);
    int const weight_rows = std::get<0>(get_dense_gemm_config(shape_n, shape_m, shape_k, num_sms, true));
    if (weight_rows == 128)
    {
        int const waves160 = div_up(div_up(m, 128) * div_up(n, 160), num_sms);
        int const waves192 = div_up(div_up(m, 128) * div_up(n, 192), num_sms);
        int const bn = waves192 < waves160 ? 192 : 160;
        int const waves = std::min(waves160, waves192);
        // At least a quarter fewer waves: the 160-wide tile holds 25 % more of the output than the route's 128 x 128,
        // so one wave fewer out of five (19456 x 2560 at M = 512: 5 -> 4) saves nothing (level alone, 4 % slower in
        // the Qwen3-4B MLP block).
        return waves >= 2 && 4 * (route_waves - waves) >= route_waves
            && dg2_tile_config(128, bn, 1, 2, shape_m, shape_n, shape_k, num_sms, cfg);
    }
    int const tiles = div_up(m, 64) * div_up(n, 128);
    int const last = m % 128;
    if (shape_m > 384u || shape_k < 4096u || last < 1 || last > 96 || tiles > num_sms || 4 * tiles < 3 * num_sms)
        return false;
    // The route may already run the 64 x 128 tile on its own (DeepGEMM's pick, rule 5): the pair is no faster there.
    Dg2DenseConfig own = get_dg2_dense_config_cached(shape_m, shape_n, shape_k, num_sms);
    if (!use_dense_swap_ab(shape_m, shape_n, shape_k, num_sms) && use_dg2_dense(own, shape_m, shape_n, shape_k, num_sms)
        && own.block_m == 64 && own.block_n == 128)
        return false;
    return dg2_tile_config(64, 128, 2, 1, shape_m, shape_n, shape_k, num_sms, cfg);
}

// use_dense_v2_override_uncached behind a small per-thread direct-mapped cache keyed by the shape, so that an eager
// call pays one lookup instead of the route evaluation.
inline bool use_dense_v2_override(
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, int num_sms, Dg2DenseConfig& cfg)
{
    struct Entry
    {
        uint32_t m = 0, n = 0, k = 0;
        int sms = -1;
        bool take = false;
        Dg2DenseConfig cfg;
    };
    thread_local Entry table[256];
    uint64_t const h = (static_cast<uint64_t>(shape_m) * 0x9E3779B97F4A7C15ull) ^ (static_cast<uint64_t>(shape_n) << 20)
        ^ (static_cast<uint64_t>(shape_k) << 40) ^ static_cast<uint64_t>(num_sms);
    Entry& e = table[(h ^ (h >> 31)) & 255u];
    if (e.sms != num_sms || e.m != shape_m || e.n != shape_n || e.k != shape_k)
    {
        e.take = use_dense_v2_override_uncached(shape_m, shape_n, shape_k, num_sms, e.cfg);
        e.m = shape_m;
        e.n = shape_n;
        e.k = shape_k;
        e.sms = num_sms;
    }
    if (e.take)
        cfg = e.cfg;
    return e.take;
}
} // namespace deep_gemm::jit
