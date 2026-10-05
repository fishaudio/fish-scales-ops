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
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <vector>

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
      "build), or set FSO_JIT_NVRTC_LIB to a CUDA 13.2 libnvrtc.so.13.";

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
// other than 13.2 is the caller's explicit choice (an override, or a replaced bundled file): it is used, with one line
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
    if (api.major != 13 || api.minor != 2)
        std::fprintf(stderr,
            "[fish_scales_ops] sm_90 JIT: compiling with NVRTC %d.%d from %s (%s); the sm_90 kernels are validated "
            "and measured with NVRTC 13.2\n",
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
// get_best_gemm_config's pick, which every grouped (MoE) path keeps using unchanged, and applies two rules measured
// on the H200 (132 SMs) on 2026-10-05 against today's pick, with outputs bitwise identical for every tile (each output
// element is accumulated by one CTA in the same order whatever the tile):
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
//    weights keep 128 rows: 64-row tiles measured mixed at N = 5120-6144 and 3-34 % slower from N = 9216, where they
//    exceed one wave.
//
// 3. Main path, M <= 128 with a narrow pick. A 128-row tile covers all of M in one M-block, and when the picker
//    then chose block_n <= 32 (the narrow outputs, N up to about 4096) every CTA streams 128 rows of A for at most
//    32 rows of B and runs WGMMAs at most 32 wide. Two 64-row tiles of twice the width keep the CTA count and the
//    output area per CTA, cut the rows each CTA loads per k-block from 128 + block_n to 64 + 2 x block_n and double
//    the WGMMA width (0.95-0.97 of the time with the 128-row tile on the five family cells where the rule applies,
//    M = 128; multicast stays off at these M).
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
} // namespace deep_gemm::jit
