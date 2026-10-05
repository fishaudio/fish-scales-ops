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
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "jit_utils.cuh"
#include "nvrtc.h" // types and enums only: the functions are called through the table of jit_utils.cuh
#include "runtime.cuh"
#include "scheduler.cuh"

#ifdef _WIN32
#include <windows.h>
#endif

namespace deep_gemm::jit
{

// Generate a unique ID for temporary directories to avoid collisions
std::string generateUniqueId()
{
    // Use current time and random number to generate a unique ID
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> distrib(0, 999999);

    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch().count();

    // Use the static random generator
    int random_value = distrib(gen);

    return std::to_string(value) + "_" + std::to_string(random_value);
}

// Root of the on-disk JIT cache, which only FSO_JIT_DUMP_CUBIN and FSO_JIT_USE_NVCC use (the default NVRTC build
// lives in memory and never touches the file system). A cubin lives in
// <root>/cache/<16-hex content key>_<kernel name>/<kKernelName> and is written through <root>/tmp. The root is, in
// order: FSO_JIT_CACHE_DIR; the deprecated TensorRT-LLM alias TRTLLM_DG_CACHE_DIR; ${XDG_CACHE_HOME:-$HOME/.cache}/
// fish_scales_ops/jit (%LOCALAPPDATA%\fish_scales_ops\jit on Windows); <temp dir>/fish_scales_ops/jit. Resolved once,
// on first use; nothing is created here (directories are made when a cubin is written).
inline std::filesystem::path const& getJitCacheRoot()
{
    static std::filesystem::path const root = []() -> std::filesystem::path
    {
        auto isSet = [](char const* value) { return value != nullptr && value[0] != '\0'; };
        if (char const* dir = getJitEnv("FSO_JIT_CACHE_DIR", "TRTLLM_DG_CACHE_DIR"); isSet(dir))
            return std::filesystem::path(dir);
        std::filesystem::path const leaf = std::filesystem::path("fish_scales_ops") / "jit";
#ifdef _WIN32
        if (char const* localAppData = std::getenv("LOCALAPPDATA"); isSet(localAppData))
            return std::filesystem::path(localAppData) / leaf;
        if (char const* appData = std::getenv("APPDATA"); isSet(appData))
            return std::filesystem::path(appData) / leaf;
#else
        // XDG base-directory rule: an unset, empty or relative XDG_CACHE_HOME means $HOME/.cache.
        if (char const* xdg = std::getenv("XDG_CACHE_HOME"); isSet(xdg) && std::filesystem::path(xdg).is_absolute())
            return std::filesystem::path(xdg) / leaf;
        if (char const* home = std::getenv("HOME"); isSet(home))
            return std::filesystem::path(home) / ".cache" / leaf;
#endif
        std::error_code ec;
        std::filesystem::path tmp = std::filesystem::temp_directory_path(ec);
        if (ec || tmp.empty())
            tmp = std::filesystem::path(".");
        return tmp / leaf;
    }();
    return root;
}

inline std::filesystem::path getTmpDir()
{
    return getJitCacheRoot() / "tmp";
}

inline std::filesystem::path getCacheDir()
{
    return getJitCacheRoot() / "cache";
}

std::string getNvccCompiler()
{
    static std::string compiler;
    if (compiler.empty())
    {
        // FSO_JIT_NVCC_COMPILER (deprecated alias TRTLLM_DG_NVCC_COMPILER), else $CUDA_HOME/bin/nvcc, else nvcc on PATH
        char const* envCompiler = getJitEnv("FSO_JIT_NVCC_COMPILER", "TRTLLM_DG_NVCC_COMPILER");
        if (envCompiler && envCompiler[0] != '\0')
        {
            compiler = envCompiler;
        }
        else
        {
            // Check CUDA_HOME
            char const* cudaHome = getenv("CUDA_HOME");
            if (cudaHome)
            {
                std::filesystem::path cudaPath(cudaHome);
#ifdef _WIN32
                compiler = (cudaPath / "bin" / "nvcc.exe").string();
#else
                compiler = (cudaPath / "bin" / "nvcc").string();
#endif
            }
            else
            {
// Default to system nvcc
#ifdef _WIN32
                compiler = "nvcc.exe";
#else
                compiler = "nvcc";
#endif
            }
        }
    }
    return compiler;
}

std::vector<std::filesystem::path> getJitIncludeDirs()
{
    static std::vector<std::filesystem::path> includeDirs;
    if (includeDirs.empty())
    {
        // The include directories of the JIT, the first of these that names any:
        //   1. FSO_JIT_INCLUDE_DIRS, a colon-separated list;
        //   2. the packaged tree <directory of the extension .so>/_jit_include, when it exists (bundledJitIncludePath
        //      in jit_utils.cuh): a wheel carries every header the JIT includes there, in one include root;
        //   3. the build-time default FSO_JIT_INCLUDE_DIRS_DEFAULT, the source-tree directories that python/setup.py
        //      bakes in, which an in-place build uses.
        auto pushFromColonList = [&](char const* list)
        {
            if (list == nullptr)
                return;
            std::string s(list);
            size_t start = 0;
            while (start <= s.size())
            {
                size_t end = s.find(':', start);
                std::string token = (end == std::string::npos) ? s.substr(start) : s.substr(start, end - start);
                if (!token.empty())
                    includeDirs.emplace_back(token);
                if (end == std::string::npos)
                    break;
                start = end + 1;
            }
        };
        char const* origin = "FSO_JIT_INCLUDE_DIRS";
        pushFromColonList(std::getenv("FSO_JIT_INCLUDE_DIRS"));
        if (includeDirs.empty())
        {
            std::filesystem::path const packaged = bundledJitIncludePath();
            std::error_code ec;
            if (!packaged.empty() && std::filesystem::is_directory(packaged, ec))
            {
                includeDirs.push_back(packaged);
                origin = "the include tree packaged with fish_scales_ops";
            }
        }
#ifdef FSO_JIT_INCLUDE_DIRS_DEFAULT
        if (includeDirs.empty())
        {
            pushFromColonList(FSO_JIT_INCLUDE_DIRS_DEFAULT);
            origin = "the build-time default";
        }
#endif
        if (!includeDirs.empty())
        {
            if (kJitDebugging)
            {
                std::string list;
                for (auto const& dir : includeDirs)
                    list += (list.empty() ? "" : ":") + dir.string();
                TLLM_LOG_INFO("sm_90 JIT include directories (%s): %s", origin, list.c_str());
            }
            // The baked default names directories of the source tree that built the extension
            // (python/setup.py). An install that moved or deleted that tree leaves NVRTC without
            // the deep_gemm and CUDA headers, and every sm_90 kernel compile then fails with a
            // bare "cannot open source file". Name the missing directories and the remedy once per
            // process (this list is resolved once).
            std::string missing;
            for (auto const& dir : includeDirs)
            {
                std::error_code ec;
                if (!std::filesystem::is_directory(dir, ec))
                    missing += "\n  " + dir.string();
            }
            if (!missing.empty())
                std::fprintf(stderr,
                    "[fish_scales_ops] sm_90 JIT: these include directories (%s) do not exist:%s\n"
                    "  The sm_90 kernels are compiled at run time. An in-place build takes their headers from the "
                    "source tree that built it, so keep that tree in place; or install a wheel built by "
                    "scripts/build_wheel.sh, which carries them in _jit_include/ next to the extension; or set "
                    "FSO_JIT_INCLUDE_DIRS to a colon-separated list of the directory that holds deep_gemm/ "
                    "(csrc/gemm/include/blockscale_gemm/arch/sm90/fp8/jit in the source tree), the CUDA include "
                    "directory and its cccl/ subdirectory, and the CUTLASS include directory.\n",
                    origin, missing.c_str());
            return includeDirs;
        }

        // Command to execute
        char const* cmd = "pip show tensorrt_llm 2>/dev/null";

        // Buffer to store the output
        std::array<char, 128> buffer;
        std::string result;

// Open pipe to command
#ifdef _MSC_VER
        FILE* pipe = _popen(cmd, "r");
#else
        FILE* pipe = popen(cmd, "r");
#endif

        if (pipe)
        {
            // Read the output
            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
            {
                result += buffer.data();
            }

// Close the pipe
#ifdef _MSC_VER
            _pclose(pipe);
#else
            pclose(pipe);
#endif

            // Parse the location using regex
            // `pip show tensorrt_llm` will output something like:
            // Location: /usr/local/lib/python3.12/dist-packages
            // Editable project location: /code
            std::regex locationRegex("(Location|Editable project location): (.+)");

            // Find all matches
            auto match_begin = std::sregex_iterator(result.begin(), result.end(), locationRegex);
            auto match_end = std::sregex_iterator();

            // Get the number of matches
            auto match_count = std::distance(match_begin, match_end);

            if (match_count > 0)
            {
                // Get the last match
                auto last_match_iter = match_begin;
                std::advance(last_match_iter, match_count - 1);

                // Get the path from the second capture group
                std::string location = last_match_iter->str(2);
                location.erase(location.find_last_not_of(" \n\r\t") + 1);

                // Set the include directory based on the package location
                includeDirs.push_back(std::filesystem::path(location) / "tensorrt_llm" / "include");

                if (!kJitUseNvcc)
                {
                    includeDirs.push_back(
                        std::filesystem::path(location) / "tensorrt_llm" / "include" / "cuda" / "include");
                }
            }
        }
        else
        {
            TLLM_LOG_WARNING("Failed to find TensorRT LLM installation, DeepGEMM will be disabled.");
        }
    }
    return includeDirs;
}

// two_wg: the non-swap GroupedContiguous FC1 with two math warp-groups split along N (fp8_gemm_kernel_2wg). Its
// scheduler enumerates shape_n / (2 * block_n) N-blocks (one per SwiGLU column block), passed explicitly as
// SchedulerSelector's kNumNBlocks; every other variant emits exactly the source it emitted before the flag existed.
// swiglu (requires two_wg): the same FC1 with the SwiGLU + 1x128 FP8 requantize fused into its epilogue
// (fp8_gemm_kernel_2wg_swiglu, fused-FC1 step 2); same scheduler, only the kernel name differs.
// swapab_pair (requires swapAB, GroupedContiguous): the swap-AB FC1 in which one CTA owns gate block b and up block b
// (256 weight rows) of one activation tile (fused-FC1 P2a design S2): fp8_gemm_kernel_swapAB_pair (bf16 gu, the
// mainloop's test vehicle), or with swiglu fp8_gemm_kernel_swapAB_swiglu (the SwiGLU + 1x128 FP8 requantize fused into
// the epilogue). Its scheduler enumerates shape_n / (2 * block_m) weight blocks (one per SwiGLU column block), passed
// explicitly as SchedulerSelectorSwapAB's kNumMBlocks.
// swapab_split (requires swapab_pair and swiglu): fp8_gemm_kernel_swapAB_swiglu_split, the same fused FC1 with each
// SwiGLU block shared by the two CTAs of a cluster (H2 Phase 2a); same scheduler, only the kernel name differs.
std::string generateKernel(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m,
    uint32_t const block_n, uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages,
    uint32_t const num_tma_multicast, deep_gemm::GemmType const gemm_type, bool swapAB = false, bool two_wg = false,
    bool swiglu = false, bool swapab_pair = false, bool swapab_split = false)
{
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;

    std::string input_type;
    if (!swapAB)
    {
        switch (gemm_type)
        {
        case deep_gemm::GemmType::Normal: input_type = "NormalSchedulerInput"; break;
        case deep_gemm::GemmType::GroupedContiguous: input_type = "GroupedContiguousSchedulerInput"; break;
        case deep_gemm::GemmType::GroupedMasked: input_type = "GroupedMaskedSchedulerInput"; break;
        case deep_gemm::GemmType::GroupedWithOffset: input_type = "GroupedWithOffsetSchedulerInput"; break;
        case deep_gemm::GemmType::StridedBatched: input_type = "StridedBatchedSchedulerInput"; break;
        default: throw std::runtime_error("Unsupported gemm type");
        }
    }
    else
    {
        switch (gemm_type)
        {
        case deep_gemm::GemmType::Normal: input_type = "NormalSchedulerInputSwapAB"; break;
        case deep_gemm::GemmType::GroupedWithOffset: input_type = "GroupedWithOffsetSchedulerInputSwapAB"; break;
        case deep_gemm::GemmType::GroupedContiguous: input_type = "GroupedContiguousSchedulerInputSwapAB"; break;
        default: throw std::runtime_error("Unsupported gemm type");
        }
    }

    if (two_wg && (swapAB || gemm_type != deep_gemm::GemmType::GroupedContiguous))
        throw std::runtime_error("two-warp-group FC1 kernel: non-swap GroupedContiguous only");
    if (swiglu && !two_wg && !swapab_pair)
        throw std::runtime_error("fused SwiGLU FC1 kernel: only as the two-warp-group FC1 or the swap-AB gate/up pair FC1");
    if (swapab_pair && (!swapAB || two_wg || gemm_type != deep_gemm::GemmType::GroupedContiguous))
        throw std::runtime_error("swap-AB gate/up pair FC1 kernel: swap-AB GroupedContiguous only");
    if (swapab_split && !(swapab_pair && swiglu))
        throw std::runtime_error("cluster-split swap-AB FC1 kernel: only as the fused swap-AB gate/up FC1");

    // Modify kernel name based on swapAB to determine which kernel function to use
    std::string const swiglu_pair_kernel
        = swapab_split ? "fp8_gemm_kernel_swapAB_swiglu_split" : "fp8_gemm_kernel_swapAB_swiglu";
    std::string kernel_name = swapAB
        ? (swapab_pair ? (swiglu ? swiglu_pair_kernel : std::string("fp8_gemm_kernel_swapAB_pair"))
                       : std::string("fp8_gemm_kernel_swapAB"))
        : (two_wg ? (swiglu ? "fp8_gemm_kernel_2wg_swiglu" : "fp8_gemm_kernel_2wg") : "fp8_gemm_kernel");
    std::string scheduler_name = swapAB ? "SchedulerSelectorSwapAB" : "SchedulerSelector";
    std::string const scheduler_n_blocks = two_wg ? ", " + std::to_string(shape_n / (2 * block_n))
                                                  : (swapab_pair ? ", " + std::to_string(shape_n / (2 * block_m))
                                                                 : std::string());

    // Create the kernel source code using raw string literal
    std::string code = R"(
#ifdef __CUDACC_RTC__
#ifndef NVRTC_JIT_COMPILATION
#define NVRTC_JIT_COMPILATION
#endif

#include <deep_gemm/nvrtc_std.cuh>

#else

#include <string>
#include <cuda.h>

#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <deep_gemm/nvrtc_cutlass.cuh>
#include <deep_gemm/fp8_gemm_impl.cuh>

using namespace deep_gemm;

using SchedulerType =
typename )"
        + scheduler_name + R"(<GemmType::)" + gemm_type_to_string(gemm_type) + R"(, )" + std::to_string(shape_n)
        + R"(, )" + std::to_string(shape_k) + R"(, )" + std::to_string(block_m) + R"(, )" + std::to_string(block_n)
        + R"(, )" + std::to_string(block_k) + R"(, )" + std::to_string(num_groups) + R"(, )"
        + std::to_string(num_tma_multicast) + scheduler_n_blocks + R"(>::type;

__global__ void dummy_kernel() {
  void *ptr = (void *)&)"
        + kernel_name + R"(<)" + std::to_string(shape_n) + R"(, )" + std::to_string(shape_k) + R"(, )"
        + std::to_string(block_m) + R"(, )" + std::to_string(block_n) + R"(, )" + std::to_string(block_k) + R"(, )"
        + std::to_string(num_groups) + R"(, )" + std::to_string(num_stages) + R"(, )" + std::to_string(kNumTMAThreads)
        + R"(, )" + std::to_string(kNumMathThreadsPerGroup) + R"(, )" + std::to_string(num_tma_multicast)
        + R"(, SchedulerType, )" + input_type + R"(>;
}
)";

    return code;
}

// FNV-1a, 64-bit. The disk cache's content key must come out the same in every process and every build of the
// extension, which std::hash does not promise; FNV-1a is fixed by its definition. Integers are fed as 8 little-endian
// bytes and strings with a length prefix, so two different field sequences never produce the same byte stream.
class Fnv1a64
{
public:
    void bytes(void const* data, size_t size)
    {
        auto const* p = static_cast<unsigned char const*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            state_ ^= p[i];
            state_ *= 0x100000001b3ull;
        }
    }

    void u64(uint64_t value)
    {
        unsigned char le[8];
        for (int i = 0; i < 8; ++i)
            le[i] = static_cast<unsigned char>(value >> (8 * i));
        bytes(le, sizeof(le));
    }

    void field(std::string const& s)
    {
        u64(s.size());
        bytes(s.data(), s.size());
    }

    [[nodiscard]] uint64_t value() const
    {
        return state_;
    }

    [[nodiscard]] std::string hex() const
    {
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(state_));
        return std::string(buf);
    }

private:
    uint64_t state_ = 0xcbf29ce484222325ull;
};

// The deep_gemm header directory a build compiles against and a digest of its contents. The directory is the first
// include directory holding deep_gemm/fp8_gemm_impl.cuh, which is where NVRTC resolves the generated source's
// <deep_gemm/...> includes (it searches the -I directories in order); the digest is an FNV-1a 64 over every regular
// file under it, visited in sorted relative-path order, each file contributing its relative path and its contents.
// `found` is false when no include directory holds the kernel header or a file cannot be read.
struct JitHeaderDigest
{
    std::filesystem::path dir;
    uint64_t hash = 0;
    size_t numFiles = 0;
    bool found = false;
};

inline JitHeaderDigest computeJitHeaderDigest(std::vector<std::filesystem::path> const& includeDirs)
{
    JitHeaderDigest digest;
    for (auto const& inc : includeDirs)
    {
        std::error_code ec;
        if (std::filesystem::is_regular_file(inc / "deep_gemm" / "fp8_gemm_impl.cuh", ec))
        {
            digest.dir = inc / "deep_gemm";
            break;
        }
    }
    if (digest.dir.empty())
        return digest;

    std::vector<std::pair<std::string, std::filesystem::path>> files;
    std::error_code ec;
    for (std::filesystem::recursive_directory_iterator it(digest.dir, ec), end; !ec && it != end; it.increment(ec))
    {
        std::error_code typeEc;
        if (it->is_regular_file(typeEc))
            files.emplace_back(it->path().lexically_relative(digest.dir).generic_string(), it->path());
    }
    if (ec)
        return digest;
    std::sort(files.begin(), files.end());

    Fnv1a64 h;
    h.u64(files.size());
    for (auto const& [rel, file] : files)
    {
        std::ifstream in(file, std::ios::binary);
        if (!in.is_open())
            return digest;
        std::string const contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        h.field(rel);
        h.field(contents);
    }
    digest.hash = h.value();
    digest.numFiles = files.size();
    digest.found = true;
    return digest;
}

// The NVRTC function table of this process (jit_utils.cuh): the library is loaded on the first call, which is the
// first NVRTC compile or disk-cache key of the process, and kept for the life of the process. FSO_JIT_USE_NVCC never
// reaches it.
inline NvrtcApi const& nvrtc()
{
    static NvrtcApi const api = loadNvrtcApi(kJitDebugging);
    return api;
}

// The compiler part of the content key: the version of the NVRTC library this process loaded, or the nvcc command
// (FSO_JIT_USE_NVCC).
inline std::string jitCompilerIdentity()
{
    if (kJitUseNvcc)
        return "nvcc " + getNvccCompiler();
    NvrtcApi const& api = nvrtc();
    return "nvrtc " + std::to_string(api.major) + "." + std::to_string(api.minor);
}

// The compiler of the sm_90 JIT in this process, as torch.ops.fish_scales_ops.jit_compiler_sm90 reports it:
// "NVRTC <major>.<minor> (<absolute path of the library>)", or "nvcc <path>" under FSO_JIT_USE_NVCC. It loads NVRTC on
// first use, so where the library cannot be loaded it throws the error the first sm_90 GEMM would.
inline std::string jitCompilerDescription()
{
    if (kJitUseNvcc)
        return "nvcc " + getNvccCompiler();
    NvrtcApi const& api = nvrtc();
    return "NVRTC " + std::to_string(api.major) + "." + std::to_string(api.minor) + " (" + api.path + ")";
}

/**
 * C++ implementation of the Compiler class
 * Compiles CUDA code into CUBINs
 */
class Compiler
{
public:
    // Get singleton instance
    static Compiler& getInstance()
    {
        static Compiler instance;
        return instance;
    }

    [[nodiscard]] bool isValid() const
    {
        return !includeDirs_.empty();
    }

    // Build function
    Runtime* build(uint32_t const shape_n, uint32_t const shape_k, uint32_t const block_m, uint32_t const block_n,
        uint32_t const block_k, uint32_t const num_groups, uint32_t const num_stages, uint32_t const num_tma_multicast,
        deep_gemm::GemmType const gemm_type, bool swapAB = false, uint32_t const ctas_per_sm = 1, bool two_wg = false,
        bool swiglu = false, bool swapab_pair = false, bool swapab_split = false)
    {
        int sm_version = tensorrt_llm::common::getSMVersion();
        if (sm_version != 90)
        {
            TLLM_THROW(
                "DeepGEMM only supports Hopper (SM90) architectures, but current device compute "
                "capability is %d.",
                sm_version);
        }

        // Kernel name, from the template arguments alone. It keys the in-memory runtime cache (the per-launch hit path
        // just below) and is the readable half of a disk-cache directory name; everything else that shapes the cubin
        // is covered by the disk cache's content key (contentKey). The two-warp-group FC1 variants are marked in the
        // prefix ("gemm_2wg_", "gemm_2wg_swiglu_"), the swap-AB gate/up pair FC1s as "gemm_swapAB_pair_",
        // "gemm_swapAB_swiglu_" and (the cluster-split one) "gemm_swapAB_swiglu_split_". The doubled num_groups /
        // num_stages segment is historical and kept so names stay stable. The name does not have to end in the GEMM type: a disk load takes the type from this
        // call, not from the path.
        char const* const swiglu_pair_prefix = swapab_split ? "gemm_swapAB_swiglu_split_" : "gemm_swapAB_swiglu_";
        std::string name
            = std::string(swapAB ? (swapab_pair ? (swiglu ? swiglu_pair_prefix : "gemm_swapAB_pair_") : "gemm_swapAB_")
                                 : (two_wg ? (swiglu ? "gemm_2wg_swiglu_" : "gemm_2wg_") : "gemm_"))
            + std::to_string(shape_n) + "_"
            + std::to_string(shape_k) + "_" + std::to_string(block_m) + "_" + std::to_string(block_n) + "_"
            + std::to_string(block_k) + "_" + std::to_string(num_groups) + "_" + std::to_string(num_stages)
            + std::to_string(num_groups) + "_" + std::to_string(num_stages) + "_" + std::to_string(num_tma_multicast)
            + "_" + gemm_type_to_string(gemm_type) + (ctas_per_sm > 1 ? "_c" + std::to_string(ctas_per_sm) : "");

        // Hit path, taken by every sm_90 GEMM launch after the first one of its kernel in this process: the name
        // above and one in-memory lookup. No hashing, file-system access or environment read happens here; the rest
        // of this function runs once per kernel per process.
        auto& runtimeCache = getGlobalRuntimeCache();
        if (Runtime* cachedRuntime = runtimeCache.find(name))
        {
            if (kJitDebugging)
            {
                TLLM_LOG_INFO("Using cached JIT runtime %s during build", name.c_str());
            }
            return cachedRuntime;
        }

        // Compiler flags
        std::vector<std::string> flags
            = {"-std=c++17", "--gpu-architecture=sm_90a", "--ptxas-options=-allow-expensive-optimizations=true",
                "--ptxas-options=--register-usage-level=10", "--diag-suppress=161,174,177,940",
                "-D__FORCE_INCLUDE_CUDA_FP16_HPP_FROM_FP16_H__=1", "-D__FORCE_INCLUDE_CUDA_BF16_HPP_FROM_BF16_H__=1"};
        if (ctas_per_sm > 1)
            flags.push_back("-DFSO_SWAPAB_CTAS_PER_SM=" + std::to_string(ctas_per_sm));
        // Developer passthrough: FSO_JIT_EXTRA_FLAGS="-DFOO=1 -DBAR=0" appends to every JIT compile. Like every other
        // flag they are part of the disk cache's content key, so a cubin built under different extra flags is never
        // loaded in their place.
        if (char const* extra = std::getenv("FSO_JIT_EXTRA_FLAGS"))
        {
            std::string tok;
            for (char const* c = extra;; ++c)
            {
                if (*c == ' ' || *c == '\0')
                {
                    if (!tok.empty())
                        flags.push_back(tok);
                    tok.clear();
                    if (*c == '\0')
                        break;
                }
                else
                    tok.push_back(*c);
            }
        }

        if (kJitUseNvcc)
        {
            flags.push_back("-O3");
            flags.push_back("-cubin");
            flags.push_back("--expt-relaxed-constexpr");
            flags.push_back("--expt-extended-lambda");

            std::vector<std::string> cxxFlags = {"-fPIC", "-O3", "-Wno-deprecated-declarations", "-Wno-abi"};
            std::string cxxFlagsStr = "--compiler-options=";
            for (size_t i = 0; i < cxxFlags.size(); ++i)
            {
                cxxFlagsStr += cxxFlags[i];
                if (i < cxxFlags.size() - 1)
                {
                    cxxFlagsStr += ",";
                }
            }
            flags.push_back(cxxFlagsStr);
        }
        else
        {
            flags.push_back("-default-device");
        }

        for (auto const& dir : includeDirs_)
        {
            flags.push_back("-I" + dir.string());
        }

        std::string code = generateKernel(shape_n, shape_k, block_m, block_n, block_k, num_groups, num_stages,
            num_tma_multicast, gemm_type, swapAB, two_wg, swiglu, swapab_pair, swapab_split);

        // Disk cache, used only with FSO_JIT_DUMP_CUBIN or FSO_JIT_USE_NVCC (the default NVRTC build stays in memory
        // and never touches the file system). A cubin lives in <cache>/<content key>_<name>/, the key covering the
        // generated source, the complete flag list, the compiler and every JIT header file, so a cubin found there
        // was built from exactly this input and is loaded instead of compiling. Directories without the key
        // (TensorRT-LLM's DeepGEMM cache layout, dumps of older fish_scales_ops builds) are never looked at.
        bool const useDiskCache = kJitUseNvcc || kJitDumpCubin;
        std::filesystem::path path;
        if (useDiskCache)
        {
            // Resolve the cache root first: a deprecated-alias notice must not land inside a log line below.
            std::filesystem::path const cacheDir = getCacheDir();
            JitHeaderDigest const& headers = headerDigest();
            if (!headers.found)
            {
                TLLM_THROW(
                    "deep_gemm JIT: could not read deep_gemm/fp8_gemm_impl.cuh and its directory under the JIT include "
                    "directories (FSO_JIT_INCLUDE_DIRS or the build-time default), so the disk cache "
                    "(FSO_JIT_DUMP_CUBIN / FSO_JIT_USE_NVCC) cannot key %s",
                    name.c_str());
            }
            path = cacheDir / (contentKey(code, flags, headers) + "_" + name);
            if (Runtime::isPathValid(path.string()))
            {
                if (kJitDebugging)
                {
                    TLLM_LOG_INFO("Loaded JIT runtime %s from the disk cache: %s", name.c_str(), path.string().c_str());
                }
                auto runtime = std::make_unique<Runtime>(path.string(), std::vector<char>(), gemm_type);
                Runtime* result = runtime.get();
                runtimeCache.set(name, std::move(runtime));
                return result;
            }
        }

        // Print options if debug enabled
        if (kJitDebugging)
        {
            TLLM_LOG_INFO("Compiling JIT runtime %s with options: ", name.c_str());
            for (auto const& flag : flags)
            {
                TLLM_LOG_INFO("%s ", flag.c_str());
            }
            TLLM_LOG_INFO("\n");
            TLLM_LOG_INFO("Generated kernel code:\n%s", code.c_str());
        }

        // A disk-cache write goes through a private temporary directory and one rename, so a concurrent reader never
        // sees a partial cubin.
        std::filesystem::path tmpPath;
        if (useDiskCache)
        {
            tmpPath = getTmpDir() / (path.filename().string() + "_" + generateUniqueId());
        }
        std::filesystem::path const tmpCubinPath = tmpPath / kKernelName;

        std::vector<char> cubin;
        bool staged = false; // the cubin sits at tmpCubinPath, ready to be published
        if (kJitUseNvcc)
        {
            std::filesystem::create_directories(tmpPath);
            std::filesystem::path tmpSrcPath = tmpPath / "kernel.cu";

            // Write files
            std::ofstream srcFile(tmpSrcPath);
            srcFile << code;
            srcFile.close();

            // Build command
            std::vector<std::string> command = {getNvccCompiler(), tmpSrcPath.string(), "-o", tmpCubinPath.string()};
            command.insert(command.end(), flags.begin(), flags.end());

            // Execute command
            std::string cmd;
            for (auto const& arg : command)
            {
                cmd += arg + " ";
            }

            // Buffer to store the output
            std::array<char, 128> buffer;
            std::string result;
            int status = -1;

            // Time the compilation
            auto start = std::chrono::high_resolution_clock::now();

            // Open pipe to command
#ifdef _MSC_VER
            FILE* pipe = _popen(cmd.c_str(), "r");
#else
            FILE* pipe = popen(cmd.c_str(), "r");
#endif

            if (pipe)
            {
                // Read the output
                while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
                {
                    result += buffer.data();
                }

// Close the pipe
#ifdef _MSC_VER
                status = _pclose(pipe);
#else
                status = pclose(pipe);
#endif

                // Output result if debug enabled
                if (kJitDebugging)
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    TLLM_LOG_INFO("NVCC compilation took %d ms", duration.count());
                    TLLM_LOG_INFO("Compilation log:\n%s", result.c_str());
                }
            }

            // Only a cubin from a successful nvcc run may reach the disk cache, which later processes trust.
            std::ifstream cubinFile(tmpCubinPath, std::ios::binary);
            if (status == 0 && cubinFile.is_open())
            {
                cubin.assign(std::istreambuf_iterator<char>(cubinFile), std::istreambuf_iterator<char>());
            }
            cubinFile.close();
            if (cubin.empty())
            {
                std::error_code ec;
                std::filesystem::remove_all(tmpPath, ec);
                TLLM_THROW("deep_gemm JIT: nvcc failed to compile %s (status %d): %s", name.c_str(), status,
                    result.c_str());
            }
            staged = true;
        }
        else
        {
            // Every NVRTC call goes through the table of the privately loaded library (jit_utils.cuh).
            NvrtcApi const& api = nvrtc();
            nvrtcProgram prog;
            CHECK_NVRTC(api.nvrtcCreateProgram(&prog, code.c_str(), "kernel.cu", 0, nullptr, nullptr));

            std::vector<char const*> options;
            for (auto const& flag : flags)
            {
                options.push_back(flag.c_str());
            }

            // Time the compilation
            auto start = std::chrono::high_resolution_clock::now();
            nvrtcResult compileResult = api.nvrtcCompileProgram(prog, options.size(), options.data());

            if (kJitDebugging)
            {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                TLLM_LOG_INFO("NVRTC compilation took %d ms", duration.count());

                size_t logSize;
                CHECK_NVRTC(api.nvrtcGetProgramLogSize(prog, &logSize));
                std::vector<char> log(logSize);
                CHECK_NVRTC(api.nvrtcGetProgramLog(prog, log.data()));
                TLLM_LOG_INFO("Compilation log:\n%s", log.data());
            }

            // Check if compilation succeeded
            if (compileResult != NVRTC_SUCCESS)
            {
                TLLM_LOG_ERROR("NVRTC compilation failed");
                CHECK_NVRTC(api.nvrtcDestroyProgram(&prog));
                throw std::runtime_error("NVRTC compilation failed");
            }

            size_t cubinSize;
            CHECK_NVRTC(api.nvrtcGetCUBINSize(prog, &cubinSize));
            cubin.resize(cubinSize);
            CHECK_NVRTC(api.nvrtcGetCUBIN(prog, cubin.data()));
            CHECK_NVRTC(api.nvrtcDestroyProgram(&prog));

            // FSO_JIT_DUMP_CUBIN: stage the cubin for the disk cache
            if (kJitDumpCubin)
            {
                try
                {
                    std::filesystem::create_directories(tmpPath);
                    std::ofstream cubinFile(tmpCubinPath, std::ios::binary);
                    cubinFile.write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
                    cubinFile.close();
                    if (!cubinFile)
                        throw std::runtime_error("cannot write " + tmpCubinPath.string());
                    staged = true;
                }
                catch (std::exception const& e)
                {
                    TLLM_LOG_ERROR("Warning: Failed to stage the cubin for the disk cache: %s", e.what());
                }
            }
        }

        // Publish the staged cubin into its content-keyed directory (rename: atomic) and drop the temporary
        // directory. A failure costs only the disk copy: the runtime below is built from the cubin in memory.
        if (staged)
        {
            try
            {
                std::filesystem::create_directories(path);
                std::filesystem::rename(tmpCubinPath, path / kKernelName);
                if (kJitDebugging)
                {
                    TLLM_LOG_INFO("Wrote JIT runtime %s to the disk cache: %s", name.c_str(), path.string().c_str());
                }
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Warning: Failed to copy kernel files to cache: %s", e.what());
            }
        }
        if (useDiskCache)
        {
            try
            {
                std::filesystem::remove_all(tmpPath);
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_ERROR("Warning: Failed to clean up temporary directory: %s", e.what());
            }
        }

        // Create runtime and cache it
        auto runtime = std::make_unique<Runtime>(useDiskCache ? path.string() : name, cubin, gemm_type);
        Runtime* result = runtime.get();
        runtimeCache.set(name, std::move(runtime));
        if (kJitDebugging)
        {
            TLLM_LOG_INFO("Successfully cached JIT runtime %s in memory", name.c_str());
        }
        return result;
    }

private:
    std::vector<std::filesystem::path> includeDirs_;

    // Private constructor for singleton pattern. Nothing is created on disk here: the disk cache makes its
    // directories when it writes a cubin.
    Compiler()
        : includeDirs_(getJitIncludeDirs())
    {
    }

    // The JIT header directory and its digest, computed once per process on the first disk-cache lookup.
    JitHeaderDigest const& headerDigest() const
    {
        static JitHeaderDigest const digest = [this]()
        {
            JitHeaderDigest d = computeJitHeaderDigest(includeDirs_);
            if (kJitDebugging)
            {
                std::string const cacheDir = getCacheDir().string();
                TLLM_LOG_INFO("JIT disk cache %s; header directory %s: %zu files, digest %016llx", cacheDir.c_str(),
                    d.dir.string().c_str(), d.numFiles, static_cast<unsigned long long>(d.hash));
            }
            return d;
        }();
        return digest;
    }

    // Content key of one build: 16 hex digits of an FNV-1a 64 over the compiler identity (NVRTC version or nvcc
    // path), the generated source, the complete flag list (FSO_JIT_EXTRA_FLAGS and the -I include directories among
    // them) and the JIT header digest. Any header edit, flag change or compiler change yields another key, and so
    // another directory.
    std::string contentKey(
        std::string const& code, std::vector<std::string> const& flags, JitHeaderDigest const& headers) const
    {
        static std::string const compilerIdentity = jitCompilerIdentity();
        Fnv1a64 h;
        h.field("fish_scales_ops deep_gemm JIT cache v1");
        h.field(compilerIdentity);
        h.field(code);
        h.u64(flags.size());
        for (auto const& flag : flags)
        {
            h.field(flag);
        }
        h.u64(headers.hash);
        return h.hex();
    }

    // Delete copy constructor and assignment operator
    Compiler(Compiler const&) = delete;
    Compiler& operator=(Compiler const&) = delete;
};

// Global function to access the Compiler singleton
inline Compiler& getGlobalCompiler()
{
    return Compiler::getInstance();
}

} // namespace deep_gemm::jit
