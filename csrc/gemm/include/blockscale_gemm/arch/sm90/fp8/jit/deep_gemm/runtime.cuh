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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "jit_utils.cuh"
#include "scheduler.cuh"

namespace deep_gemm::jit
{

// The JIT's environment knobs carry fish_scales_ops names (FSO_JIT_*). The TensorRT-LLM names this subtree was
// vendored with (TRTLLM_DG_*) stay honoured as deprecated aliases until fish-scales-ops 0.3.0: an alias is read only when its
// FSO_JIT_* name is unset, and when it is what supplied the value a notice names it on stderr. Every knob is read
// once per process (the flags below at load time, the cache directory and the nvcc path on first use), so each
// notice prints once.
inline char const* getJitEnv(char const* name, char const* legacyName)
{
    if (char const* value = std::getenv(name))
        return value;
    char const* value = std::getenv(legacyName);
    if (value != nullptr)
        std::fprintf(stderr,
            "[fish_scales_ops] %s is deprecated and will be removed in fish-scales-ops 0.3.0; set %s instead\n",
            legacyName, name);
    return value;
}

inline bool getJitEnvFlag(char const* name, char const* legacyName)
{
    char const* value = getJitEnv(name, legacyName);
    return value && (std::string(value) == "1" || std::string(value) == "true");
}

// FSO_JIT_DEBUG: verbose compile / cache log on stderr.
inline bool const kJitDebugging = getJitEnvFlag("FSO_JIT_DEBUG", "TRTLLM_DG_JIT_DEBUG");

// FSO_JIT_USE_NVCC: compile with nvcc instead of NVRTC; nvcc's output always goes through the disk cache.
inline bool const kJitUseNvcc = getJitEnvFlag("FSO_JIT_USE_NVCC", "TRTLLM_DG_JIT_USE_NVCC");

// FSO_JIT_DUMP_CUBIN: also write every NVRTC-compiled cubin to the disk cache (and load from it).
inline bool const kJitDumpCubin = getJitEnvFlag("FSO_JIT_DUMP_CUBIN", "TRTLLM_DG_JIT_DUMP_CUBIN");

inline std::string const kKernelName = kJitUseNvcc ? "nvcc_kernel.cubin" : "nvrtc_kernel.cubin";

/**
 * C++ implementation of the Runtime class from runtime.py
 * Loads and executes JIT-compiled kernels. With a non-empty cubin the kernel comes from memory and `path` is only a
 * label; with an empty one it is read from `path`/kKernelName (a disk-cache directory) on the first getKernel().
 */
class Runtime
{
public:
    Runtime(std::string const& path, std::vector<char> const& cubin, deep_gemm::GemmType gemm_type)
        : path_(path)
        , cubin_(cubin)
        , gemm_type_(gemm_type)
        , lib_(nullptr)
        , kernel_(nullptr)
    {
        DG_HOST_ASSERT(!cubin.empty() || isPathValid(path_));
    }

    ~Runtime()
    {
        if (lib_ != nullptr)
        {
            CHECK_CUDA(cuLibraryUnload(lib_));
        }
    }

    static bool isPathValid(std::string const& path)
    {
        // Check if path exists and is a directory
        if (!std::filesystem::exists(path) || !std::filesystem::is_directory(path))
        {
            return false;
        }

        // Check if all necessary files exist
        return std::filesystem::exists(std::filesystem::path(path) / kKernelName);
    }

    CUkernel getKernel()
    {
        // Load shared object if not already loaded
        if (kernel_ == nullptr)
        {
            if (cubin_.empty())
            {
                std::filesystem::path cubinPath = std::filesystem::path(path_);
                cubinPath /= kKernelName;
                std::ifstream cubinFile(cubinPath.string(), std::ios::binary);
                cubin_ = std::vector<char>(std::istreambuf_iterator<char>(cubinFile), {});
            }

            CHECK_CUDA(cuLibraryLoadData(&lib_, cubin_.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

            unsigned int numKernels = 0;
            CHECK_CUDA(cuLibraryGetKernelCount(&numKernels, lib_));

            std::vector<CUkernel> kernels(numKernels);
            CHECK_CUDA(cuLibraryEnumerateKernels(kernels.data(), numKernels, lib_));

            for (auto kernel : kernels)
            {
                char const* kernelName;
                CHECK_CUDA(cuKernelGetName(&kernelName, kernel));
                std::string kernelNameStr(kernelName);
                if (kernelNameStr.find("fp8_gemm_kernel") != std::string::npos)
                {
                    kernel_ = kernel;
                    break;
                }
            }

            if (!kernel_)
            {
                throw std::runtime_error("Failed to find fp8_gemm_kernel");
            }
        }

        return kernel_;
    }

private:
    std::string path_;
    std::vector<char> cubin_;
    CUlibrary lib_;
    CUkernel kernel_;
    deep_gemm::GemmType gemm_type_;
};

/**
 * In-memory cache of the process's Runtime instances, keyed by the kernel name Compiler::build derives from its
 * template arguments. Lookups never touch the file system: the content-keyed disk cache (FSO_JIT_DUMP_CUBIN /
 * FSO_JIT_USE_NVCC only) is consulted by Compiler::build on an in-memory miss, with the GEMM type taken from the
 * build call rather than parsed back out of a path.
 */
class RuntimeCache
{
public:
    static RuntimeCache& getInstance()
    {
        static RuntimeCache instance;
        return instance;
    }

    // The per-launch hit path of every sm_90 GEMM: one map lookup.
    Runtime* find(std::string const& name) const
    {
        auto it = cache_.find(name);
        return it != cache_.end() ? it->second.get() : nullptr;
    }

    void set(std::string const& name, std::unique_ptr<Runtime>&& runtime)
    {
        cache_[name] = std::move(runtime);
    }

private:
    // Private constructor for singleton pattern
    RuntimeCache() = default;

    // Delete copy constructor and assignment operator
    RuntimeCache(RuntimeCache const&) = delete;
    RuntimeCache& operator=(RuntimeCache const&) = delete;

    std::unordered_map<std::string, std::unique_ptr<Runtime>> cache_;
};

// Global function to access the singleton
RuntimeCache& getGlobalRuntimeCache()
{
    return RuntimeCache::getInstance();
}

} // namespace deep_gemm::jit
