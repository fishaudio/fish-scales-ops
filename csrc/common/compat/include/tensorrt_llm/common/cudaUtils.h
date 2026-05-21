/*
 * Standalone shim for tensorrt_llm/common/cudaUtils.h
 *
 * Provides only the symbols actually referenced by blockscale_gemm and
 * deep_gemm: getSMVersion, getMultiProcessorCount, check_cuda_error,
 * sync_check_cuda_error.
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime.h>
#include <string>

TRTLLM_NAMESPACE_BEGIN
namespace common
{

inline void check_cuda_error_impl(cudaError_t result, char const* file, int line, char const* expr)
{
    if (result != cudaSuccess)
    {
        TLLM_THROW("CUDA error %s at %s:%d: %s", cudaGetErrorString(result), file, line, expr);
    }
}

#define check_cuda_error(call) ::tensorrt_llm::common::check_cuda_error_impl((call), __FILE__, __LINE__, #call)

#define sync_check_cuda_error(stream)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        check_cuda_error(cudaStreamSynchronize(stream));                                                               \
        check_cuda_error(cudaGetLastError());                                                                          \
    } while (0)

#define TLLM_CUDA_CHECK(call) check_cuda_error(call)
#define TLLM_CU_CHECK(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult _r = (call);                                                                                          \
        if (_r != CUDA_SUCCESS)                                                                                        \
        {                                                                                                              \
            char const* err = nullptr;                                                                                 \
            cuGetErrorString(_r, &err);                                                                                \
            TLLM_THROW("CUDA driver error at %s:%d: %s", __FILE__, __LINE__, err ? err : "<unknown>");                 \
        }                                                                                                              \
    } while (0)

inline int getSMVersion(bool /*queryRealSmArch*/ = false)
{
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    int sm_major = 0;
    int sm_minor = 0;
    check_cuda_error(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
    check_cuda_error(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
    return sm_major * 10 + sm_minor;
}

inline int getMultiProcessorCount()
{
    int nSM{0};
    int deviceID{0};
    check_cuda_error(cudaGetDevice(&deviceID));
    check_cuda_error(cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID));
    return nSM;
}

inline int getMaxSharedMemoryPerBlockOptin()
{
    int v = 0;
    int deviceID = 0;
    check_cuda_error(cudaGetDevice(&deviceID));
    check_cuda_error(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID));
    return v;
}

} // namespace common
TRTLLM_NAMESPACE_END
