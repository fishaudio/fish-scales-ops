/*
 * Standalone shim for tensorrt_llm/common/config.h
 *
 * Provides only the namespace-handling macros used by blockscale_gemm and
 * deep_gemm. Mirrors the public ABI of the original config.h so vendored
 * sources compile unchanged.
 */
#pragma once
#ifndef TRTLLM_CONFIG_H
#define TRTLLM_CONFIG_H

#ifndef TRTLLM_ABI_NAMESPACE
#define TRTLLM_ABI_NAMESPACE _v1
#endif

#ifndef TRTLLM_ABI_NAMESPACE_BEGIN
#define TRTLLM_ABI_NAMESPACE_BEGIN inline namespace TRTLLM_ABI_NAMESPACE {
#endif

#ifndef TRTLLM_ABI_NAMESPACE_END
#define TRTLLM_ABI_NAMESPACE_END }
#endif

#define TRTLLM_NAMESPACE_BEGIN namespace tensorrt_llm { TRTLLM_ABI_NAMESPACE_BEGIN
#define TRTLLM_NAMESPACE_END   TRTLLM_ABI_NAMESPACE_END }

#endif // TRTLLM_CONFIG_H
