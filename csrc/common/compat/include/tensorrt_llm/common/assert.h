/*
 * Standalone shim for tensorrt_llm/common/assert.h
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/tllmException.h"

#if defined(_WIN32)
#define TLLM_LIKELY(x) (x)
#define TLLM_UNLIKELY(x) (x)
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#define TLLM_UNLIKELY(x) __builtin_expect((x), 0)
#endif

#define TLLM_CHECK(val)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0) : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val);                              \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        TLLM_LIKELY(static_cast<bool>(val))                                                                            \
        ? ((void) 0)                                                                                                   \
        : tensorrt_llm::common::throwRuntimeError(                                                                     \
            __FILE__, __LINE__, tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__).c_str());                            \
    } while (0)

#define TLLM_CHECK_DEBUG(val) ((void) 0)
#define TLLM_CHECK_DEBUG_WITH_INFO(val, info, ...) ((void) 0)
