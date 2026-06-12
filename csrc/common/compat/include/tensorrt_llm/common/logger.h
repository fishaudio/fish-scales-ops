/*
 * Standalone shim for tensorrt_llm/common/logger.h
 *
 * Replaces the full TRT-LLM logger with thin printf-based macros. Sufficient
 * for the JIT-driven deep_gemm path used by blockscale_gemm.
 */
#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/assert.h"

#include <cstdio>

#define TLLM_LOG(level, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        std::fprintf(stderr, "[blockscale_gemm][" level "] ");                                                             \
        std::fprintf(stderr, __VA_ARGS__);                                                                             \
        std::fprintf(stderr, "\n");                                                                                    \
    } while (0)

#define TLLM_LOG_TRACE(...)   ((void) 0)
#define TLLM_LOG_DEBUG(...)   ((void) 0)
#define TLLM_LOG_INFO(...)    TLLM_LOG("INFO", __VA_ARGS__)
#define TLLM_LOG_WARNING(...) TLLM_LOG("WARN", __VA_ARGS__)
#define TLLM_LOG_ERROR(...)   TLLM_LOG("ERR ", __VA_ARGS__)
#define TLLM_LOG_EXCEPTION(ex) TLLM_LOG_ERROR("%s", (ex).what())
