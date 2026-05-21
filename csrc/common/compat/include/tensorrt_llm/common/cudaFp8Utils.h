/*
 * Standalone shim for tensorrt_llm/common/cudaFp8Utils.h
 *
 * The original file pulls in a large amount of FP8-related utility code that
 * the kernel does not actually need. We only forward the FP8 type header.
 */
#pragma once

#include <cuda_fp8.h>
