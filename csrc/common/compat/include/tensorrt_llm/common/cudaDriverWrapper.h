/*
 * Standalone shim for tensorrt_llm/common/cudaDriverWrapper.h
 *
 * Only provides direct CUDA driver headers; no CUDA driver dlsym wrapper.
 */
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>
