/*
 * The alignment checks of the operands the kernels read and of the caller-owned buffers they write (issue #12, subtasks
 * AL, CL1 and CL2).
 *
 * The bf16 inputs of the activation quantizers (the dense ones and the MoE gather / SwiGLU quantizers) and of the MoE
 * combines: each of these kernels reads its bf16 input with vector loads of a fixed width (16, 8 or 4 bytes) straight
 * from the input's data pointer, row by row. A load from an address that is not a multiple of its width is a device
 * fault ("misaligned address"), and that fault is sticky: it poisons the process's CUDA context, so every later CUDA
 * call of the process fails too. A contiguous view that starts at an element offset which is not a multiple of the
 * vector's element count (x = buf[3:].view(M, K)) is such an input. The dense quantizers select their 16-byte kernels
 * only for 16-byte aligned pointers and fall back to narrower ones; the MoE quantizers and the combines read one
 * width whatever the pointer (8 bytes; 16 for the pairwise SwiGLU form and the combines). The launchers in
 * quant_kernels.cu say which width the kernel they select reads (the *_x_vector_bytes functions; the combines'
 * width is their kernel's vector in moe_glue.cu), and the op bodies refuse, with a ValueError and before anything is
 * allocated or launched, an input whose rows that kernel cannot read (require_bf16_vector_aligned).
 *
 * The pre-quantized operands of the dense GEMMs (the fp8 activation and its scale): TMA reads them, and a TMA
 * descriptor needs a 16-byte aligned global address. For a misaligned one, CuTe's host assertion fails while the
 * descriptor is built and the process aborts (sm_100 / sm_120 CUTLASS), or the descriptor encode fails;
 * require_tma_aligned refuses it first, with ValueError.
 *
 * The weights and the int32 weight scales of the dense and grouped GEMMs, and every operand of the grouped GEMMs: TMA
 * reads them as well, and require_tma_aligned refuses a misaligned one the same way. Without the check such an operand
 * aborts the process at the host assertion (sm_100 / sm_120 CUTLASS), faults on the device where the descriptors are
 * built on the device (the sm_100 pointer-array grouped route), or fails the descriptor encode (sm_90). The fp32 weight
 * scales of the sm_90 GEMMs are read with plain loads (and repacked into a tensor of their own on sm_100 / sm_120), so
 * they are not checked.
 *
 * The caller-owned outputs of the out-style ops (the fp8 codes of the MoE quantizers, moe_combine's out, the outputs of
 * the grouped GEMMs that take one): their kernels store vectors of a fixed width at the buffer's data pointer, row by
 * row, and a store to an address that is not a multiple of its width is a device fault. require_out_aligned refuses
 * such a buffer with ValueError, by the width of the kernel the call runs (data pointer and row pitch), and
 * require_tma_store_aligned a buffer on which a GEMM builds a TMA store descriptor (16 bytes). Their int32 / fp32
 * scale outputs are written one element at a time, which a tensor of that dtype always allows.
 *
 * Host code only, included by the ATen op translation units (ops/fp8.cu, ops/mxfp8.cu, ops/mxfp8_compact.cu,
 * ops/moe_glue.cu).
 */
#pragma once

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace blockscale_gemm::detail
{

// `t`: the bf16 tensor the kernel reads (after any .contiguous() copy the op makes), `rows` rows of `row_elems`
// elements each, row r starting at element r * row_elems. `vector_bytes`: the width of the kernel's loads (from the
// launcher's *_x_vector_bytes function). Every row start must be a multiple of it: the data pointer, and, when there
// is more than one row, the row pitch. Raises ValueError naming the op, the input, the offset found and the
// requirement. An empty tensor reads nothing and passes.
inline void require_bf16_vector_aligned(at::Tensor const& t, int64_t rows, int64_t row_elems, int vector_bytes,
    char const* op, char const* name)
{
    if (t.numel() == 0 || vector_bytes <= 2)
        return;
    auto const addr = reinterpret_cast<uintptr_t>(t.data_ptr());
    auto const pitch = static_cast<uintptr_t>(row_elems) * 2u;
    auto const vb = static_cast<uintptr_t>(vector_bytes);
    TORCH_CHECK_VALUE(addr % vb == 0, op, ": the bf16 ", name, "'s data pointer is ", addr % 16,
        " bytes past a 16-byte boundary; the kernel this call runs reads it in ", vector_bytes,
        "-byte vectors, so it must be ", vector_bytes, "-byte aligned (a 16-byte aligned ", name,
        " is read on every path). Pass a tensor of its own (e.g. .clone()) or a view that starts at a multiple of 8 "
        "elements");
    TORCH_CHECK_VALUE(rows <= 1 || pitch % vb == 0, op, ": the bf16 ", name, "'s rows are ", row_elems,
        " elements (", pitch, " bytes) apart; the kernel this call runs reads each row in ", vector_bytes,
        "-byte vectors, so the row pitch must be a multiple of ", vector_bytes, " bytes");
}

// `t`: an operand a TMA descriptor of the GEMM reads from its data pointer (the fp8 activation, its packed scale).
// Its address must be a multiple of 16 bytes. Raises ValueError naming the op, the operand and the offset found. An
// empty tensor passes.
inline void require_tma_aligned(at::Tensor const& t, char const* op, char const* name)
{
    if (t.numel() == 0)
        return;
    auto const addr = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK_VALUE(addr % 16 == 0, op, ": ", name, "'s data pointer is ", addr % 16,
        " bytes past a 16-byte boundary; the GEMM reads it with TMA, which needs a 16-byte aligned address. Pass a "
        "tensor of its own (e.g. .clone()) or a view that starts at a multiple of 16 bytes");
}

// `t`: a caller-owned output buffer on which the GEMM builds a TMA store descriptor (or that it writes in 16-byte
// vectors). Its address must be a multiple of 16 bytes. Raises ValueError naming the op, the buffer and the offset
// found. An empty tensor passes.
inline void require_tma_store_aligned(at::Tensor const& t, char const* op, char const* name)
{
    if (t.numel() == 0)
        return;
    auto const addr = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK_VALUE(addr % 16 == 0, op, ": the output ", name, "'s data pointer is ", addr % 16,
        " bytes past a 16-byte boundary; the GEMM writes it through a TMA descriptor or in 16-byte vectors, which "
        "need a 16-byte aligned address. Pass a tensor of its own or a view that starts at a multiple of 16 bytes");
}

// `t`: a caller-owned output buffer that the kernel this call runs writes with stores of `vector_bytes` bytes from its
// data pointer, `rows` rows of `row_bytes` bytes each, row r starting at byte r * row_bytes. Every row start must be a
// multiple of the store width: the data pointer, and, when there is more than one row, the row pitch. Raises
// ValueError naming the op, the buffer, the offset found and the requirement. An empty tensor is written nothing and
// passes.
inline void require_out_aligned(at::Tensor const& t, int64_t rows, int64_t row_bytes, int vector_bytes, char const* op,
    char const* name)
{
    if (t.numel() == 0 || vector_bytes <= 1)
        return;
    auto const addr = reinterpret_cast<uintptr_t>(t.data_ptr());
    auto const vb = static_cast<uintptr_t>(vector_bytes);
    TORCH_CHECK_VALUE(addr % vb == 0, op, ": the output ", name, "'s data pointer is ", addr % 16,
        " bytes past a 16-byte boundary; the kernel this call runs writes it in ", vector_bytes,
        "-byte stores, so it must be ", vector_bytes, "-byte aligned (a 16-byte aligned ", name,
        " is written on every path). Pass a tensor of its own or a view that starts at a multiple of 16 bytes");
    TORCH_CHECK_VALUE(rows <= 1 || static_cast<uintptr_t>(row_bytes) % vb == 0, op, ": the output ", name,
        "'s rows are ", row_bytes, " bytes apart; the kernel this call runs writes each row in ", vector_bytes,
        "-byte stores, so the row pitch must be a multiple of ", vector_bytes, " bytes");
}

} // namespace blockscale_gemm::detail
