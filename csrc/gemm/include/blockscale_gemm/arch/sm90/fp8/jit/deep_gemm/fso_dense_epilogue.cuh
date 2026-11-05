// JIT-only header (fish_scales_ops): the dense epilogue of interface v1 (issue #12) on the sm_90 dense GEMM kernels,
// fp8_gemm_kernel and fp8_gemm_kernel_swapAB (fp8_gemm_impl.cuh) and DeepGEMM v2's sm90_fp8_gemm_1d2d_impl
// (impls/sm90_fp8_gemm_1d2d.cuh). The formula and the struct are the architecture-independent
// blockscale_gemm/common/epilogue_device.cuh; this header maps the WGMMA accumulator fragment of one thread to the
// (row, column) of the output it holds and applies the formula there, to the fp32 promoted accumulator, before the
// kernel's one rounding to bf16.
//
// The fragment: an m64nNk32 WGMMA leaves in thread t of a warp-group the accumulators acc[4i + j], i < N / 8, of
// rows r0 = 16 * (warp in the group) + lane / 4 (j = 0, 1) and r0 + 8 (j = 2, 3), columns 8i + 2 * (lane % 4) + (j % 2).
// The kernels of this file stack their warp-groups along M, so with warp_idx the warp's index among the math warps
// r0 = 16 * warp_idx + lane / 4 is the row inside the tile (the formula the kernels use for their A scales).
//
// Variant selection: a kernel template learns the epilogue it applies from one of its type arguments through
// DenseEpilogueOf<T> (kEnabled, kBias, kAct, kResidual): the scheduler input of fp8_gemm_kernel* (fp8_gemm_impl.cuh
// specializes it for NormalSchedulerInputEpi and NormalSchedulerInputSwapABEpi) and the epilogue_op_t of
// sm90_fp8_gemm_1d2d_impl (Sm90DenseEpilogueOp below). Every other type is the identity, for which the kernels'
// epilogue blocks are discarded at instantiation, so the identity kernels are the template instantiations they were
// before this header existed.
//
// Operand staging: at the start of every tile (after get_next_block, before the main loop) the math threads either
// load their bias and residual into registers, held across the main loop (DenseOperandPolicy decides, from the
// accumulator registers the tile leaves free), or prefetch them (bias into L1, residual into L2) for the epilogue's
// loads after the main loop. Either way nothing is read on the host, nothing is allocated, and the kernel's
// launch arguments carry only the pointers.
//
// PDL: every kernel that calls these helpers executes griddepcontrol.wait in all of its threads before its role
// split (fso_pdl_wait in fp8_gemm_impl.cuh, the inline griddepcontrol.wait in sm90_fp8_gemm_1d2d_impl), and the
// helpers run in the math threads inside their tile loop, after that wait, so every bias and residual load (and
// prefetch) comes after it. The wait's "memory" clobber keeps the compiler from hoisting a load above it.
#pragma once

#include "blockscale_gemm/common/epilogue_device.cuh"

namespace deep_gemm
{

template <typename T>
struct DenseEpilogueOf
{
    static constexpr bool kEnabled = false;
    static constexpr bool kBias = false;
    static constexpr int kAct = fso::kDenseActNone;
    static constexpr bool kResidual = false;
};

// The epilogue_op_t of sm90_fp8_gemm_1d2d_impl for a non-identity dense epilogue: the identity's column mapping
// (EpilogueIdentity::apply_index_n), with the fields as template arguments; the pointers come as the kernel argument
// of sm90_fp8_gemm_1d2d_impl_fso_epilogue.
template <bool kBias_, int kAct_, bool kResidual_>
struct Sm90DenseEpilogueOp
{
    template <uint32_t STORE_BLOCK_N>
    __device__ __forceinline__ static uint32_t apply_index_n(uint32_t const& n_idx)
    {
        return n_idx;
    }
};

template <bool kBias_, int kAct_, bool kResidual_>
struct DenseEpilogueOf<Sm90DenseEpilogueOp<kBias_, kAct_, kResidual_>>
{
    static constexpr bool kEnabled = true;
    static constexpr bool kBias = kBias_;
    static constexpr int kAct = kAct_;
    static constexpr bool kResidual = kResidual_;
};

// Whether 2-element loads at even offsets of p (row stride ld elements) are 4-byte aligned. A contiguous bf16 tensor
// may start at any even byte address (a view with an odd storage offset), so the vector form is chosen at run time;
// the condition is the same for every thread of the grid.
__device__ __forceinline__ bool dense_pair_aligned(__nv_bfloat16 const* p, int64_t ld)
{
    return ((reinterpret_cast<unsigned long long>(p) & 3ull) | (static_cast<unsigned long long>(ld) & 1ull)) == 0ull;
}

__device__ __forceinline__ __nv_bfloat162 dense_load_pair(__nv_bfloat16 const* p, bool aligned)
{
    if (aligned)
        return fso::dense_load_bf16x2(p);
    __nv_bfloat162 out;
    out.x = fso::dense_load_bf16(p);
    out.y = fso::dense_load_bf16(p + 1);
    return out;
}

__device__ __forceinline__ __nv_bfloat162 dense_zero_pair()
{
    return __floats2bfloat162_rn(0.f, 0.f);
}

// ------------------------------------------------------------------------------------------------ staging policy
// Which operands a tile keeps in registers from the start of the tile to the epilogue. The budget is 160 registers
// of accumulators plus staged operands per thread (the math warp-groups run with 232 or 248): the bias first
// (kNumQuads registers), then the residual (2 * kNumQuads per wave, normal layout; 4 * kNumQuads, swap-AB). Above
// the budget the operand is prefetched instead: holding the bias of the 128 x 192 tile (96 accumulators per wave, 24
// bias registers) made it spill to a 136-byte stack. kNumAccumRegs counts the fp32 accumulator registers of one
// thread (the WGMMA accumulators plus the promoted ones of every wave).
template <bool kBias, bool kResidual, uint32_t kNumAccumRegs, uint32_t kNumQuads, uint32_t kResRegsPerQuad>
struct DenseOperandPolicy
{
    static constexpr uint32_t kBudget = 160u;
    static constexpr bool kBiasInRegs = kBias && kNumAccumRegs + kNumQuads <= kBudget;
    static constexpr bool kResInRegs = kResidual
        && kNumAccumRegs + (kBiasInRegs ? kNumQuads : 0u) + kResRegsPerQuad * kNumQuads <= kBudget;
};

// A register array that is an empty struct when the operand is not held in registers (an identity kernel declares
// only empty ones, which generate nothing).
template <bool kOn, typename T, uint32_t kN>
struct DenseRegs
{
    T v[kN];

    __device__ __forceinline__ T const* ptr() const
    {
        return v;
    }
};

template <typename T, uint32_t kN>
struct DenseRegs<false, T, kN>
{
    __device__ __forceinline__ T const* ptr() const
    {
        return nullptr;
    }
};

// The staged operands of a normal tile: bias.v[i] = the pair at col0 + 8i; res.v[(2w + h) * kNumQuads + i] = the
// pair of row row0 + w * kWaveRows + 8h at col0 + 8i (wave w of kWaves).
template <bool kBiasInRegs, bool kResInRegs, uint32_t kNumQuads, uint32_t kWaves>
struct DenseOperandRegs
{
    DenseRegs<kBiasInRegs, __nv_bfloat162, kNumQuads> bias;
    DenseRegs<kResInRegs, __nv_bfloat162, 2 * kWaves * kNumQuads> res;
};

// The staged operands of a swap-AB tile: bias.v[h] = bias[wrow0 + 8h]; res.v[4i + 2h + t] = residual[tok0 + 8i + t,
// wrow0 + 8h].
template <bool kBias, bool kResInRegs, uint32_t kNumQuads>
struct DenseOperandRegsSwapAB
{
    DenseRegs<kBias, __nv_bfloat16, 2> bias;
    DenseRegs<kResInRegs, __nv_bfloat16, 4 * kNumQuads> res;
};

// ------------------------------------------------------------------------------------------------ prefetch
// Prefetch (prefetch.L1 with kL1, else prefetch.L2) the 128-byte lines of rows [row_base, row_base + kRows) and
// columns [col_base, col_base + kCols) of a bf16 matrix with row stride ld, clipped to num_rows x num_cols, by the
// threads tid < kThreads (strided). Every address prefetched lies inside the clipped segment of its row.
template <uint32_t kRows, uint32_t kCols, uint32_t kThreads, bool kL1>
__device__ __forceinline__ void dense_prefetch_tile(__nv_bfloat16 const* base, int64_t ld, uint32_t row_base,
    uint32_t num_rows, uint32_t col_base, uint32_t num_cols, uint32_t tid)
{
    // A row segment of kCols bf16 starting at any even byte address touches at most this many lines.
    constexpr uint32_t kLinesPerRow = (kCols * 2u + 125u) / 128u + 1u;
    if (col_base >= num_cols)
        return;
    uint32_t const cols = min(kCols, num_cols - col_base);
    for (uint32_t idx = tid; idx < kRows * kLinesPerRow; idx += kThreads)
    {
        uint32_t const row = idx / kLinesPerRow, line = idx % kLinesPerRow;
        if (row_base + row >= num_rows)
            continue;
        __nv_bfloat16 const* const seg = base + static_cast<int64_t>(row_base + row) * ld + col_base;
        unsigned long long const first = reinterpret_cast<unsigned long long>(seg);
        unsigned long long const last = reinterpret_cast<unsigned long long>(seg + cols - 1);
        unsigned long long const a = line == 0 ? first : (first & ~127ull) + 128ull * line;
        if (a > last)
            continue;
        if constexpr (kL1)
            asm volatile("prefetch.L1 [%0];" ::"l"(a));
        else
            asm volatile("prefetch.L2 [%0];" ::"l"(a));
    }
}

// ------------------------------------------------------------------------------------------------ staging
// The start of a normal tile: rows [m0, m0 + kBlockM), columns [n0, n0 + kBlockN); this thread's rows start at
// row0 = m0 + r0 (wave w adds w * kWaveRows) and its columns at col0 = n0 + 2 * (lane % 4). Operands that the policy
// keeps in registers are loaded into `regs`, the others prefetched by the kThreads math threads (tid).
template <typename Epi, typename Policy, uint32_t kBlockM, uint32_t kBlockN, uint32_t kThreads, uint32_t kNumQuads,
    uint32_t kWaves, uint32_t kWaveRows>
__device__ __forceinline__ void dense_stage_tile(
    DenseOperandRegs<Policy::kBiasInRegs, Policy::kResInRegs, kNumQuads, kWaves>& regs, fso::DenseEpilogue const& e,
    uint32_t m0, uint32_t row0, uint32_t num_rows, uint32_t n0, uint32_t col0, uint32_t num_cols, uint32_t tid)
{
    if constexpr (Policy::kBiasInRegs)
    {
        bool const aligned = dense_pair_aligned(e.bias, 0);
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
            uint32_t const col = col0 + 8u * i;
            regs.bias.v[i] = col < num_cols ? dense_load_pair(e.bias + col, aligned) : dense_zero_pair();
        }
    }
    else if constexpr (Epi::kBias)
    {
        dense_prefetch_tile<1, kBlockN, kThreads, true>(e.bias, 0, 0, 1, n0, num_cols, tid);
    }
    if constexpr (Policy::kResInRegs)
    {
        bool const aligned = dense_pair_aligned(e.residual, e.ld_residual);
#pragma unroll
        for (uint32_t w = 0; w < kWaves; ++w)
        {
#pragma unroll
            for (uint32_t h = 0; h < 2; ++h)
            {
                uint32_t const row = row0 + w * kWaveRows + 8u * h;
                __nv_bfloat16 const* const p = e.residual + static_cast<int64_t>(row) * e.ld_residual;
#pragma unroll
                for (uint32_t i = 0; i < kNumQuads; ++i)
                {
                    uint32_t const col = col0 + 8u * i;
                    regs.res.v[(2 * w + h) * kNumQuads + i]
                        = (col < num_cols && row < num_rows) ? dense_load_pair(p + col, aligned) : dense_zero_pair();
                }
            }
        }
    }
    else if constexpr (Epi::kResidual)
    {
        dense_prefetch_tile<kBlockM, kBlockN, kThreads, false>(
            e.residual, e.ld_residual, m0, num_rows, n0, num_cols, tid);
    }
}

// The start of a swap-AB tile: weight rows [w0, w0 + kBlockM) are the output columns, tokens [t0, t0 + kBlockN) the
// output rows; this thread's weight rows are wrow0, wrow0 + 8 and its tokens tok0 + 8i + {0, 1}.
template <typename Epi, typename Policy, uint32_t kBlockM, uint32_t kBlockN, uint32_t kThreads, uint32_t kNumQuads>
__device__ __forceinline__ void dense_stage_tile_swapab(
    DenseOperandRegsSwapAB<Epi::kBias, Policy::kResInRegs, kNumQuads>& regs, fso::DenseEpilogue const& e, uint32_t w0,
    uint32_t wrow0, uint32_t num_wrows, uint32_t t0, uint32_t tok0, uint32_t num_tokens, uint32_t tid)
{
    if constexpr (Epi::kBias)
    {
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h)
            regs.bias.v[h]
                = wrow0 + 8u * h < num_wrows ? fso::dense_load_bf16(e.bias + wrow0 + 8u * h) : __ushort_as_bfloat16(0);
    }
    if constexpr (Policy::kResInRegs)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
#pragma unroll
            for (uint32_t h = 0; h < 2; ++h)
            {
#pragma unroll
                for (uint32_t t = 0; t < 2; ++t)
                {
                    uint32_t const wrow = wrow0 + 8u * h, tok = tok0 + 8u * i + t;
                    regs.res.v[4 * i + 2 * h + t] = (wrow < num_wrows && tok < num_tokens)
                        ? fso::dense_load_bf16(e.residual + static_cast<int64_t>(tok) * e.ld_residual + wrow)
                        : __ushort_as_bfloat16(0);
                }
            }
        }
    }
    else if constexpr (Epi::kResidual)
    {
        dense_prefetch_tile<kBlockN, kBlockM, kThreads, false>(
            e.residual, e.ld_residual, t0, num_tokens, w0, num_wrows, tid);
    }
}

// ------------------------------------------------------------------------------------------------ epilogue
// The epilogue on the fragment of a normal (not swapped) tile, one wave: acc[4i + j] is output row row0 (j < 2) or
// row0 + 8, column col0 + 8i + (j % 2). num_rows / num_cols are M / N: a row or column outside them is not loaded (its
// accumulators are clipped by the kernel's TMA store). bias_regs (kNumQuads pairs) and res_regs (2 * kNumQuads
// pairs, [h][i]) are the staged operands when the policy holds them in registers; otherwise they are loaded here,
// every load of the fragment issued before the first one is used (the loops are unrolled, nothing orders them).
template <bool kBias, int kAct, bool kResidual, uint32_t kNumQuads, bool kBiasInRegs, bool kResInRegs>
__device__ __forceinline__ void dense_epilogue_fragment(float* acc, fso::DenseEpilogue const& e,
    __nv_bfloat162 const* bias_regs, __nv_bfloat162 const* res_regs, uint32_t row0, uint32_t num_rows, uint32_t col0,
    uint32_t num_cols)
{
    __nv_bfloat162 bias[kNumQuads];
    if constexpr (kBias && kBiasInRegs)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
            bias[i] = bias_regs[i];
    }
    else if constexpr (kBias)
    {
        bool const aligned = dense_pair_aligned(e.bias, 0);
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
            uint32_t const col = col0 + 8u * i;
            bias[i] = col < num_cols ? dense_load_pair(e.bias + col, aligned) : dense_zero_pair();
        }
    }
    __nv_bfloat162 r[2][kNumQuads];
    if constexpr (kResidual && kResInRegs)
    {
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h)
#pragma unroll
            for (uint32_t i = 0; i < kNumQuads; ++i)
                r[h][i] = res_regs[h * kNumQuads + i];
    }
    else if constexpr (kResidual)
    {
        bool const aligned = dense_pair_aligned(e.residual, e.ld_residual);
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h)
        {
            uint32_t const row = row0 + 8u * h;
            __nv_bfloat16 const* const p = e.residual + static_cast<int64_t>(row) * e.ld_residual;
#pragma unroll
            for (uint32_t i = 0; i < kNumQuads; ++i)
            {
                uint32_t const col = col0 + 8u * i;
                r[h][i] = (col < num_cols && row < num_rows) ? dense_load_pair(p + col, aligned) : dense_zero_pair();
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumQuads; ++i)
    {
        __nv_bfloat16 b0 = __ushort_as_bfloat16(0), b1 = __ushort_as_bfloat16(0);
        if constexpr (kBias)
        {
            b0 = bias[i].x;
            b1 = bias[i].y;
        }
        __nv_bfloat16 r00 = __ushort_as_bfloat16(0), r01 = r00, r10 = r00, r11 = r00;
        if constexpr (kResidual)
        {
            r00 = r[0][i].x;
            r01 = r[0][i].y;
            r10 = r[1][i].x;
            r11 = r[1][i].y;
        }
        acc[4 * i + 0] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 0], b0, r00);
        acc[4 * i + 1] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 1], b1, r01);
        acc[4 * i + 2] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 2], b0, r10);
        acc[4 * i + 3] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 3], b1, r11);
    }
}

// The epilogue on the fragment of a swap-AB tile, whose kernel M is the weight's N and whose kernel N is the tokens
// (the output is stored transposed, out[token][weight row]): acc[4i + j] is weight row wrow0 (j < 2) or wrow0 + 8,
// token tok0 + 8i + (j % 2), with tok0 = t0 + 2 * (lane % 4). The bias is indexed by the weight row (the output
// column) and the residual by (token, weight row). num_wrows / num_tokens are N / M. The two tokens of a pair are a
// row stride apart in the residual, so its loads are 2-byte ones (8 consecutive weight rows per 4 lanes).
template <bool kBias, int kAct, bool kResidual, uint32_t kNumQuads, bool kResInRegs>
__device__ __forceinline__ void dense_epilogue_fragment_swapab(float* acc, fso::DenseEpilogue const& e,
    __nv_bfloat16 const* bias_regs, __nv_bfloat16 const* res_regs, uint32_t wrow0, uint32_t num_wrows, uint32_t tok0,
    uint32_t num_tokens)
{
    __nv_bfloat16 const z = __ushort_as_bfloat16(0);
    __nv_bfloat16 r[kNumQuads][4];
    if constexpr (kResidual && kResInRegs)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
                r[i][j] = res_regs[4 * i + j];
    }
    else if constexpr (kResidual)
    {
#pragma unroll
        for (uint32_t i = 0; i < kNumQuads; ++i)
        {
#pragma unroll
            for (uint32_t h = 0; h < 2; ++h)
            {
#pragma unroll
                for (uint32_t t = 0; t < 2; ++t)
                {
                    uint32_t const wrow = wrow0 + 8u * h, tok = tok0 + 8u * i + t;
                    r[i][2 * h + t] = (wrow < num_wrows && tok < num_tokens)
                        ? fso::dense_load_bf16(e.residual + static_cast<int64_t>(tok) * e.ld_residual + wrow)
                        : z;
                }
            }
        }
    }
    __nv_bfloat16 b0 = z, b1 = z;
    if constexpr (kBias)
    {
        b0 = bias_regs[0];
        b1 = bias_regs[1];
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumQuads; ++i)
    {
        acc[4 * i + 0] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 0], b0, kResidual ? r[i][0] : z);
        acc[4 * i + 1] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 1], b0, kResidual ? r[i][1] : z);
        acc[4 * i + 2] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 2], b1, kResidual ? r[i][2] : z);
        acc[4 * i + 3] = fso::dense_epilogue_value<kBias, kAct, kResidual>(acc[4 * i + 3], b1, kResidual ? r[i][3] : z);
    }
}

} // namespace deep_gemm
