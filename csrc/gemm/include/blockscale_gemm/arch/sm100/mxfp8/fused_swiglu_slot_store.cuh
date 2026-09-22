/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The dual store of the sm_100 / sm_103 fused FC1 epilogue in the SWAP
// orientation — the geometry the slot-bound decode route runs. It turns one
// epilogue subtile of the FP32 accumulator into MXFP8(silu(gate) * up): the fp8
// bytes plus the 1x32 UE8M0 scale bytes that the second grouped GEMM (FC2)
// reads as its activation operand.
//
// Why a second store at all
// -------------------------
// `fused_swiglu_store.cuh` does the same job for the POINTER-ARRAY route, where
// the routed token rows sit on the GEMM's M axis and the weight rows on N. A
// thread there owns one output row and 64 consecutive N columns, so after the
// gate/up interleave the pairing and the 32-wide amax are both register-local.
// The slot route swaps the operands: the expert weight rows go on M and the
// tokens on a 64-wide N tile. The accumulator rows are then WEIGHT rows, so
// gate_j and up_j land in two different TMEM datapaths and the 32 outputs of
// one scale block span 64 datapaths, i.e. two warps. Neither the pairing nor
// the amax is register-local any more, and the destination is transposed with
// respect to the fragment. That is a different store, not a parameterisation of
// the first one, which is why this file exists (run b300_mxfp8_20260917/M-E1
// section 1.8 recorded the swap orientation as out of scope for exactly this
// reason; run b300_mxfp8_20260917/M-A2 is the work that closed it).
//
// The fragment this is written against, and how it is obtained
// ------------------------------------------------------------
// The slot GEMM's fused variant asks the CUTLASS epilogue builder for the
// direct-store (NoSmem) EVT epilogue, which pins the epilogue tile to
// (TileM, min(64, TileN)) = (128, 64) — one subtile for the whole CTA tile —
// and it declares the epilogue's D as N-MAJOR. The D layout is what selects the
// TMEM-to-register copy: `sm100_get_tmem_load_op` returns the 16dp256b
// (stmatrix_t) atom for an M-major bf16 D, where a thread owns two rows and two
// columns and no lane neighbour holds the gate/up partner, and the 32dp32b atom
// for an N-major one. With 32dp32b and `make_tmem_copy`'s four-warp split the
// mapping is exactly
//
//     accumulator row m = cta_m * 128 + 32 * warp + lane,
//     thread (warp, lane) holding 64 CONSECUTIVE token columns of that row,
//
// i.e. LANE = ROW. Nothing is stored through D, so declaring it N-major costs
// nothing; it only steers the copy atom. `FSO_FUSED_SWIGLU_SLOT_CHECK_FRAG=1`
// turns the mapping into a device-side trap (`(m & 31) == lane` and
// `((m >> 5) & 3) == warp`) instead of an assumption.
//
// What the store then does, per chunk of token columns
// ----------------------------------------------------
//   * pairing. With the gate/up interleave row 2j is gate_j and row 2j+1 is
//     up_j, so the partner of lane l is lane l^1: one `shfl_xor(v, 1)` per token
//     column gives both lanes both operands, and both compute the same
//     h = silu(gate) * up in FP32. The even (gate) lane owns output element
//     j = m/2; the odd lane's copy is deliberate — see the next point.
//   * the amax over one 1x32 output block, for ONE token. A warp holds 16
//     consecutive output elements, so a block of 32 spans warps 2b and 2b+1.
//     Inside a warp the reduction is a single `redux.sync.max.u32`: both lanes
//     of a pair computed the same h, so all 32 lanes may take part and the
//     instruction replaces a four-step shuffle butterfly. The u32 max is the
//     float max because the values are non-negative and IEEE-754 magnitudes are
//     monotonic in their bit pattern. Across the warp pair it is one small
//     shared-memory exchange behind the epilogue's own named barrier.
//   * ONE LANE PER TOKEN derives that token's scale. This is the step the swap
//     orientation makes or breaks, and getting it wrong is what a first version
//     of this file did: the UE8M0 derivation is three float divisions and about
//     thirty-five instructions, and in this orientation a lane owns ONE output
//     element of a token, not thirty-two, so deriving the scale inside the
//     per-token loop makes every lane pay it for every token. Measured on the
//     Family B layer at M = 32 that cost 18.5 microseconds, turning the fusion
//     into a 17.8 per cent loss (run b300_mxfp8_20260917/M-A2, jsonl/v1/). The
//     per-token maxima are already in shared memory after the cross-warp step,
//     so lane l instead reads token `base + l`'s pair of maxima, derives that
//     ONE token's scale, and writes that one token's scale byte -- the whole
//     32-token chunk's scale work in one pass of warp instructions instead of
//     thirty-two. The quantise loop then gets each token's multiplier with a
//     single `shfl_sync` broadcast.
//   * the stores. The fp8 byte for (token c, output element j) is written by
//     the gate lane that owns j, so one warp instruction writes 16 CONSECUTIVE
//     bytes of one token's row and the four warps of the epilogue cover 64.
//     That is eight times more store instructions than a fully vectorised store
//     would issue for the same bytes, which is the price of the transposed
//     destination and is affordable at the decode row counts this route serves
//     (m_cap <= 64). The scale byte is written by the lane that derived it, so
//     one warp instruction covers 32 tokens of one block.
//
// Why the token tile is walked in chunks
// --------------------------------------
// A thread's accumulator run is 64 floats, which is already 64 registers of a
// budget the epilogue shares with the stock tensors. Keeping all 64 activated
// values alive across the cross-warp exchange as well would add 64 more and
// spill to local memory. The tile is therefore walked in chunks of
// `kSlotChunk` = 32 columns -- one warp width, which is what lets one lane own
// one token in the scale pass -- and each chunk publishes into its OWN slice of
// the shared buffer so that one barrier per chunk is enough: a later chunk
// never overwrites a region an earlier chunk's readers have not finished with,
// and the hazard against the PREVIOUS tile is closed by the barrier the stock
// epilogue already takes at the top of its subtile loop.
//
// Numerics are the pointer-array store's, instruction for instruction: FP32
// accumulator -> silu in FP32 -> 32-wide amax -> UE8M0 byte -> e4m3, with no
// bf16 staging anywhere. The helpers are vendored from this repository's own
// `csrc/gemm/ops/quant_kernels.cu` (`e8m0_from_amax<true>`, `silu_tanh_approx`
// and the conversion of `fp8x4_from_floats`), which cannot be included here
// because they live in that translation unit's anonymous namespace; the
// scale-factor word index is vendored from the same file's
// `sf_word_index_grouped_atom` with the token index taken from the GEMM's N
// coordinate instead of its M coordinate, which is the whole of what the swap
// changes about the destination.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cute/tensor.hpp"

namespace cutlass
{
namespace epilogue
{
namespace collective
{
namespace fso_swiglu_slot
{

// Arguments of the fused store, carried through the epilogue's Arguments and
// Params by the generated epilogue clone.
//
// The two destinations are plain base pointers plus per-group element counts
// rather than pointer arrays, because the MoE slabs are contiguous in the group
// index: group g's fp8 rows start at `ptr_h + g * h_group_elems` and its scale
// slab at `ptr_sfh + g * sf_group_words`. The group index the epilogue uses is
// the one the slot kernel already remapped from slot to expert id, so no extra
// indirection reaches this struct.
struct FusedSwiGluSlotArgs
{
    void* ptr_h = nullptr;         // __nv_fp8_e4m3 base of [G, m_cap, I]
    void* ptr_sfh = nullptr;       // int32 base of [G, pad(m_cap,128) * I/128]
    long long h_group_elems = 0;   // m_cap * I
    long long sf_group_words = 0;  // pad(m_cap,128) * (I/128)
};

// ---- vendored from csrc/gemm/ops/quant_kernels.cu -------------------------
// e8m0_from_amax<true> (UE8M0). Derives the descale via 448/amax -> 1/qs rather
// than amax * (1/448): the two differ by up to one ULP and pick different bytes
// at power-of-two boundaries, so the order matters for byte-for-byte agreement
// with the standalone kernel.
__device__ __forceinline__ void fso_slot_e8m0_from_amax(float amax, float& quant_scale_out, unsigned char& byte_out)
{
    float const quant_scale_raw = 448.f / amax;
    float const dequant_raw = 1.f / quant_scale_raw;
    unsigned int const bits = __float_as_uint(dequant_raw);
    unsigned int const exp_field = (bits >> 23) & 0xFFu;
    unsigned int const has_mantissa = (bits & 0x7FFFFFu) != 0u ? 1u : 0u;
    unsigned int byte = exp_field + has_mantissa;
    byte = byte > 254u ? 254u : byte;
    byte_out = static_cast<unsigned char>(byte);
    unsigned int const ds_bits = byte << 23;
    float const dequant_scale = __uint_as_float(ds_bits);
    quant_scale_out = byte != 0u ? (1.f / dequant_scale) : 1.f;
}

// silu_tanh_approx: silu(x) = 0.5*x*(1 + tanh(x/2)), one MUFU instead of the
// two __expf costs.
__device__ __forceinline__ float fso_slot_silu(float x)
{
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * x));
    return 0.5f * x * (1.f + t);
}

// One e4m3 byte, the same cvt.rn.satfinite the paired conversion of
// `fp8x4_from_floats` issues. The swap orientation cannot pair two outputs in
// one thread — the elements a lane owns are 16 apart in the output — so this is
// the scalar form.
__device__ __forceinline__ unsigned char fso_slot_fp8(float v)
{
    return static_cast<unsigned char>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// Warp-wide max of a non-negative float, as one instruction. IEEE-754
// magnitudes are monotonic in their unsigned bit pattern, so the integer max is
// the float max; a NaN input wins the max and leaves that ONE token's scale
// byte undefined, which is inside the masked contract (a token at or past
// masked_m[g] is undefined in the output) and cannot reach another token
// because the reduction is per token.
__device__ __forceinline__ float fso_slot_warp_amax(float v)
{
    unsigned int const bits = __float_as_uint(v);
    unsigned int out;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(out) : "r"(bits));
    return __uint_as_float(out);
}
// ---------------------------------------------------------------------------

#ifndef FSO_FUSED_SWIGLU_SLOT_CHECK_FRAG
#define FSO_FUSED_SWIGLU_SLOT_CHECK_FRAG 0
#endif

// The token columns one epilogue subtile of the fused slot variant covers
// (TileN 64, one subtile), the chunk the store walks them in, and the floats of
// epilogue shared storage the cross-warp amax exchange needs: one row of 64 per
// epilogue warp, sliced by chunk so one barrier per chunk suffices.
constexpr int kSlotEpiTokens = 64;
constexpr int kSlotChunk = 32;
constexpr int kSlotAmaxFloats = 4 * kSlotEpiTokens;

// The fused store, called once per epilogue subtile in place of the EVT visit
// loop and the bf16 D store.
//
// `acc_frag` is the FP32 accumulator this thread holds for ONE epilogue subtile
// and `coord_frag` the matching coordinate tensor, both exactly as the stock
// epilogue built them. (M, N) are the problem extents — M is the expert weight
// row count 2*I, N the row capacity m_cap — and `l_coord` the expert id the
// slot kernel remapped the batch coordinate to. `smem_amax` is
// `kSlotAmaxFloats` floats of the epilogue's own shared storage and `sync` the
// epilogue's named-barrier lambda.
template <class AccEngine, class AccLayout, class CoordEngine, class CoordLayout, class SyncFn>
CUTLASS_DEVICE void fused_swiglu_slot_mxfp8_store(cute::Tensor<AccEngine, AccLayout> const& acc_frag,
    cute::Tensor<CoordEngine, CoordLayout> const& coord_frag, FusedSwiGluSlotArgs const& f, int M, int N, int l_coord,
    float* smem_amax, int thread_idx, SyncFn sync)
{
    using namespace cute;
    auto acc = coalesce(acc_frag);
    auto crd = coalesce(coord_frag);
    constexpr int kRun = CUTE_STATIC_V(size(acc));
    static_assert(kRun == kSlotEpiTokens,
        "fused SwiGLU slot epilogue: a thread's accumulator run must be the whole 64-column token tile (TileN 64, "
        "one epilogue subtile); a different epilogue tile would break the lane = row mapping this store rests on");
    static_assert(kRun == CUTE_STATIC_V(size(crd)), "fused SwiGLU slot epilogue: coordinate/accumulator size mismatch");
    static_assert(kRun % kSlotChunk == 0, "fused SwiGLU slot epilogue: the token run must be a whole number of chunks");

    auto c0 = crd(_0{});
    int const m = get<0>(c0);   // accumulator row = expert WEIGHT row
    int const n0 = get<1>(c0);  // first token column of this thread's run
    int const lane = thread_idx & 31;
    int const warp = (thread_idx >> 5) & 3;

#if FSO_FUSED_SWIGLU_SLOT_CHECK_FRAG
    // The CUTLASS-internal partitioning the whole store rests on: one row per
    // thread, contiguous in n from the tile's first column, with the row
    // determined by the lane and the warp. A debug build traps here instead of
    // letting a differently-partitioned epilogue produce a silently wrong
    // answer.
    if ((m & 31) != lane || ((m >> 5) & 3) != warp || (n0 & 63) != 0)
        __trap();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kRun; ++i)
    {
        auto ci = crd(i);
        if (get<0>(ci) != m || get<1>(ci) != n0 + i)
            __trap();
    }
#endif

    // Everything below is warp-uniform except `m`, `lane` and the accumulator
    // itself, so the warp-collective steps (the pairing shuffle, the redux, the
    // barrier) are reached by every lane of every epilogue warp under the same
    // conditions. That is why nothing returns early and why the row and column
    // predicates are applied at the stores instead of at the top.
    int const I = M >> 1;       // output width: the GEMM's M is 2*I
    int const num_kp = I >> 7;  // scale-slab blocks of 128 output columns
    bool const is_gate = (m & 1) == 0;
    int const j = m >> 1; // this lane's output element
    // How many of this thread's 64 token columns are inside the problem. The
    // chunk loop is uniform across the whole epilogue warpgroup because N and
    // n0 are, which is what keeps the barrier, the redux and the shuffles
    // converged; only the stores are predicated per column.
    int const kLive = (N - n0) < kRun ? (N - n0) : kRun;

    unsigned char* const hbase = reinterpret_cast<unsigned char*>(f.ptr_h)
        + static_cast<long long>(l_coord) * f.h_group_elems + static_cast<long long>(j);
    unsigned char* const sfbase
        = reinterpret_cast<unsigned char*>(f.ptr_sfh) + 4ll * (static_cast<long long>(l_coord) * f.sf_group_words);
    // The 1x32 output block this warp pair covers. The CTA tile starts on a
    // multiple of 128 weight rows, so its output elements start on a multiple of
    // 64 and the block base is `j` rounded down to 32; warps 2b and 2b+1
    // therefore agree on it, which is what lets one of them write the byte for
    // both.
    int const jb = j & ~31;
    int const kp = jb >> 7;
    int const sf_byte = (jb >> 5) & 3;
    // Who writes what. The fp8 byte of (token c, element j) belongs to the gate
    // lane that owns j. The scale byte of (token c, block) is derived and
    // written by the lane that owns token c in the scale pass, and the two
    // warps of a pair would derive the same byte, so only the even warp of each
    // pair writes it.
    bool const stores_h = is_gate && m < M;
    bool const stores_sf = ((warp & 1) == 0) && m < M;
    int const partner = warp ^ 1;

    CUTLASS_PRAGMA_UNROLL
    for (int base = 0; base < kRun; base += kSlotChunk)
    {
        if (base >= kLive)
            break;

        // Pass 1: the activation, and this warp's half of each token's amax.
        // The maximum goes straight to shared memory rather than into a second
        // register array, four tokens at a time from one lane.
        float h[kSlotChunk];
        float pub0 = 0.f, pub1 = 0.f, pub2 = 0.f, pub3 = 0.f;
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kSlotChunk; ++t)
        {
            // The decode band runs this kernel at m_cap as small as 4 against a
            // 32-column chunk, so stopping at the live columns is most of the
            // epilogue's work at the smallest M. The bound is warp-uniform and
            // a multiple of four (the op requires m_cap % 4 == 0), so the exit
            // lands between two publishes and leaves neither the redux nor the
            // publish half-done.
            if (base + t >= kLive)
                break;
            float const mine = acc(base + t);
            float const other = __shfl_xor_sync(0xffffffffu, mine, 1);
            float const g = is_gate ? mine : other;
            float const u = is_gate ? other : mine;
            float const v = fso_slot_silu(g) * u;
            h[t] = v;
            float const a = fso_slot_warp_amax(fabsf(v));
            if ((t & 3) == 0)
                pub0 = a;
            else if ((t & 3) == 1)
                pub1 = a;
            else if ((t & 3) == 2)
                pub2 = a;
            else
            {
                pub3 = a;
                if (lane == 0)
                    *reinterpret_cast<float4*>(smem_amax + warp * kSlotEpiTokens + base + t - 3)
                        = make_float4(pub0, pub1, pub2, pub3);
            }
        }
        sync();

        // Pass 2: one lane per token. Lane l owns token `base + l` of this
        // chunk: it combines the two warps' halves, derives that token's UE8M0
        // byte and its quantise multiplier once, and writes that token's scale
        // byte. The whole chunk's scale work is one pass of warp instructions.
        int const c_lane = n0 + base + lane;
        // A lane past the live columns has nothing to derive and its slice of
        // the buffer was not written by this tile, so it takes a harmless
        // constant instead of reading the previous tile's leftovers.
        float ax = (base + lane < kLive)
            ? fmaxf(smem_amax[warp * kSlotEpiTokens + base + lane],
                smem_amax[partner * kSlotEpiTokens + base + lane])
            : 1.f;
        ax = fmaxf(ax, 1e-10f); // the floor the standalone kernel applies
        float qs_lane;
        unsigned char byte_lane;
        fso_slot_e8m0_from_amax(ax, qs_lane, byte_lane);
        if (stores_sf && c_lane < N)
        {
            // sf_word_index_grouped_atom(c, kp, m_cap, num_kp, g), with the
            // group term already folded into sfbase:
            //     word = (c/128 * num_kp + kp) * 128 + (r%32)*4 + r/32,
            //     r = c % 128
            int const r = c_lane & 127;
            long long const word = (static_cast<long long>(c_lane >> 7) * num_kp + kp) * 128
                + static_cast<long long>((r & 31) * 4 + (r >> 5));
            sfbase[4ll * word + sf_byte] = byte_lane;
        }

        // Pass 3: quantise and store, one token at a time, taking that token's
        // multiplier from the lane that derived it.
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kSlotChunk; ++t)
        {
            float const qs = __shfl_sync(0xffffffffu, qs_lane, t);
            int const c = n0 + base + t;
            if (c >= N)
                break;
            if (stores_h)
                hbase[static_cast<long long>(c) * I] = fso_slot_fp8(h[t] * qs);
        }
    }
}

} // namespace fso_swiglu_slot
} // namespace collective
} // namespace epilogue
} // namespace cutlass
