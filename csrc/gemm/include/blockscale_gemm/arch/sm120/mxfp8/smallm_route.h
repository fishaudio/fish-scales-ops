/*
 * Host side of the sm_120 / sm_121 small-M dense MXFP8 route ("sm120.smallm", smallm_tc.cuh): when a call takes it,
 * and with which configuration. Included by the ATen translation unit (ops/mxfp8.cu) and by the kernel translation
 * units (ops/mxfp8_smallm_*_kernel.cu); it pulls in nothing but the C library (the shared-memory layout function is
 * also compiled for the device, where the kernel reads it).
 *
 * Two rules, formulas in (M, N, K) with guards on the plan config() picks, refitted on library timings of this kernel
 * (issue #12 Q120d: RTX 5090 at a locked 2407 MHz, cold weights, CUDA-graph replay, two passes each of 168 cells of
 * twelve decode shapes, N 1024 .. 19456, K 512 .. 9728, M 1 .. 64, and of 108 held-out cells, N {1280, 2560, 4096} x
 * K {1280, 3072, 5120, 7680}). Both serve the decode band only: nothing above M = 64 changes.
 *
 *   R1, the 2b call linear(xq, w, x_scale=sx):  G1 instead of the CUTLASS cascade when
 *           pad8(M) * K * (1 + N / 16384) <= 73728, the activation stays resident in shared memory (no K chunks),
 *           nc <= 5, and the N / (16 rt) tile groups fit one round of the persistent grid on the device (at most its SM
 *           count: on the RTX 5090's 170 SMs N <= 2720 at rt = 1 and N <= 5440 at rt = 2).
 *   R2, the bf16-x call linear(x, w):           G2 (one launch) instead of quantize + the 2b route when
 *           f(M) <= 65536 and (M <= 8 or f(pad8(M)) <= 65536), f(m) = m * K * (1 + N / 24576) * (K / 2048)^(1/4),
 *           and R1.
 *
 * Why this form: the family's cost over a pure weight stream grows with the activation every CTA stages (and, in G2,
 * quantizes), M * K bytes, while the cascade amortizes its fixed cost over more rows as N grows, so the budget for
 * M * K shrinks with N; R2 also pays the quantize per CTA, which grows faster with K than the chain's separate
 * quantize does. The guards follow what the measured cells show: a plan whose tile groups need more than one round of
 * the persistent grid loses to the cascade already at M = 1 (6144 x 2560, 9216 x 2048, 12288 x 2048, 19456 x 2560,
 * 4096 x 5120 and 4096 x 7680 measured 0.5-24 % slower at M <= 16), plans that stage the activation in K chunks lose,
 * and nc >= 6 (M above 40) loses on the short-K shapes. Where R1 routes the 2b call and R2 does not take G2, the
 * bf16-x call runs quantize + G1, which was measured slower than quantize + the cascade on cells where G1 alone wins
 * (the quantize kernel overlaps the cascade better than the small-M kernel under PDL), so R1's budget is also bound by
 * that call. The kernel's configuration changes with every 8 rows (nc = ceil(M / 8)), so above M = 8 a rule may only
 * take an 8-row bucket whose top passes too (R1 is written in pad8(M) for the same reason). On the 276 measured cells
 * the rules route the 2b call on 83 and the bf16-x call to G2 on 75, and neither call is more than 1 % slower than
 * before on any cell in either pass; they leave on the cascade 39 cells where G1 was more than 3 % faster and 41
 * where G2 was (among them the multi-round shapes that still win, 4096 x 1280, 4096 x 3072 and 5120 x 2048, and
 * 2560 x 9728 at M <= 4, whose 8-row bucket R1 cannot split). R2 implies R1 on every cell; the code requires it.
 *
 * R2 implies R1. That keeps interface v1's identity linear(xq, w, x_scale=sx) == linear(x, w) byte for byte, with
 * xq, sx = quantize_activation(x): where R2 picks G2 the 2b call runs G1, and G2 equals quantize + G1 bit for bit
 * (same staged codes and scale bytes, same accumulation); where R2 does not pick G2, the bf16-x call is quantize +
 * the 2b call (G1 where R1 picks it, else the cascade).
 *
 * FSO_SMALLM_ROUTE (read once per process): unset or "rules" = the rules; "0" / "off" = never (the routes of
 * 0.2.2); "1" / "on" = every call with M <= 64 the kernel can serve (tests, A/B). A forced cascade configuration
 * (FSO_FORCE_TILE, FSO_FORCE_KSPLIT, FSO_FORCE_SMALLM, FSO_FORCE_MIN_BLOCKS, FSO_FORCE_SCHED_GROUP) switches the
 * route off, so a forced tile always runs the tile it names.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__CUDACC__)
#define FSO_SMALLM_HD __host__ __device__
#else
#define FSO_SMALLM_HD
#endif

namespace fso::sm120_smallm
{

constexpr int kMaxM = 64;                  // the decode band
constexpr int kSmemPerBlockMax = 101376;   // sm_120 opt-in dynamic shared memory per block (99 KB)
constexpr int kSmemMargin = 1024;
constexpr int kRing = 8;                   // weight-load ring depth, in steps (smallm_tc.cuh)
constexpr int kRedStride = 20;             // reduction buffer floats per (warp, tile, column): 16 rows + pad
// The rules' constants (see the comment at the top of this file).
constexpr double kBudgetR1 = 73728.0;    // R1: pad8(M) * K * (1 + N / kN0R1) <= kBudgetR1
constexpr double kN0R1 = 16384.0;
constexpr bool kR1ResidentOnly = true;   // R1 plan guard: only plans whose activation stays resident (cj == 0)
constexpr int kR1NcMax = 5;              // R1 plan guard: only plans with nc <= kR1NcMax
constexpr bool kR1OneRound = true;       // R1 plan guard: only plans whose tile groups fit one round of the persistent
                                         // grid on the device (N / (16 rt) <= its SM count)
constexpr double kBudgetR2 = 65536.0;    // R2: f(m) <= kBudgetR2, f(m) = m * K * (1 + N / kN0R2) * (K / 2048)^q
constexpr double kN0R2 = 24576.0;
constexpr int kR2KExpQuarters = 1;       // R2: q in quarters, 0 (q = 0), 1 (q = 1/4) or 2 (q = 1/2)
constexpr int kR2KMax = 1 << 30;         // R2: only for K <= kR2KMax

enum Mode : int
{
    kModeRules = 0,
    kModeOff = 1,
    kModeOn = 2,
};

inline bool env_set(char const* name)
{
    char const* e = std::getenv(name);
    return e != nullptr && e[0] != '\0';
}

// FSO_SMALLM_ROUTE, and the cascade's force knobs, read once per process.
inline int mode()
{
    static int const s_mode = []
    {
        if (env_set("FSO_FORCE_TILE") || env_set("FSO_FORCE_KSPLIT") || env_set("FSO_FORCE_SMALLM")
            || env_set("FSO_FORCE_MIN_BLOCKS") || env_set("FSO_FORCE_SCHED_GROUP"))
            return int(kModeOff);
        char const* e = std::getenv("FSO_SMALLM_ROUTE");
        if (e == nullptr || e[0] == '\0' || std::strcmp(e, "rules") == 0)
            return int(kModeRules);
        if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0)
            return int(kModeOff);
        if (std::strcmp(e, "1") == 0 || std::strcmp(e, "on") == 0)
            return int(kModeOn);
        return int(kModeRules);
    }();
    return s_mode;
}

// The shapes the kernel serves at all: M in [1, 64], K a multiple of 128 (the activation contract), N a multiple of
// 32 (two 16-row tiles; the dense ops require N % 128 anyway).
inline bool shape_served(int M, int N, int K)
{
    return M >= 1 && M <= kMaxM && K > 0 && K % 128 == 0 && N > 0 && N % 32 == 0;
}

// ---------------------------------------------------------------------------------------------------------- plan
struct Smem
{
    int rs;        // bytes per staged activation row (chunk elements + 32)
    int cb;        // blocks per chunk (the whole K when resident)
    int xq_bytes;
    int xb_bytes;
    int red_bytes;
    int total;
};

// The kernel's shared-memory layout: x codes [8 nc][rs], their scale bytes [8 nc][cb], then (resident mode, cj == 0)
// the reduction buffer [wk][rt][8 nc][kRedStride] floats; with K chunks (cj > 0: cj K blocks per warp per chunk,
// cj * wk blocks per chunk) the reduction buffer aliases the x buffer.
FSO_SMALLM_HD inline Smem smem_layout(int nc, int rt, int wk, int K, int cj)
{
    Smem s;
    int const nb = K / 32;
    s.cb = cj == 0 ? nb : cj * wk;
    if (s.cb > nb)
        s.cb = nb;
    s.rs = s.cb * 32 + 32;
    s.xq_bytes = 8 * nc * s.rs;
    s.xb_bytes = (8 * nc * s.cb + 15) / 16 * 16;
    s.red_bytes = wk * rt * 8 * nc * kRedStride * 4;
    int const x = s.xq_bytes + s.xb_bytes;
    s.total = cj == 0 ? x + s.red_bytes : (x > s.red_bytes ? x : s.red_bytes);
    return s;
}

struct Plan
{
    bool use = false;       // take the small-M route
    int nc = 0;             // 8-row column groups the MMA covers (ceil(M / 8), or one more: see config())
    int rt = 0;             // 16-row weight tiles per tile group
    int wk = 0;             // warps per CTA (K split): 4 or 8
    int cj = 0;             // 0: the activation is staged once per CTA; > 0: K chunks of cj blocks per warp
    int smem = 0;           // dynamic shared memory bytes the layout needs
};

inline bool fits(int nc, int rt, int wk, int K, int cj)
{
    return smem_layout(nc, rt, wk, K, cj).total <= kSmemPerBlockMax - kSmemMargin;
}

// The configuration of one call (whether or not a rule picks the route), from the shape alone, so that G1 and G2
// get the same one and their bytes agree (the summation order depends on wk only). Chosen on the Q120a sweep of
// every (nc, rt, wk, chunk) the kernel can run (runs/Q120b/fit/policy_search.py): over the 152 cells this policy is
// at the per-cell best on half of them and within 6 % on average; the sweep's best is a configuration this policy
// does not pick mostly on cells the rules do not route (M >= 48 at K >= 2048).
//   * nc = ceil(M / 8); when that is 1 and N >= 4096, 2: the MMA chains of a warp are nc * rt, and one chain per
//     warp leaves the tensor core idle between dependent MMAs (M <= 8 at N >= 4096: 2-21 % faster with two);
//   * on short K (at most 4 K blocks per warp with 8 warps, K <= 1024) two tiles per tile group, 8 warps if the
//     activation and the reduction buffer stay resident, else 4; on longer K one tile, 8 warps resident, else 4
//     resident;
//   * when nothing keeps the activation resident, 8 warps with the largest K chunk that fits (cj >= 1 always
//     does; the activation is then re-staged for every tile group of the CTA, which costs on CTAs that own several
//     tile groups, and the extra chain of the first rule is dropped if that makes the activation resident again).
inline Plan config(int M, int N, int K)
{
    Plan p;
    int const nc0 = (M + 7) / 8;
    int const nb = K / 32;
    bool const short_k = nb / 8 <= 4;
    p.nc = (nc0 == 1 && N >= 4096) ? 2 : nc0;
    // (wk, rt) in order of preference; the first that keeps the activation resident wins.
    int const order_short[4][2] = {{8, 2}, {4, 2}, {8, 1}, {4, 1}};
    int const order_long[2][2] = {{8, 1}, {4, 1}};
    int const n_order = short_k ? 4 : 2;
    p.wk = 0;
    for (int i = 0; i < n_order; ++i)
    {
        int const wk = short_k ? order_short[i][0] : order_long[i][0];
        int const rt = short_k ? order_short[i][1] : order_long[i][1];
        if (fits(p.nc, rt, wk, K, 0))
        {
            p.wk = wk;
            p.rt = rt;
            p.cj = 0;
            break;
        }
    }
    if (p.wk == 0)
    {
        p.wk = 8;
        p.rt = short_k ? 2 : 1;
        if (p.nc > nc0 && fits(nc0, p.rt, p.wk, K, 0))
        {
            p.nc = nc0;
            p.cj = 0;
        }
        else
        {
            int const nbw = (nb + p.wk - 1) / p.wk;
            p.cj = 0;
            for (int cj = 1; cj <= nbw; ++cj)
                if (fits(p.nc, p.rt, p.wk, K, cj))
                    p.cj = cj;
            // cj = 1 (8 blocks of 32 elements per chunk, 8 * nc rows) always fits; keep the guard anyway.
            if (p.cj == 0)
            {
                p.smem = kSmemPerBlockMax + 1;
                return p;
            }
        }
    }
    p.smem = smem_layout(p.nc, p.rt, p.wk, K, p.cj).total;
    return p;
}

// R1 (the 2b call): pad8(M) * K * (1 + N / kN0R1) <= kBudgetR1, for the plan config() picks for the shape when it
// passes R1's plan guards (kR1ResidentOnly: the activation resident in shared memory; nc <= kR1NcMax; kR1OneRound: the
// N / (16 rt) tile groups fit one round of the persistent grid on a device of `sms` multiprocessors). `sms` is the
// device's SM count (the launch paths pass the current device's; the host query op takes it as an argument).
inline bool rule_r1(int M, int N, int K, int sms)
{
    if (M < 1 || M > kMaxM)
        return false;
    double const s = double((M + 7) / 8 * 8) * double(K);
    if (s * (1.0 + double(N) / kN0R1) > kBudgetR1)
        return false;
    if (kR1ResidentOnly || kR1NcMax < 8 || kR1OneRound)
    {
        Plan const c = config(M, N, K);
        if (kR1ResidentOnly && c.cj != 0)
            return false;
        if (c.nc > kR1NcMax)
            return false;
        if (kR1OneRound && N / (16 * c.rt) > sms)
            return false;
    }
    return true;
}

// R2 (the bf16-x call): K <= kR2KMax, f(M) <= kBudgetR2 and, above the first 8-row bucket, f(pad8(M)) <= kBudgetR2 as
// well, with f(m) = m * K * (1 + N / kN0R2) * (K / 2048)^q; and R1.
inline double r2_kq(int K)
{
    double const r = double(K) / 2048.0;
    return kR2KExpQuarters == 0 ? 1.0 : (kR2KExpQuarters == 1 ? std::sqrt(std::sqrt(r)) : std::sqrt(r));
}

inline double r2_cost(int m, int N, int K)
{
    return double(m) * double(K) * (1.0 + double(N) / kN0R2) * r2_kq(K);
}

inline bool rule_r2(int M, int N, int K, int sms)
{
    if (M < 1 || M > kMaxM || K > kR2KMax)
        return false;
    if (r2_cost(M, N, K) > kBudgetR2)
        return false;
    if (M > 8 && r2_cost((M + 7) / 8 * 8, N, K) > kBudgetR2)
        return false;
    return rule_r1(M, N, K, sms);
}

// The route of one call: `bf16x` = the bf16-x call (R2), else the 2b call (R1), on a device of `sms` multiprocessors.
inline Plan plan(int M, int N, int K, bool bf16x, int sms)
{
    if (!shape_served(M, N, K))
        return Plan{};
    int const md = mode();
    if (md == kModeOff)
        return Plan{};
    Plan p = config(M, N, K);
    if (p.smem > kSmemPerBlockMax - kSmemMargin || N % (16 * p.rt) != 0)
        return Plan{};
    p.use = md == kModeOn || (bf16x ? rule_r2(M, N, K, sms) : rule_r1(M, N, K, sms));
    return p;
}

// The persistent grid: at most one CTA per SM (each CTA requests at least half an SM's shared memory, so two never
// share one: a grid of two-per-SM CTAs launched under PDL while the previous kernel still holds SMs packs onto the
// free ones, measured up to 30 % slower in Q120a), balanced so that every CTA owns the same number of tile groups
// +- 1. `smem_per_sm` is the device's shared memory per multiprocessor.
inline int grid(Plan const& p, int N, int num_sms)
{
    int const ntg = N / (16 * p.rt);
    int const g0 = ntg < num_sms ? ntg : num_sms;
    if (g0 <= 0)
        return 1;
    int const per = (ntg + g0 - 1) / g0;
    return (ntg + per - 1) / per;
}

inline int launch_smem(Plan const& p, int smem_per_sm)
{
    int const half = smem_per_sm / 2 + 1;   // more than half: one CTA per SM
    int const want = p.smem > half ? p.smem : half;
    return want < kSmemPerBlockMax ? want : kSmemPerBlockMax;
}

} // namespace fso::sm120_smallm
