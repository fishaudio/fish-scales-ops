/*
 * sm_120 / sm_121 small-M dense MXFP8 route ("sm120.smallm", issue #12 decision 3): the host entry points the ATen
 * translation unit (ops/mxfp8.cu) calls. The rules and the configuration are in
 * blockscale_gemm/arch/sm120/mxfp8/smallm_route.h, the kernel in smallm_tc.cuh, its instantiations in the four
 * ops/mxfp8_smallm_{q,q_epi,bf16,bf16_epi}_kernel.cu translation units.
 */
#include "blockscale_gemm/arch/sm120/common/env_overrides.cuh"
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_route.h"
#include "blockscale_gemm/arch/sm120/mxfp8/smallm_tc.cuh"
#include "blockscale_gemm/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace blockscale_gemm::detail
{

namespace
{

namespace sm = tensorrt_llm::kernels::blockscale_gemm::sm120_smallm;
namespace rt = fso::sm120_smallm;

constexpr int kMaxDevices = 64;

struct DeviceInfo
{
    int sms = 0;
    int smem_per_sm = 0;
};

// SM count and shared memory per SM of the current device, read once per device (host queries, no stream work).
DeviceInfo device_info()
{
    static DeviceInfo s_info[kMaxDevices];
    static std::once_flag s_once[kMaxDevices];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= kMaxDevices)
        dev = 0;
    std::call_once(s_once[dev], [dev] {
        cudaDeviceGetAttribute(&s_info[dev].sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&s_info[dev].smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    });
    return s_info[dev];
}

// The opt-in shared-memory attribute of every kernel of the route, once per process (the first call of the route;
// the capture contract makes it an eager one, so cudaFuncSetAttribute never runs inside a capture).
cudaError_t init_once()
{
    static cudaError_t s_err = cudaErrorNotReady;
    static std::once_flag s_once;
    std::call_once(s_once, [] {
        cudaError_t e = sm::smallm_init<sm::kSmallmFormQ, false>();
        if (e == cudaSuccess)
            e = sm::smallm_init<sm::kSmallmFormQ, true>();
        if (e == cudaSuccess)
            e = sm::smallm_init<sm::kSmallmFormBf16, false>();
        if (e == cudaSuccess)
            e = sm::smallm_init<sm::kSmallmFormBf16, true>();
        s_err = e;
    });
    return s_err;
}

} // namespace

// The route of one call, as ten integers for the host query op (mxfp8_dense_smallm_route_sm120): use, nc, rt, wk,
// cj (K blocks per warp per chunk, 0 = the activation resident), layout smem bytes, launch smem bytes, grid, R1, R2
// (the rules' verdicts, whatever FSO_SMALLM_ROUTE says). Host only.
void sm120_smallm_route_query(int M, int N, int K, bool bf16x, int64_t* out10)
{
    DeviceInfo const d = device_info();
    rt::Plan const p = rt::plan(M, N, K, bf16x, d.sms);
    rt::Plan const c = rt::config(M, N, K);
    out10[0] = p.use ? 1 : 0;
    out10[1] = c.nc;
    out10[2] = c.rt;
    out10[3] = c.wk;
    out10[4] = c.cj;
    out10[5] = c.smem;
    out10[6] = rt::launch_smem(c, d.smem_per_sm);
    out10[7] = c.rt > 0 && N % (16 * c.rt) == 0 ? rt::grid(c, N, d.sms) : 0;
    out10[8] = rt::rule_r1(M, N, K, d.sms) ? 1 : 0;
    out10[9] = rt::rule_r2(M, N, K, d.sms) ? 1 : 0;
}

// The decision alone: whether the 2b (bf16x = false) or the bf16-x call (bf16x = true) of this shape takes the route
// in this process on the current device (its SM count read once per device, a host attribute query).
bool sm120_smallm_takes(int M, int N, int K, bool bf16x)
{
    return rt::plan(M, N, K, bf16x, device_info().sms).use;
}

// One call of the route. `form`: sm::kSmallmFormQ (x fp8 + sx) or sm::kSmallmFormBf16 (x bf16, sx unused). `epi`: the
// dense epilogue (identity, or bias / erf-GELU / residual; the quantized output is refused before this is reached).
// The plan is plan(M, N, K, form == bf16); the caller has checked that it says use.
cudaError_t launch_sm120_mxfp8_smallm(int form, void const* x, int32_t const* sx, __nv_fp8_e4m3 const* w,
    int32_t const* ws, __nv_bfloat16* y, int M, int N, int K, fso::DenseEpilogue const& epi, cudaStream_t stream)
{
    rt::Plan const p = rt::config(M, N, K);
    if (p.smem > rt::kSmemPerBlockMax - rt::kSmemMargin || N % (16 * p.rt) != 0)
        return cudaErrorInvalidValue;
    cudaError_t const ie = init_once();
    if (ie != cudaSuccess)
        return ie;
    DeviceInfo const d = device_info();
    sm::SmallmParams prm = {};
    prm.w = reinterpret_cast<uint8_t const*>(w);
    prm.ws = ws;
    prm.x = x;
    prm.sx = sx;
    prm.y = y;
    prm.epi = epi;
    prm.M = M;
    prm.N = N;
    prm.K = K;
    prm.cj = p.cj;
    int const grid = rt::grid(p, N, d.sms);
    int const smem = rt::launch_smem(p, d.smem_per_sm);
    bool const pdl = tensorrt_llm::kernels::blockscale_gemm::fso_pdl_enabled();
    bool const has_epi = !epi.is_identity();
    bool const chunked = p.cj > 0;   // the chunked kernel (wk == 8 only, as config() produces chunked plans)
    if (form == sm::kSmallmFormQ)
        return has_epi
            ? sm::smallm_launch<sm::kSmallmFormQ, true>(prm, p.nc, p.rt, p.wk, chunked, grid, smem, pdl, stream)
            : sm::smallm_launch<sm::kSmallmFormQ, false>(prm, p.nc, p.rt, p.wk, chunked, grid, smem, pdl, stream);
    return has_epi
        ? sm::smallm_launch<sm::kSmallmFormBf16, true>(prm, p.nc, p.rt, p.wk, chunked, grid, smem, pdl, stream)
        : sm::smallm_launch<sm::kSmallmFormBf16, false>(prm, p.nc, p.rt, p.wk, chunked, grid, smem, pdl, stream);
}

} // namespace blockscale_gemm::detail
