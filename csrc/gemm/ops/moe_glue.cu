/*
 * MoE routing/combine glue kernels for the sm_120 grouped MXFP8 path (M1).
 *
 * The M1 layer composition initially built its masked-layout index tensors
 * with ~8 torch ops (argsort / scatter_add / cumsum / index fills) and its
 * weighted combine with ~4 more. Under CUDA-graph replay each tiny kernel
 * still costs its GPU-side gap, and the measured glue (~45 us/layer at M=1
 * on RTX 5090) exceeded the grouped GEMMs themselves — the case-05
 * launch-economics failure reproduced inside our own pipeline. These two
 * kernels replace all of it:
 *
 *   moe_build_routing : topk_ids -> (masked_m, row_map, slot_of_flat) in ONE
 *     launch. Single CTA, shared-memory histogram (G <= kMaxGroups), atomic
 *     rank assignment. Slot order within a group is atomic-arrival order —
 *     a permutation of the sorted recipe, semantically equivalent (every
 *     consumer goes through row_map / slot_of_flat consistently). row_map
 *     slots at or beyond masked_m[g] are left uninitialised on purpose: no
 *     consumer reads them, and skipping the G*m_cap clear matters at decode.
 *   moe_combine : out[t] = sum_j topk_w[t,j] * dn_flat[slot_of_flat[t*topk+j]]
 *     vectorised 8 bf16 per thread.
 *
 * Both are capture-safe: fixed shapes, no host reads, masked_m content is
 * produced on device. Layer composition after this file: routing(1) +
 * gather-quant(1) + grouped gate_up(1) + silu-quant(1) + grouped down(1) +
 * combine(1) = 6 kernels, zero torch-op glue.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cstdint>
#include <cuda_bf16.h>

namespace blockscale_gemm
{
namespace
{

constexpr int kMaxGroups = 1024;
constexpr int kRoutingThreads = 512;

__global__ void moe_build_routing_kernel(
    int32_t const* __restrict__ topk_ids, // [M * topk]
    int32_t* __restrict__ masked_m,       // [G]
    int32_t* __restrict__ row_map,        // [G * m_cap] (valid slots only)
    int32_t* __restrict__ slot_of_flat,   // [M * topk]
    int num_pairs, int topk, int num_groups, int m_cap, bool pdl)
{
    // PDL: order the topk_ids read behind the parent (upstream layer /
    // router), then release the dependent's prologue.
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    __shared__ int32_t cnt[kMaxGroups];
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        cnt[g] = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < num_pairs; i += blockDim.x)
    {
        int const e = topk_ids[i];
        int const r = atomicAdd(&cnt[e], 1);
        int const slot = e * m_cap + r;
        row_map[slot] = i / topk; // source token row
        slot_of_flat[i] = slot;
    }
    __syncthreads();

    for (int g = threadIdx.x; g < num_groups; g += blockDim.x)
        masked_m[g] = cnt[g];
}

__global__ void moe_combine_kernel(
    __nv_bfloat16 const* __restrict__ dn, // [G * m_cap, H]
    int32_t const* __restrict__ slot_of_flat, // [M * topk]
    float const* __restrict__ topk_w,     // [M, topk]
    __nv_bfloat16* __restrict__ out,      // [M, H]
    int M, int topk, int H, bool pdl)
{
    // PDL entry (see moe_build_routing_kernel).
    if (pdl)
    {
        cudaGridDependencySynchronize();
        if (threadIdx.x == 0)
        {
            cudaTriggerProgrammaticLaunchCompletion();
        }
    }
    constexpr int kVec = 8; // 8 bf16 = 16 bytes per thread
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const num_hvec = H / kVec;
    if (tid >= M * num_hvec)
        return;
    int const t = tid / num_hvec;
    int const h = (tid % num_hvec) * kVec;

    float acc[kVec];
#pragma unroll
    for (int v = 0; v < kVec; ++v)
        acc[v] = 0.f;

    for (int j = 0; j < topk; ++j)
    {
        int const slot = slot_of_flat[t * topk + j];
        float const w = topk_w[t * topk + j];
        // LDG.128: 8 bf16 from the routed row
        float4 const raw = *reinterpret_cast<float4 const*>(
            &dn[static_cast<int64_t>(slot) * H + h]);
        __nv_bfloat162 const* v2 = reinterpret_cast<__nv_bfloat162 const*>(&raw);
#pragma unroll
        for (int p = 0; p < kVec / 2; ++p)
        {
            float2 const f = __bfloat1622float2(v2[p]);
            acc[2 * p] += w * f.x;
            acc[2 * p + 1] += w * f.y;
        }
    }

    __nv_bfloat162 packed[kVec / 2];
#pragma unroll
    for (int p = 0; p < kVec / 2; ++p)
        packed[p] = __floats2bfloat162_rn(acc[2 * p], acc[2 * p + 1]);
    *reinterpret_cast<float4*>(&out[static_cast<int64_t>(t) * H + h])
        = *reinterpret_cast<float4 const*>(&packed[0]);
}

} // anonymous namespace


// PDL launch helper (mirrors quant_kernels.cu): programmatic-stream-
// serialization attribute so downstream launches overlap this kernel.
// FSO_DISABLE_PDL=1 restores plain serialised launches.
static inline bool fso_pdl_enabled()
{
    static bool v = []
    {
        char const* e = std::getenv("FSO_DISABLE_PDL");
        return !(e && e[0] == '1');
    }();
    return v;
}

// See quant_kernels.cu: huge programmatic dependents flood the SMs and
// throttle the parent's tail, so large grids launch plain.
constexpr unsigned kFsoPdlMaxGridCtas = 4096;

template <typename KernelT, typename... Args>
static inline void fso_pdl_launch(KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args)
{
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args..., pdl);
}


// moe_build_routing: topk_ids [M, topk] int32 -> (masked_m [G] int32,
// row_map [G * m_cap] int32, slot_of_flat [M * topk] int32), one launch.
std::tuple<at::Tensor, at::Tensor, at::Tensor> moe_build_routing(
    at::Tensor topk_ids, int64_t num_groups, int64_t m_cap)
{
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.dtype() == at::kInt,
        "topk_ids must be CUDA int32");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [M, topk]");
    TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
    TORCH_CHECK(num_groups >= 1 && num_groups <= kMaxGroups,
        "num_groups must be in [1, ", kMaxGroups, "]");
    TORCH_CHECK(m_cap >= 1 && m_cap % 4 == 0, "m_cap must be a positive multiple of 4");
    int const M = topk_ids.size(0);
    int const topk = topk_ids.size(1);
    // Caller contract: with no-replacement topk routing a group receives at
    // most M rows, so m_cap >= M guarantees no slot overflow.
    TORCH_CHECK(m_cap >= M, "m_cap must be >= M (per-expert count can reach M)");

    auto opts = topk_ids.options();
    auto masked_m = at::empty({num_groups}, opts);
    auto row_map = at::empty({num_groups * m_cap}, opts);
    auto slot_of_flat = at::empty({static_cast<int64_t>(M) * topk}, opts);

    auto stream = at::cuda::getCurrentCUDAStream();
    fso_pdl_launch(moe_build_routing_kernel, dim3(1), dim3(kRoutingThreads), stream, fso_pdl_enabled(),
        reinterpret_cast<int32_t const*>(topk_ids.data_ptr()),
        reinterpret_cast<int32_t*>(masked_m.data_ptr()),
        reinterpret_cast<int32_t*>(row_map.data_ptr()),
        reinterpret_cast<int32_t*>(slot_of_flat.data_ptr()),
        M * topk, topk, static_cast<int>(num_groups), static_cast<int>(m_cap));
    return {masked_m, row_map, slot_of_flat};
}


// moe_combine: dn [G, m_cap, H] bf16 + slot_of_flat [M*topk] + topk_w [M,topk]
// fp32 -> out [M, H] bf16 (weighted sum over the topk routed rows).
at::Tensor moe_combine(at::Tensor dn, at::Tensor slot_of_flat, at::Tensor topk_w)
{
    TORCH_CHECK(dn.is_cuda() && dn.dtype() == at::kBFloat16, "dn must be CUDA bf16");
    TORCH_CHECK(dn.dim() == 3 && dn.is_contiguous(), "dn must be contiguous [G, m_cap, H]");
    TORCH_CHECK(slot_of_flat.dtype() == at::kInt && slot_of_flat.is_contiguous(),
        "slot_of_flat must be contiguous int32");
    TORCH_CHECK(topk_w.dtype() == at::kFloat && topk_w.dim() == 2 && topk_w.is_contiguous(),
        "topk_w must be contiguous fp32 [M, topk]");
    int const M = topk_w.size(0);
    int const topk = topk_w.size(1);
    int const H = dn.size(2);
    TORCH_CHECK(H % 8 == 0, "H must be a multiple of 8 (16-byte vectorised combine)");
    TORCH_CHECK(slot_of_flat.numel() == static_cast<int64_t>(M) * topk,
        "slot_of_flat must have M*topk entries");

    auto out = at::empty({M, H}, dn.options());
    int const threads = 256;
    int64_t const total = static_cast<int64_t>(M) * (H / 8);
    int const grid = static_cast<int>((total + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();
    bool const pdl = fso_pdl_enabled() && static_cast<unsigned>(grid) <= kFsoPdlMaxGridCtas;
    fso_pdl_launch(moe_combine_kernel, dim3(grid), dim3(threads), stream, pdl,
        reinterpret_cast<__nv_bfloat16 const*>(dn.data_ptr()),
        reinterpret_cast<int32_t const*>(slot_of_flat.data_ptr()),
        reinterpret_cast<float const*>(topk_w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        M, topk, H);
    return out;
}

} // namespace blockscale_gemm
