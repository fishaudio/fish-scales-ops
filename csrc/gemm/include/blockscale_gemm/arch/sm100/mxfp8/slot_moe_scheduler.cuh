/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The tile scheduler of the sm_100 / sm_103 persistent FC1 + FC2 slot kernel (issue #13 lever 5,
// `FsoSm100SlotGemmMoe`, run fso_issue13_20261031/I13-L5; design note I13-D45 section 4).
//
// What the kernel runs
// --------------------
// One grid runs every tile of a MoE layer's two grouped GEMMs on the compact slot route (row capacity m_cap <= 64,
// one 128-row block per active expert): first every FC1 tile (the fused SwiGLU + MXFP8 requantize GEMM, weight
// panels of w13), then every FC2 tile (the plain bf16 down projection, weight panels of w2). An "entry" is one live
// slot, i.e. one active expert; L entries are live (the routing op's blk0[G]). Entry i has T1 = ceil(N1 / 128) FC1
// tiles and T2 = ceil(N2 / 128) FC2 tiles (Family B: 12 and 16).
//
// Tile numbering. Over the LIVE tiles of both phases: tile t < T1 * L is FC1 tile t (entry t / T1, weight panel
// t % T1); tile t < (T1 + T2) * L is FC2 tile u = t - T1 * L (entry u / T2, panel u % T2). The FC1 and FC2 tiles of
// one entry share the entry index, which is also the token-side batch coordinate (the slot = blk0[expert]). Every
// tile t < (T1 + T2) * L is live, every later one is past the work: dead slots and padded rows produce no entry and
// so no tile, as in the two-kernel path.
//
// Hand-out order: every tile is drawn. A CTA starts with a placeholder tile that every warp role skips (it is valid,
// so the roles' loops run once, and dead, so they do no work), and its scheduler warp hands it tile
// atomicAdd(&counter[0], 1), two tiles ahead of the load warp (generator edit c9), until the draws pass the live
// tiles. Since the counter is monotone and the FC1 tiles have the lower numbers, every FC1 tile is handed out before
// any FC2 tile, and in every CTA's own queue its FC1 tiles precede its FC2 tiles.
//
// Why there is no static FC1 first wave (D45 section 4.2: CTA b takes FC1 tile b without a draw, as today's slot
// kernels start). It ties FC1 tile b to CTA b, so an FC2 tile can wait for a CTA that has not landed, and the kernel
// then needs every CTA of its grid to become resident eventually. Two of these kernels running at the same time on
// two streams break that between them: each one's resident CTAs spin on FC1 tiles of its own pending CTAs, and the
// other's spinning CTAs hold the SMs those would land on. Measured on the B300 (run fso_issue13_20261031/I13-L5,
// scripts/l5_residency.py, tests/g1_residency): with a blocker kernel holding 60 SMs and the second layer call on a
// higher-priority stream, the static wave deadlocked at M = 16 and M = 64, while drawing every tile completed every
// round at every blocker width. Two layer calls on two streams are part of the layer's contract, so the static wave
// is not offered, not even as an A/B arm. For the same reason there is no all-static walk: it would hand a CTA an
// FC2 tile it can only wait for.
//
// The first draw. After the kernel-wide griddepcontrol.wait every thread loads L and the packed facts of slot b / T1
// (`fso_resolve_first`; the slot is derived in the prologue) before the scheduler warp's first draw. The facts value
// is not used, but these loads made the layer at M = 1 about 1.5 us faster than drawing straight after the wait
// (I13-L5 tests xa / m1a: 16.2 against 17.8 us per layer, two runs), and without the facts load the layer with eight
// active experts was 2.2-3.3 % slower at M = 16-64 (run fso_issue12_20261028/M4, step-1 screen). The mechanism is
// inferred, not isolated: the load brings the line of the packed facts into the SM's L1, so the scheduler warp's
// lookup after its draw hits there instead of going to L2. ptxas removes a load whose value nothing reads, so
// `fso_keep_facts` consumes it in a prefetch that never runs; the two loads go out back to back after the wait.
//
// The FC1 -> FC2 hand-off (D45 section 4.3, the hybrid form). An FC2 tile of entry i reads FC1's output rows and
// scale words of entry i, which the 12 (T1) FC1 tiles of entry i write. Each FC1 tile's epilogue releases one count
// on done[i] (counter[2 + i]) after its stores (`red.release.gpu`, in the kernel). The scheduler warp, when it draws
// an FC2 tile of entry i, performs ONE `ld.acquire.gpu` of done[i] and publishes the verdict done[i] >= T1 as the
// tile's ready bit; it does not wait. The load warp then either issues the tile's stages as for any tile (ready:
// after a `fence.proxy.async.global`, because the TMA reads through the async proxy what the acquire ordered in the
// generic proxy), or issues the weights first, spins on done[i] itself, fences and then issues the activation
// (FsoMoeMainloop::fso_load_after_entry). The chain of the ready path is FC1's release -> this warp's acquire (GPU
// scope) -> the response's mbarrier arrival (release, CTA scope) -> the load warp's mbarrier wait (acquire) -> the
// proxy fence -> the TMA. The first tile of a CTA and every FC1 tile carry no ready bit.
//
// Freedom from deadlock. (1) Every FC1 tile is handed out before every FC2 tile (the counter's order). (2) A CTA runs
// its tiles in the order it received them, so its FC1 tiles precede its FC2 tiles. (3) An FC1 tile waits on nothing
// inside the kernel. (4) An FC2 tile waits only on the FC1 tiles of its own entry, each of which was drawn by a CTA
// that was running when it drew it and that reaches it without waiting on anything. So the kernel finishes whatever
// else shares the machine and however few of its CTAs are resident at a time; a CTA that lands after the work is
// handed out draws past the live tiles and exits. (A static FC1 first wave would break (4): there an FC2 tile can
// wait on a CTA that has not landed, see above.)
//
// The deferred draw (both modes; issue #13 follow-up, run fso_issue12_20261028/FB5). The scheduler warp draws two tiles
// ahead of the load warp. When a call has few live (expert, row) pairs, its tiles fill the grid about one and a half
// times: every CTA draws one tile in the first round, and the second round's tiles are FC2 tiles. A CTA whose first
// tile is an FC2 tile published not ready could then win a second-round tile while its load warp spins in the first.
// That second tile's stages stay the first tile's until the first one's FC1 dependency clears and its MMA frees them,
// so it starts after every other tile and ends the kernel about two microseconds after the rest. Which CTAs win the
// second round is a race that code layout tips either way. Measured on the B300 at Family B ep2 M = 1 in the
// published bench: 8 or 9 of the 112 FC2 tiles queued so, and the kernel ended 12.8 us after its wait against 10.8 us
// where none did; the layer took 17.1 us, against 15.7 us with the hold below.
// So in that regime (the second-round gate below) a scheduler warp whose last published tile is an FC2 tile published
// not ready holds its next draw until either every tile has been handed out (the draw counter has passed the live
// tiles: the next draw can only end the CTA's work) or that tile's FC1 count is complete. It polls the two words with a
// relaxed load of the counter and the acquire the load warp itself spins with, 256 ns apart. Holding the draw only
// while tiles remain to be handed out matters: holding it until the dependency clears (or, equivalently, until the
// CTA's own load warp has passed it) delays the CTA's last draw and with it the exit of its warps and the launch of the
// dependent kernel, which cost 1 to 2.5 us per layer in measurement.
// The second-round gate, evaluated by the holding warp from L (read after the kernel-wide wait): the launch has a second
// round, and its tiles past one per CTA all fit on CTAs whose first tile is an FC1 tile, grid < D <= grid + min(D1,
// grid) with D the live tiles and D1 the FC1 tiles. Without a second round every tile goes out in the first one; past
// the bound (for example Family B uniform M = 2) a holding CTA would keep tiles that no other CTA can take yet. In both
// cases the draw goes out at once and the scheduling stays as it was. Under the bound the token-tile mode draws single
// tiles (tile pairs need at least kGroupMinTilesPerCta tiles per CTA), so the gate and the counter count tiles in both
// modes. Freedom from deadlock holds as argued above: a holding CTA has drawn an FC2 tile, so every FC1 tile had been
// handed out before (1), each to a CTA that runs it without waiting (3); the hold ends at the latest when that tile's
// FC1 count completes, and every tile is still drawn. The tiles and their arithmetic are unchanged, so the outputs are
// the same bytes.
//
// The counters (int32 [2 + 2G], per call): counter[0] the draws, counter[1] the CTAs that made their last draw,
// counter[2 + i] done[i], and counter[2 + G + i] the packed facts of slot i (expert | rows << 16, -1 past the active
// experts). moe_build_routing_compact(..., with_slots=True) returns them (`slot_sched`): the counters zero, the facts
// written. The CTA whose last draw completes the grid puts counter[0] and counter[1] back to zero, as today's slot
// scheduler does; done[] is left at T1 for every live entry, so a counter tensor serves ONE call of this kernel and
// must come fresh (from the routing op of the same layer call) to the next.
//
// The token-tile mode (`ListMode`, the tag FsoSlotMoeWorkStealingListScheduler; issue #13 lever 5 step 3, run
// fso_issue13_20261031/I13-L5b). Past a row capacity of one token tile (m_cap > 64) an expert spans several 64-row
// token tiles and several 128-row blocks, and the two-kernel path walks the routing op's packed list of live token
// tiles (FsoSm100SlotGemmCompactList, slot_work_stealing.cuh's tile-list mode). Here an entry is one entry of that
// list (q | e << 14 | (r - 1) << 24, q = 2 * block + half; L = the list's count word, num_slots = the list's capacity),
// numbered exactly as above, and the scheduler rewrites the coordinate to the entry's block (L_idx) and half (N_idx)
// when it resolves the tile, so `get<3>` of the kernel's coordinate is the block. What changes is the hand-off's
// counter. An FC2 tile's activation-scale (SFB) load reads the whole 128-row scale atom of its block: CUTLASS's
// TileN-64 handling maps N tile h to atom h / 2 and shifts the TMEM scale address by h % 2 in the MMA. The atom's words
// of the block's two 64-row halves are interleaved (word (m % 32) * 4 + (m % 128) / 32), and the other half is a
// different entry whose FC1 tiles may still be storing its words. The FC2 tile never uses those words, but it would
// read bytes another thread is writing (D45 section 4.3). So the counters are per BLOCK: the FC1 tiles of both halves
// release into the block's word, and the FC2 tiles of either half wait until both halves' FC1 tiles have released.
// The block's word (int32 [B] after the 2 + 2G words above, B = the routing op's block count) is written by the
// routing op of the same layer call as live_halves << 20 (bits 20-21: how many of the block's halves hold rows, 1 or 2,
// i.e. the expected FC1 count in units of T1), and each FC1 tile adds one to it, so the block is complete when the word
// reaches live_halves << 20 | T1 * live_halves. The scheduler warp's one acquire per FC2 draw reads the word (after the
// list entry that names the block: one dependent hop, two tiles ahead of the load warp, the entry's line in L1 after
// the first draws), computes that target from the word's own high bits and publishes it with the tile; the load warp
// spins on `done >= target` exactly as in slot mode (`done_target(tile)`). The list holds a block's two entries next
// to each other (sorted by q), so their FC1 tiles are drawn back to back. Freedom from deadlock is argument (1)-(4)
// above with "its own entry" read as "its block's one or two entries": every FC1 tile, of either half, is handed out
// before any FC2 tile.
//
// Tile groups (token-tile mode). One draw hands out a GROUP of two adjacent weight panels of one entry (the last
// panel of an entry alone when its panel count is odd), published to the CTA as two consecutive tiles: one atomic, one
// list lookup and, for an FC2 group, one acquire serve two tiles, and an FC1 group's epilogue releases its two counts
// once, after its second tile. Why: past one token tile of row capacity an expert's second and later token tiles read
// the weight panels the first one just streamed, from L2, so those tiles are short, and the per-tile hand-off
// operations -- the acquire on the scheduler warp's draw path (an ld.acquire.gpu also invalidates the SM's L1, so the
// next draw's list lookup goes to L2 too) and the GPU-scope release that ends every FC1 tile's epilogue -- were
// exposed: with one tile per draw the fused kernel was no faster than the two-kernel path at M = 1024 (run
// fso_issue13_20261031/I13-L5b, logs/x1.log and x2.log: the hand-off cost 8 to 11 us per layer there, about half
// each in the acquire and in the release). The group's tiles carry the draw's facts, ready bit and target; the
// panels are adjacent, so the second tile streams the weights right after the first's. Freedom from deadlock holds
// as before: groups are drawn in the tile order, a CTA runs a group's tiles back to back, and an FC1 group's release
// waits for nothing but its own second tile. Groups are used only when the launch has at least kGroupMinTilesPerCta
// live tiles per CTA (decided on the device from L, so every CTA numbers the draws alike): with few tiles per CTA a
// group is a coarser unit of the FC1 phase's last wave, and the FC2 tiles that wait for it lose more than the groups
// save. Measured with every token on the same eight experts (run I13-L5b, logs/x6.log, x7.log; per layer against the
// two-kernel path): at M = 256 (6 tiles per CTA) 43.4 us with groups, 36.9 with single tiles, 37.7 two kernels; at
// M = 512 (12) single tiles -1.1 %, groups +0.6 %; at M = 640 (15) single tiles +1.6 %, groups +1.1 %; at M = 768 (18)
// single tiles +3.0 %, groups +0.7 %. Hence the threshold of 14.
//
// The list in registers (token-tile mode). The acquire of an FC2 draw invalidates the SM's L1, so with the list read
// through L1 every FC2 draw paid three dependent L2 round trips: the draw's atomic, the list entry, the acquire. The
// scheduler warp therefore copies the list into its lanes' registers once, behind its first draw (entry j in lane
// j % 32's list_regs_[j / 32], up to 256 entries, which covers the layer's lists up to M of about 1100 for Family
// B), and looks entries up with a register select and a shuffle; a longer list is read from memory as before.
//
// The first-panel prefetch (token-tile mode only; issue #13, runs fso_issue13_20261031/I13-G0 form C and I13-G1). In
// token-tile mode CTA b's first tile is drawn, so the CTA does not know it in its prologue, and its load warp starts
// streaming only after the scheduler warp has copied the list into its registers and made the first draw. One
// epilogue thread (thread 128, the first thread of the epilogue warps 4-7, which idle until the first accumulator)
// therefore issues, right after the kernel-wide griddepcontrol.wait, bulk L2 prefetches (`cp.async.bulk.prefetch.L2`,
// 16 KB pieces) of the weight panel and the scale block of tile b, its guess at its first tile: FC1 panel b % T1 of the
// expert in list entry b / T1, whose packed word `fso_resolve_first` has just loaded. Whoever ends up drawing each tile,
// the union over the grid is the first draw round (its first half when the launch draws two tiles at a time), so the
// panels are in L2 when the first draws' loads ask for them. Both the list entry and L are loaded after the wait, so no
// routing product is read early and the PDL rule of the layer holds. A prefetch writes nothing, so the outputs are the
// same bytes either way; it is issued only for a word that is not -1, an entry below L, an expert below G and a panel
// below T1, so every address lies inside w13 / sw13. Measured with a diagnostic build at Family B M = 256 (I13-G0,
// holds 6 and 7): about 0.9 us per layer, because the scheduler warp's list copy leaves the load warp idle for a while
// after the wait. The slot mode does not prefetch (it lost there: its CTAs land as the gather-quantize's leave, so no
// gap is left, and the bulk requests queue in front of the CTA's own first loads). FSO_MOE_FIRST_PANEL_PREFETCH=0 (read
// once in the launcher) passes a null weight pointer, and the kernel then issues nothing.
//
// Requirements (enforced by the launcher): cluster 1x1x1; in slot mode the batch coordinate is the slot (m_cap <= 64),
// in token-tile mode the list and the per-block words come from the routing op of the same call; a grid of at most
// min(SMs, (T1 + T2) * num_slots) CTAs.

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include <cstdint>

namespace cutlass::gemm
{
// The tags the fused FC1 + FC2 launcher hands `FsoSm100SlotGemmMoe` as its TileSchedulerTag: slot mode (an entry is one
// active expert, m_cap <= 64) and token-tile mode (an entry is one entry of the routing op's live token-tile list).
struct FsoSlotMoeWorkStealingScheduler
{
};
struct FsoSlotMoeWorkStealingListScheduler
{
};
} // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail
{

// One GPU-scope acquire load and one GPU-scope release add: the two halves of the FC1 -> FC2 hand-off.
CUTLASS_DEVICE uint32_t fso_ld_acquire_gpu(int const* p)
{
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// The deferred draw's poll of the draw counter (slot_moe_scheduler's header): a GPU-scope relaxed load.
CUTLASS_DEVICE uint32_t fso_ld_relaxed_gpu(int const* p)
{
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}

CUTLASS_DEVICE void fso_red_release_gpu_add(int* p, uint32_t v)
{
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

CUTLASS_DEVICE void fso_fence_proxy_async_global()
{
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// The first draw's facts preload (see the header), in two halves. `fso_ld_facts` issues a plain global load of the
// word; `fso_keep_facts` consumes the value, so that ptxas keeps the load (it removes a load whose value nothing
// reads): it decides a prefetch of the facts line that never runs, since a packed facts word is -1 or
// expert | rows << 16 with rows <= 64, never -2.
CUTLASS_DEVICE uint32_t fso_ld_facts(int const* p)
{
    uint32_t v;
    asm volatile("ld.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
    return v;
}

CUTLASS_DEVICE void fso_keep_facts(uint32_t v, int const* p)
{
    asm volatile(
        "{\n"
        "  .reg .pred q;\n"
        "  setp.eq.s32 q, %0, -2;\n"
        "  @q prefetch.global.L1 [%1];\n"
        "}\n" ::"r"(v),
        "l"(p));
}

// The first-panel prefetch's one instruction (token-tile mode, see the header): a bulk L2 prefetch of `bytes` (a
// multiple of 16) at the 16-byte aligned `p`, through the SM's copy engine; it completes nothing the kernel waits on.
CUTLASS_DEVICE void fso_prefetch_l2_bulk(void const* p, uint32_t bytes)
{
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(p), "r"(bytes));
}

template <class ClusterShape_, uint32_t Stages_, bool ListMode_ = false>
class FsoSlotMoeWorkStealingTileScheduler
{
public:
    using ClusterShape = ClusterShape_;
    static constexpr bool IsDynamicPersistent = true;
    static constexpr uint32_t Stages = Stages_;
    // Token-tile mode (see the header): an entry is one entry of the packed live token-tile list, the hand-off's
    // counters are per 128-row block.
    static constexpr bool ListMode = ListMode_;
    // Token-tile mode: the rows of one list entry (the kernel's TileN; the list is packed for 64-row token tiles).
    static constexpr int32_t kTokenTileRows = 64;
    // Token-tile mode: the block word's live-halves field (bits 20 and up) and its FC1 count field (bits 0-19).
    static constexpr int kBlockHalvesShift = 20;
    // Token-tile mode: tiles per draw (see "Tile groups" in the header): one atomic, one list lookup and, for FC2, one
    // acquire hand out two adjacent weight panels of one entry, and an FC1 group releases its count once -- when the
    // launch has at least kGroupMinTilesPerCta live tiles per CTA; with fewer, one tile per draw.
    static constexpr int32_t kTilesPerDraw = ListMode ? 2 : 1;
    static constexpr int32_t kGroupMinTilesPerCta = 14;
    // Token-tile mode: registers per lane of the scheduler warp's copy of the list (lists of up to 256 entries; the
    // layer's list holds ceil(pairs / 64) + experts entries at most: 254 for Family B at M = 1024).
    static constexpr int kListRegs = ListMode ? 8 : 1;
    static_assert(cute::is_static_v<ClusterShape> && cute::size(ClusterShape{}) == 1,
        "the fso fused FC1 + FC2 slot scheduler publishes each drawn tile to its own CTA only (cluster 1x1x1)");

    struct CLCResponse
    {
        uint32_t data[4] = {0};
    };

    struct WorkTileInfo
    {
        int32_t M_idx = 0;
        int32_t N_idx = 0;
        int32_t L_idx = 0;
        bool is_valid_tile = false;
        // Resolved by the scheduler: whether the tile is live, the weights' batch index (the expert), the expert's
        // routed rows (the store bound), the phase (0 = FC1, 1 = FC2) and, for a drawn FC2 tile, whether its entry's
        // FC1 tiles had all released when it was drawn.
        bool live = false;
        int32_t a_batch = 0;
        int32_t rows = 0;
        int32_t phase = 0;
        bool ready = false;
        // Token-tile mode, a drawn FC2 tile: the value of its block's word at which every FC1 tile of the block has
        // released (live_halves << 20 | T1 * live_halves); the load warp's spin target. Unused in slot mode.
        uint32_t target = 0;
        // Token-tile mode, an FC1 tile: how many FC1 completions its epilogue releases after its stores (the tiles of
        // its group, on its last tile; 0 on the others). Unused in slot mode, where every FC1 tile releases one.
        int32_t release = 0;

        CUTLASS_HOST_DEVICE bool is_valid() const
        {
            return is_valid_tile;
        }

        CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile()
        {
            return {-1, -1, -1, false, false, 0, 0, 0, false};
        }

        CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const
        {
            return true;
        }

        CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const
        {
            return -1;
        }
    };

    struct Arguments
    {
        // int32 [2 + 2G]: draws, finished CTAs, done[G], then the packed slot facts (see the header).
        int* counter = nullptr;
        // Device scalar: the live entries L (blk0[G]).
        int const* live_slots = nullptr;
        // Weight tiles per entry of FC2 (T2); T1 comes from the problem shape (FC1's).
        int tiles_m2 = 0;
        // The packed active-expert list and the per-expert routed rows (routing products, read after a wait). The
        // kernel reads the same facts from the counter tensor's packed slot words; these are checked for presence.
        int const* slot_to_expert = nullptr;
        int const* masked_m = nullptr;
        int num_slots = 0;
        int num_groups = 0;
        // Token-tile mode: the packed live token-tile list (num_slots entries, -1 past the live ones; `live_slots`
        // points at its count word), and the per-block words int32 [B] (see the header). Slot mode: unused.
        int const* tile_list = nullptr;
        int* block_done = nullptr;
        // Token-tile mode, the first-panel prefetch (see the header): FC1's weights w13 [G, N1, K1] fp8 (null: no
        // prefetch), their per-group scale slab sw13 (N1 * K1 / 32 bytes per expert) and K1. Slot mode: unused.
        void const* first_panel_w = nullptr;
        void const* first_panel_sf = nullptr;
        int first_panel_k = 0;
    };

    struct SlotParams
    {
        int* counter = nullptr;
        int const* live_slots = nullptr;
        int32_t tiles_m1 = 1;
        int32_t tiles_m2 = 1;
        int32_t grid = 1;
        int const* slot_to_expert = nullptr;
        int const* masked_m = nullptr;
        int32_t num_slots = 0;
        int32_t num_groups = 0;
    };
    // Token-tile mode only, so that the slot mode's kernel parameters keep their layout.
    struct ListParams : SlotParams
    {
        int const* tile_list = nullptr;
        int* block_done = nullptr;
        char const* first_panel_w = nullptr;
        char const* first_panel_sf = nullptr;
        int32_t first_panel_k = 0;
    };
    using Params = cute::conditional_t<ListMode, ListParams, SlotParams>;

    using Pipeline = PipelineCLCFetchAsync<Stages, ClusterShape>;

    //
    // Host side
    //
    template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
    static Params to_underlying_arguments(ProblemShapeMNKL problem_shape_mnkl, TileShape tile_shape_mnk,
        AtomThrShape, ClusterShapeMNK, KernelHardwareInfo const& hw_info, Arguments const& args,
        [[maybe_unused]] void* workspace = nullptr)
    {
        Params p;
        p.counter = args.counter;
        p.live_slots = args.live_slots;
        p.tiles_m1 = static_cast<int32_t>(
            (cute::get<0>(problem_shape_mnkl) + cute::size<0>(tile_shape_mnk) - 1) / cute::size<0>(tile_shape_mnk));
        p.tiles_m2 = static_cast<int32_t>(args.tiles_m2 > 0 ? args.tiles_m2 : 1);
        // The slot bound: the problem's batch extent as the kernel's edit 5 hands it (num_slots).
        long long const slots = static_cast<long long>(cute::get<3>(problem_shape_mnkl));
        long long const total = (static_cast<long long>(p.tiles_m1) + p.tiles_m2) * slots;
        long long const sms = hw_info.sm_count > 0 ? hw_info.sm_count : 1;
        p.grid = static_cast<int32_t>(total < sms ? (total > 0 ? total : 1) : sms);
        p.slot_to_expert = args.slot_to_expert;
        p.masked_m = args.masked_m;
        p.num_slots = static_cast<int32_t>(args.num_slots);
        p.num_groups = static_cast<int32_t>(args.num_groups);
        if constexpr (ListMode)
        {
            p.tile_list = args.tile_list;
            p.block_done = args.block_done;
            p.first_panel_w = static_cast<char const*>(args.first_panel_w);
            p.first_panel_sf = static_cast<char const*>(args.first_panel_sf);
            p.first_panel_k = static_cast<int32_t>(args.first_panel_k);
        }
        return p;
    }

    template <class ProblemShapeMNKL, class TileShape, class AtomThrShape, class ClusterShapeMNK>
    CUTLASS_HOST_DEVICE static dim3 get_grid_shape(Params const& params, ProblemShapeMNKL, TileShape, AtomThrShape,
        ClusterShapeMNK, KernelHardwareInfo)
    {
        return dim3(static_cast<uint32_t>(params.grid), 1u, 1u);
    }

    template <class ProblemShape, class ElementAccumulator>
    static size_t get_workspace_size(
        Arguments const&, ProblemShape, KernelHardwareInfo const&, uint32_t, uint32_t = 1, uint32_t = 1)
    {
        return 0;
    }

    template <class ProblemShape, class ElementAccumulator>
    static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t, ProblemShape,
        KernelHardwareInfo const&, uint32_t, uint32_t = 1, uint32_t = 1, CudaHostAdapter* = nullptr)
    {
        return Status::kSuccess;
    }

    static bool can_implement(Arguments const& args)
    {
        if constexpr (ListMode)
            return args.counter != nullptr && args.live_slots != nullptr && args.tile_list != nullptr
                && args.block_done != nullptr && args.num_slots > 0 && args.num_groups > 0 && args.num_groups <= 1024
                && args.tiles_m2 > 0;
        else
            return args.counter != nullptr && args.live_slots != nullptr && args.slot_to_expert != nullptr
                && args.masked_m != nullptr && args.num_slots > 0 && args.num_groups > 0 && args.num_groups <= 4096
                && args.tiles_m2 > 0;
    }

    //
    // Device side
    //
    CUTLASS_DEVICE
    FsoSlotMoeWorkStealingTileScheduler(CLCResponse* clc_response_ptr, Params const& params, dim3)
        : clc_response_ptr_(clc_response_ptr)
        , params_(params)
    {
    }

    // The live entries, read once per thread after a griddepcontrol.wait.
    CUTLASS_DEVICE int32_t live_entries()
    {
        if (live_ < 0)
        {
            int32_t const l = *params_.live_slots;
            live_ = l < 0 ? 0 : (l > params_.num_slots ? params_.num_slots : l);
        }
        return live_;
    }

    CUTLASS_DEVICE int32_t live_tiles()
    {
        return (params_.tiles_m1 + params_.tiles_m2) * live_entries();
    }

    // Tile index -> phase, weight panel, entry. Only called for t < live_tiles().
    CUTLASS_DEVICE WorkTileInfo decode(int32_t t)
    {
        WorkTileInfo w{};
        int32_t const fc1 = params_.tiles_m1 * live_entries();
        if (t < fc1)
        {
            w.phase = 0;
            w.L_idx = t / params_.tiles_m1;
            w.M_idx = t - w.L_idx * params_.tiles_m1;
        }
        else
        {
            int32_t const u = t - fc1;
            w.phase = 1;
            w.L_idx = u / params_.tiles_m2;
            w.M_idx = u - w.L_idx * params_.tiles_m2;
        }
        w.N_idx = 0;
        w.is_valid_tile = true;
        return w;
    }

    // Token-tile mode: the draws, and draw index -> the group's phase, entry, first weight panel and tile count. The
    // groups of an entry are its panels two by two (the last one alone if the panel count is odd), FC1 groups first.
    // The launch's tiles per draw, from the live entries (read after the kernel-wide wait), so that every CTA derives
    // the same numbering: groups when the live tiles give every CTA at least kGroupMinTilesPerCta of them, else
    // single tiles (with few tiles per CTA a group is a coarser unit of the FC1 phase's last wave, and the FC2 tiles
    // that wait for it lose more than the groups save).
    CUTLASS_DEVICE int32_t group_size()
    {
        if (group_size_ == 0)
        {
            int32_t const tiles = (params_.tiles_m1 + params_.tiles_m2) * live_entries();
            group_size_ = (kTilesPerDraw > 1 && tiles >= kGroupMinTilesPerCta * params_.grid) ? kTilesPerDraw : 1;
        }
        return group_size_;
    }

    CUTLASS_DEVICE int32_t groups_per_entry(int32_t tiles)
    {
        int32_t const g = group_size();
        return (tiles + g - 1) / g;
    }

    CUTLASS_DEVICE int32_t live_groups()
    {
        return (groups_per_entry(params_.tiles_m1) + groups_per_entry(params_.tiles_m2)) * live_entries();
    }

    CUTLASS_DEVICE WorkTileInfo decode_group(int32_t g, int32_t& count)
    {
        WorkTileInfo w{};
        int32_t const g1 = groups_per_entry(params_.tiles_m1);
        int32_t const fc1 = g1 * live_entries();
        int32_t tiles;
        int32_t j;
        if (g < fc1)
        {
            w.phase = 0;
            w.L_idx = g / g1;
            j = g - w.L_idx * g1;
            tiles = params_.tiles_m1;
        }
        else
        {
            int32_t const g2 = groups_per_entry(params_.tiles_m2);
            int32_t const u = g - fc1;
            w.phase = 1;
            w.L_idx = u / g2;
            j = u - w.L_idx * g2;
            tiles = params_.tiles_m2;
        }
        int32_t const gsz = group_size();
        w.M_idx = j * gsz;
        int32_t const left = tiles - w.M_idx;
        count = left < gsz ? left : gsz;
        w.N_idx = 0;
        w.is_valid_tile = true;
        return w;
    }

    // A valid tile no role works on: every CTA's first tile.
    CUTLASS_DEVICE static WorkTileInfo placeholder()
    {
        WorkTileInfo w{};
        w.is_valid_tile = true;
        w.live = false;
        return w;
    }

    // CTA b's first tile, the placeholder: this runs in the kernel's prologue, ahead of any griddepcontrol.wait, so it
    // reads nothing; it only derives the slot b / T1 whose facts `fso_resolve_first` loads after the wait.
    template <class ClusterShapeMNK>
    CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShapeMNK)
    {
        first_slot_ = static_cast<int32_t>(blockIdx.x) / params_.tiles_m1;
        return placeholder();
    }

    // Every thread, right after the kernel-wide griddepcontrol.wait (generator edit 4c): L, and the packed facts of
    // slot b / T1 (token-tile mode: list entry b / T1) loaded and discarded ("The first draw" in the header); the CTA
    // keeps the placeholder. Token-tile mode: then the first-panel prefetch from those two loads.
    CUTLASS_DEVICE void fso_resolve_first(WorkTileInfo& w)
    {
        int32_t const l_raw = *params_.live_slots;
        uint32_t const facts = first_slot_ < params_.num_slots ? fso_ld_facts(facts_words() + first_slot_) : ~0u;
        live_ = l_raw < 0 ? 0 : (l_raw > params_.num_slots ? params_.num_slots : l_raw);
        w = placeholder();
        fso_keep_facts(facts, facts_words());
        if constexpr (ListMode)
            fso_first_panel_prefetch(facts);
    }

    // Token-tile mode: the first-panel prefetch (see the header) of CTA b's guessed first tile, tile b: FC1 weight panel
    // b % T1 (128 rows of K1 bytes) of the expert in list entry b / T1 (`facts`, bits 14-23), and the panel's scale
    // block (128 rows of K1 / 32 UE8M0 bytes: K1 / 128 atoms of 512 B, the Sm1xx atom slab being block-major, k-atoms
    // within a block), each in 16 KB pieces. Thread 128 only: the first thread of the epilogue warps (warps 4-7 of
    // FsoSm100SlotGemmMoe's 256 threads), which have nothing to do before the first accumulator. Nothing is issued
    // with the switch off (a null weight pointer), for a -1 word, an entry at or past L, an expert at or past G, or a
    // panel at or past T1; so the pieces end at most at G * N1 * K1 bytes into w13 and G * N1 * K1 / 32 into sw13
    // (N1 = 128 * T1, which the launcher's shape rule makes exact). Compiled for the sm_100 / sm_103 device passes only
    // (the arch test the token-tile unit uses for its PDL switch): the kernel template is also instantiated in the
    // sm_90a and sm_120a passes, where it is never launched, and its code there stays as it was.
    CUTLASS_DEVICE void fso_first_panel_prefetch(uint32_t facts) const
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
        constexpr uint32_t kPiece = 16384;
        if (threadIdx.x != 4 * NumThreadsPerWarp || params_.first_panel_w == nullptr || facts == ~0u
            || first_slot_ >= live_)
            return;
        int32_t const expert = static_cast<int32_t>((facts >> 14) & 0x3ffu);
        int32_t const panel = static_cast<int32_t>(blockIdx.x) % params_.tiles_m1;
        if (expert >= params_.num_groups || panel >= params_.tiles_m1)
            return;
        // K1 < 65536 (the compact layout's rule), so a panel's 128 * K1 bytes fit 32 bits.
        uint32_t const w_bytes = 128u * static_cast<uint32_t>(params_.first_panel_k);
        uint32_t const sf_bytes = w_bytes / 32u;
        int64_t const tile = static_cast<int64_t>(expert) * params_.tiles_m1 + panel;
        char const* const w = params_.first_panel_w + tile * w_bytes;
        char const* const sf = params_.first_panel_sf + tile * sf_bytes;
        CUTLASS_PRAGMA_NO_UNROLL
        for (uint32_t off = 0; off < w_bytes; off += kPiece)
            fso_prefetch_l2_bulk(w + off, w_bytes - off < kPiece ? w_bytes - off : kPiece);
        CUTLASS_PRAGMA_NO_UNROLL
        for (uint32_t off = 0; off < sf_bytes; off += kPiece)
            fso_prefetch_l2_bulk(sf + off, sf_bytes - off < kPiece ? sf_bytes - off : kPiece);
#else
        (void) facts;
#endif
    }

    // The deferred draw (see the header), the scheduler warp's leader lane (lane 0 in token-tile mode, the elected lane in
    // slot mode). After publishing a live tile: an FC2 tile published not ready arms the hold with its done word and spin
    // target, any other live tile leaves it as it is (the queue is still behind the armed tile). Before the next draw:
    // hold while the launch is in the second-round regime, the counter has not passed the live tiles and the armed word
    // has not reached its target. Compiled for the sm_100 / sm_103 device passes only: the kernel template is also
    // instantiated in the sm_90a and sm_120a passes, where it is never launched, and its code there stays as it was.
    CUTLASS_DEVICE void fso_note_deferred_draw(WorkTileInfo const& w)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
        if (w.live && w.phase == 1 && !w.ready)
        {
            defer_word_ = done_counter(w.L_idx);
            defer_target_ = done_target(w);
        }
#else
        (void) w;
#endif
    }

    CUTLASS_DEVICE void fso_wait_deferred_draw()
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
        if (defer_word_ == nullptr)
            return;
        int const* const word = defer_word_;
        defer_word_ = nullptr;
        // The live tiles D and the FC1 tiles D1, from L (read after the kernel-wide wait). Counted in tiles in both
        // modes: in the regime the gate admits, the token-tile mode draws single tiles (tile pairs need at least
        // kGroupMinTilesPerCta tiles per CTA), so a draw is a tile and the counter counts tiles.
        int64_t const l = live_entries();
        int64_t const d = static_cast<int64_t>(params_.tiles_m1 + params_.tiles_m2) * l;
        int64_t const d1 = static_cast<int64_t>(params_.tiles_m1) * l;
        int64_t const g = params_.grid;
        if (d <= g || d - g > (d1 < g ? d1 : g))
            return;
        while (fso_ld_acquire_gpu(word) < defer_target_)
        {
            if (static_cast<int64_t>(fso_ld_relaxed_gpu(params_.counter)) >= d)
                break;
            __nanosleep(256);
        }
#endif
    }

    // The words the scheduler reads a drawn tile's facts from: slot mode the packed slot facts (`slot_info`), token-tile
    // mode the list entries (non-negative or -1, so `fso_keep_facts`' never-taken test holds for them too).
    CUTLASS_DEVICE int const* facts_words() const
    {
        if constexpr (ListMode)
            return params_.tile_list;
        else
            return slot_info();
    }

    // The routing op's packed facts of a slot (moe_glue.cu, compact_emit_slot): expert | rows << 16, -1 past the
    // active experts. One load gives the tile both, where today's slot scheduler loads slot_to_expert[slot] and then
    // masked_m[expert].
    CUTLASS_DEVICE int const* slot_info() const
    {
        return params_.counter + 2 + params_.num_groups;
    }

    CUTLASS_DEVICE void set_facts(WorkTileInfo& w, int32_t info) const
    {
        uint32_t const u = static_cast<uint32_t>(info);
        int32_t const e = static_cast<int32_t>(u & 0xffffu);
        int32_t const rows = static_cast<int32_t>(u >> 16);
        w.live = info != -1 && e < params_.num_groups && rows > 0;
        w.a_batch = w.live ? e : 0;
        w.rows = w.live ? rows : 0;
    }

    // Token-tile mode: the facts of list entry `info` (q | e << 14 | (r - 1) << 24, -1 past the live entries; the
    // packing of slot_work_stealing.cuh's tile-list mode), with the coordinate moved to the entry's block (L_idx) and
    // half (N_idx) and the store bound the half's first row plus the entry's rows. A -1 entry or an expert out of
    // range leaves the tile dead and its coordinate unchanged.
    CUTLASS_DEVICE void set_facts_list(WorkTileInfo& w, int32_t info) const
    {
        w.live = false;
        if (info < 0)
            return;
        int32_t const e = (info >> 14) & 0x3ff;
        if (e >= params_.num_groups)
            return;
        int32_t const q = info & 0x3fff;
        int32_t const h = q & 1;
        w.N_idx = h;
        w.L_idx = q >> 1;
        w.a_batch = e;
        w.rows = h * kTokenTileRows + ((info >> 24) & 0x3f) + 1;
        w.live = true;
    }

    CUTLASS_DEVICE auto work_tile_to_cta_coord(WorkTileInfo const& w) const
    {
        return cute::make_coord(w.M_idx, w.N_idx, cute::_, w.L_idx);
    }

    // The scheduler warp: one drawn tile per call, published through the CLC response pipeline (the protocol of
    // slot_work_stealing.cuh: a plain shared store of the 16-byte response, then the slot's full barrier arrival).
    CUTLASS_DEVICE
    PipelineState<Stages> advance_to_next_work(Pipeline& pipeline, PipelineState<Stages> state)
    {
        if constexpr (ListMode)
            return advance_list(pipeline, state);
        else
            return advance_slot(pipeline, state);
    }

    // Token-tile mode: one draw hands out a group of up to kTilesPerDraw tiles (see "Tile groups" in the header); the
    // first is published at once, the next ones at the following calls without touching memory. The whole warp runs
    // this warp-uniformly (every lane keeps the same group state): lane 0 makes the draw and the acquire, and the
    // warp's 32 lanes hold the token-tile list in registers (`list_regs_`, see "The list in registers" in the header),
    // so a draw's list lookup is a register select and a shuffle instead of a load.
    CUTLASS_DEVICE
    PipelineState<Stages> advance_list(Pipeline& pipeline, PipelineState<Stages> state)
    {
        uint32_t const lane = threadIdx.x % NumThreadsPerWarp;
        bool const leader = lane == 0;
        WorkTileInfo w = WorkTileInfo::invalid_work_tile();
        bool last_draw = false;
        if (pending_ > 0)
        {
            // The group's next tile: the next weight panel of the same entry, with the facts, the ready bit and
            // the target of the group's draw.
            w = group_;
            w.M_idx = group_.M_idx + (group_count_ - pending_);
            --pending_;
            w.release = (pending_ == 0 && w.phase == 0) ? group_count_ : 0;
        }
        else
        {
            int32_t old = 0;
            if (leader)
            {
                fso_wait_deferred_draw();
                old = atomicAdd(params_.counter, 1);
            }
            if (!list_loaded_)
            {
                // Once per CTA, right behind its first draw (both in flight together), after the kernel-wide wait.
                list_loaded_ = true;
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < kListRegs; ++k)
                {
                    int32_t const j = static_cast<int32_t>(lane) + k * static_cast<int32_t>(NumThreadsPerWarp);
                    list_regs_[k] = (list_in_regs() && j < params_.num_slots) ? params_.tile_list[j] : -1;
                }
            }
            old = __shfl_sync(0xffffffffu, old, 0);
            if (live_ < 0)
            {
                int32_t const l_raw = *params_.live_slots;
                live_ = l_raw < 0 ? 0 : (l_raw > params_.num_slots ? params_.num_slots : l_raw);
            }
            if (old < live_groups())
            {
                int32_t count = 1;
                w = decode_group(old, count);
                w.live = false;
                if (w.L_idx < params_.num_slots)
                {
                    set_facts_list(w, list_entry(w.L_idx, lane));
                    // The list entry names the block, so the acquire of the block's word follows it.
                    if (w.phase == 1 && w.live)
                    {
#if !defined(FSO_SLOT_MOE_NEGATIVE_CONTROL_NO_ACQUIRE)
                        uint32_t v = 0u;
                        if (leader)
                            v = fso_ld_acquire_gpu(params_.block_done + w.L_idx);
                        v = __shfl_sync(0xffffffffu, v, 0);
                        // The live halves from the word's high bits, written by the routing op (1 or 2 for every
                        // block a live entry names; clamped so that a word without them still waits for T1).
                        uint32_t const halves_raw = v >> kBlockHalvesShift;
                        uint32_t const halves = halves_raw < 1u ? 1u : (halves_raw > 2u ? 2u : halves_raw);
                        w.target = (halves_raw << kBlockHalvesShift) + halves * static_cast<uint32_t>(params_.tiles_m1);
                        w.ready = v >= w.target;
#else
                        // Negative-control build only (gate 1): the hand-off without its acquire, as in slot mode.
                        w.target = 0u;
                        w.ready = true;
#endif
                    }
                }
                group_ = w;
                group_count_ = count;
                pending_ = count - 1;
                w.release = (pending_ == 0 && w.phase == 0) ? count : 0;
            }
            else
                last_draw = true;
        }
        __syncwarp();
        while (!(pipeline.producer_try_acquire(state) == BarrierStatus::WaitDone))
        {
        }
        if (leader)
        {
            uint32_t const full = pipeline.producer_get_barrier(state);
            uint32_t const dst = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(static_cast<uint32_t>(w.M_idx)),
                "r"(static_cast<uint32_t>(w.L_idx)), "r"(pack_batch_rows(w)), "r"(pack_flags(w))
                : "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(full) : "memory");
            fso_note_deferred_draw(w);
            if (last_draw)
            {
                // As in slot mode: the CTA whose last draw completes the grid puts the draw and finish counters back to
                // zero; the block words are not touched (FC2 tiles may still be reading them).
                __threadfence();
                if (atomicAdd(params_.counter + 1, 1) == static_cast<int>(gridDim.x) - 1)
                {
                    __threadfence();
                    atomicExch(params_.counter, 0);
                    atomicExch(params_.counter + 1, 0);
                }
            }
        }
        __syncwarp();
        ++state;
        return state;
    }

    // Token-tile mode: whether the whole list fits the scheduler warp's registers, and list entry `i` (warp-uniform
    // `i`): a register select in every lane and a shuffle from lane i % 32, or, for a longer list, a load by every lane.
    CUTLASS_DEVICE bool list_in_regs() const
    {
        return params_.num_slots <= kListRegs * static_cast<int32_t>(NumThreadsPerWarp);
    }

    CUTLASS_DEVICE int32_t list_entry(int32_t i, uint32_t lane) const
    {
        (void) lane;
        if (!list_in_regs())
            return params_.tile_list[i];
        int32_t const k = i / static_cast<int32_t>(NumThreadsPerWarp);
        int32_t v = list_regs_[0];
        CUTLASS_PRAGMA_UNROLL
        for (int r = 1; r < kListRegs; ++r)
            v = k == r ? list_regs_[r] : v;
        return __shfl_sync(0xffffffffu, v, i % static_cast<int32_t>(NumThreadsPerWarp));
    }

    // Slot mode: one drawn tile per call. The deferred draw's state lives in the elected leader's registers from one call
    // to the next; were a later call to elect another lane, that call would only draw without the hold.
    CUTLASS_DEVICE
    PipelineState<Stages> advance_slot(Pipeline& pipeline, PipelineState<Stages> state)
    {
        bool const leader = cute::elect_one_sync();
        WorkTileInfo w = WorkTileInfo::invalid_work_tile();
        bool last_draw = false;
        if (leader)
        {
            // The live entry count: `fso_resolve_first` read it after the kernel-wide wait; were it not read yet, it
            // would go out together with the draw (the draw does not depend on it).
            int32_t l_raw = live_;
            if (l_raw < 0)
                l_raw = *params_.live_slots;
            fso_wait_deferred_draw();
            int32_t const old = atomicAdd(params_.counter, 1);
            if (live_ < 0)
                live_ = l_raw < 0 ? 0 : (l_raw > params_.num_slots ? params_.num_slots : l_raw);
            int32_t const t = old;
            if (t < live_tiles())
            {
                w = decode(t);
                // The tile's facts: one load of the slot's packed facts, and for an FC2 tile the acquire of the
                // entry's done counter right behind it (an acquire orders only the loads AFTER it, so the plain
                // load goes out first and both are in flight together).
                w.live = false;
                if (w.L_idx < params_.num_slots)
                {
                    int32_t const info = slot_info()[w.L_idx];
#if !defined(FSO_SLOT_MOE_NEGATIVE_CONTROL_NO_ACQUIRE)
                    if (w.phase == 1)
                        w.ready = fso_ld_acquire_gpu(params_.counter + 2 + w.L_idx)
                            >= static_cast<uint32_t>(params_.tiles_m1);
#else
                    // Negative-control build only (gate 1): the hand-off without its acquire. Every FC2 tile is
                    // published ready and its load warp never waits, so FC2 may read FC1 rows before they are
                    // written; the bit-equality tests must fail on this build.
                    if (w.phase == 1)
                        w.ready = true;
#endif
                    set_facts(w, info);
                }
            }
            else
                last_draw = true;
        }
        __syncwarp();
        while (!(pipeline.producer_try_acquire(state) == BarrierStatus::WaitDone))
        {
        }
        if (leader)
        {
            uint32_t const full = pipeline.producer_get_barrier(state);
            uint32_t const dst = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(static_cast<uint32_t>(w.M_idx)),
                "r"(static_cast<uint32_t>(w.L_idx)), "r"(pack_batch_rows(w)), "r"(pack_flags(w))
                : "memory");
            asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" ::"r"(full) : "memory");
            fso_note_deferred_draw(w);
            if (last_draw)
            {
                // As in slot_work_stealing.cuh: the CTA whose last draw completes the grid puts the draw and finish
                // counters back to zero. done[] is not touched (FC2 tiles may still be reading it).
                __threadfence();
                if (atomicAdd(params_.counter + 1, 1) == static_cast<int>(gridDim.x) - 1)
                {
                    __threadfence();
                    atomicExch(params_.counter, 0);
                    atomicExch(params_.counter + 1, 0);
                }
            }
        }
        __syncwarp();
        ++state;
        return state;
    }

    // Every warp role: take the next drawn tile from the pipeline.
    template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
    CUTLASS_DEVICE auto fetch_next_work(
        WorkTileInfo current, TileSchedulerPipeline& pipeline, TileSchedulerPipelineState state)
    {
        (void) current;
        pipeline.consumer_wait(state);
        uint32_t const src = cute::cast_smem_ptr_to_uint(&clc_response_ptr_[state.index()]);
        uint32_t m, l, br, v;
        asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(m), "=r"(l), "=r"(br), "=r"(v)
                     : "r"(src)
                     : "memory");
        cutlass::arch::fence_view_async_shared();
        pipeline.consumer_release(state);
        WorkTileInfo w{};
        w.M_idx = static_cast<int32_t>(m);
        w.N_idx = 0;
        w.L_idx = static_cast<int32_t>(l);
        w.is_valid_tile = (v & 1u) != 0u;
        w.live = (v & 2u) != 0u;
        w.phase = static_cast<int32_t>((v >> 2) & 1u);
        w.ready = (v & 8u) != 0u;
        w.a_batch = static_cast<int32_t>(br & 0xffffu);
        w.rows = static_cast<int32_t>(br >> 16);
        if constexpr (ListMode)
        {
            w.N_idx = static_cast<int32_t>((v >> 4) & 1u);
            w.release = static_cast<int32_t>((v >> 5) & 3u);
            w.target = v >> 8;
        }
        return cute::make_tuple(w, true);
    }

    //
    // K tile API (every tile covers the whole K extent of its phase's problem)
    //
    template <class ProblemShapeMNKL, class TileShape, class Shape>
    CUTLASS_DEVICE auto get_k_tile_iterator(WorkTileInfo const&, ProblemShapeMNKL problem_shape_MNKL,
        TileShape tile_shape, Shape)
    {
        auto k_tiles = cute::ceil_div(cute::get<2>(problem_shape_MNKL), cute::get<2>(tile_shape));
        return cute::make_coord_iterator(k_tiles);
    }

    template <class ProblemShape, class TileShape>
    CUTLASS_HOST_DEVICE static int get_work_k_tile_count(
        WorkTileInfo const&, ProblemShape problem_shape, TileShape tile_shape)
    {
        return cute::size(cute::ceil_div(cute::get<2>(problem_shape), cute::get<2>(tile_shape)));
    }

    CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(WorkTileInfo const&)
    {
        return 0u;
    }

    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&, Params const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE static bool compute_epilogue(WorkTileInfo const&)
    {
        return true;
    }

    CUTLASS_HOST_DEVICE static bool requires_fixup(Params const&, WorkTileInfo const)
    {
        return false;
    }

    template <bool IsComplex, class TiledMma, class AccEngine, class AccLayout, class AccumulatorPipeline,
        class AccumulatorPipelineState, class CopyOpT2R>
    CUTLASS_DEVICE AccumulatorPipelineState fixup(TiledMma const&, WorkTileInfo const&,
        cute::Tensor<AccEngine, AccLayout>&, AccumulatorPipeline, AccumulatorPipelineState acc_pipe_consumer_state,
        CopyOpT2R) const
    {
        return acc_pipe_consumer_state;
    }

    CUTLASS_DEVICE static bool continue_current_work(WorkTileInfo&)
    {
        return false;
    }

    // The value of the tile's done counter at which its entry's FC1 tiles have all released, and the counter itself
    // (the kernel's FC1 epilogue releases into it, the FC2 load warp spins on it). Slot mode: the entry's counter,
    // target T1. Token-tile mode: the word of the tile's block (`get<3>` of its coordinate is the block), target as
    // published with the tile.
    CUTLASS_DEVICE uint32_t done_target(WorkTileInfo const& w) const
    {
        if constexpr (ListMode)
            return w.target;
        else
            return static_cast<uint32_t>(params_.tiles_m1);
    }

    // How many FC1 completions the epilogue of this FC1 tile releases after its stores: slot mode always one, token-tile
    // mode the group's tile count on its last tile and none on the others.
    CUTLASS_DEVICE uint32_t fso_release_count(WorkTileInfo const& w) const
    {
        if constexpr (ListMode)
            return static_cast<uint32_t>(w.release);
        else
            return 1u;
    }

    CUTLASS_DEVICE int* done_counter(int32_t entry) const
    {
        if constexpr (ListMode)
            return params_.block_done + entry;
        else
            return params_.counter + 2 + entry;
    }

private:
    // The response's third word: the weights' batch index (bits 0-15) and the store-bound rows (bits 16-31); the
    // fourth: valid (bit 0), live (bit 1), phase (bit 2), ready (bit 3), and in token-tile mode the half (bit 4), the
    // FC1 release count (bits 5-6) and the spin target (bits 8-31; at most 2 << 20 | 2 * T1, i.e. 22 bits for
    // T1 < 2^19).
    CUTLASS_DEVICE static uint32_t pack_batch_rows(WorkTileInfo const& w)
    {
        uint32_t const rows = static_cast<uint32_t>(w.rows < 0 ? 0 : (w.rows > 0xffff ? 0xffff : w.rows));
        return (static_cast<uint32_t>(w.a_batch) & 0xffffu) | (rows << 16);
    }

    CUTLASS_DEVICE static uint32_t pack_flags(WorkTileInfo const& w)
    {
        if constexpr (ListMode)
            return (w.is_valid_tile ? 1u : 0u) | (w.live ? 2u : 0u) | ((w.phase & 1) ? 4u : 0u) | (w.ready ? 8u : 0u)
                | ((w.N_idx & 1) ? 16u : 0u) | ((static_cast<uint32_t>(w.release) & 3u) << 5) | (w.target << 8);
        else
            return (w.is_valid_tile ? 1u : 0u) | (w.live ? 2u : 0u) | ((w.phase & 1) ? 4u : 0u) | (w.ready ? 8u : 0u);
    }

    CLCResponse* clc_response_ptr_ = nullptr;
    Params const& params_;
    int32_t live_ = -1;
    int32_t first_slot_ = 0;
    // Token-tile mode, the scheduler warp (every lane the same): the current group's first tile, its tile count, how
    // many of its tiles are still to be published, and the token-tile list in registers (entry j in lane j % 32's
    // list_regs_[j / 32]).
    WorkTileInfo group_{};
    int32_t group_count_ = 0;
    int32_t pending_ = 0;
    bool list_loaded_ = false;
    int32_t list_regs_[ListMode ? 8 : 1] = {};
    // Token-tile mode: the launch's tiles per draw (0 until L is known).
    int32_t group_size_ = 0;
    // The deferred draw (the leader lane): the done word and spin target of the FC2 tile that armed the hold.
    int const* defer_word_ = nullptr;
    uint32_t defer_target_ = 0;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<FsoSlotMoeWorkStealingScheduler, arch::Sm100, TileShape, ClusterShape,
    SchedulerPipelineStageCount>
{
    using Scheduler = FsoSlotMoeWorkStealingTileScheduler<ClusterShape, SchedulerPipelineStageCount>;
};

template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<FsoSlotMoeWorkStealingListScheduler, arch::Sm100, TileShape, ClusterShape,
    SchedulerPipelineStageCount>
{
    using Scheduler = FsoSlotMoeWorkStealingTileScheduler<ClusterShape, SchedulerPipelineStageCount, true>;
};

} // namespace cutlass::gemm::kernel::detail
