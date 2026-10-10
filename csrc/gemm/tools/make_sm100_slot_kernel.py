#!/usr/bin/env python3
"""Generate the sm_100 slot-bound GEMM kernel from CUTLASS's stock dense kernel.

Origin: this generator is the prototype of runs b300_mxfp8_20260917/M-W1 and
b300_mxfp8_20260917/M-P1 (`make_slot_blockscaled.py`), vendored into the
library with fso naming and with the two prototype knobs that run
b300_mxfp8_20260917/M-P1 proved wrong — the "slot-spread" and
"tile-major" slot reorderings — removed. Nothing in `3rdparty/cutlass` is
modified: the generator reads CUTLASS's header and writes a separate,
independently named class into the build directory.

Which CUTLASS header, and why this one
--------------------------------------
`KernelTmaWarpSpecialized1SmMxf8f6f4Sm100` is a builder-facing schedule tag.
The block-scaled collective builder turns it into the dispatch policy
`MainloopSm100TmaUmmaWarpSpecializedBlockScaled`, whose `DispatchPolicy::Schedule`
is `KernelTmaWarpSpecializedBlockScaledSm100`. The kernel layer selects on that
tag, and the partial specialisation that accepts it lives in
`cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp`, whose enable_if is a
disjunction over `KernelTmaWarpSpecializedSm100` and
`KernelTmaWarpSpecializedBlockScaledSm100` — the hardware block-scaled DENSE
policy is served by the same kernel file as the plain dense policy, not by a
block-scaled-specific kernel file.

What the fork changes, and why a fork is needed at all
-----------------------------------------------------
fso's masked expert slab is ONE contiguous tensor per operand with a uniform
per-expert stride, so a single 3-D TMA descriptor per operand already addresses
every expert through the batch coordinate. CUTLASS's pointer-array (grouped)
kernel cannot assume that, so it rebuilds a descriptor per CTA per group. The
dense kernel never rebuilds, but it indexes batch b as expert b, which would
launch a tile for all G experts. The fork adds a slot -> expert remap of the
batch coordinate plus a whole-CTA early exit, and the caller sizes the grid by
the host-static bound S on the number of experts that can hold rows, so only
the experts the routing actually touched get tiles.

The edits, and what each one does:

  1  turn the partial specialisation of `GemmUniversal` into a standalone
     class, so fso can instantiate it explicitly without competing with the
     stock specialisation;
  2  add the slot list, the device row counts and the slot bound to the host
     `Arguments` (2) and to the device `Params` (2b);
  3  carry those four fields from `Arguments` into `Params` in
     `to_underlying_arguments`;
  4  remap the batch coordinate from slot to expert id at every assignment of
     `cta_coord_mnkl` — the four re-assignments inside the warp roles'
     persistent loops (4) and the initial one (4b). Every consumer of the
     coordinate (the mainloop's A / B / SFA / SFB global slices, the
     epilogue's D slice, the k-tile iterator) reads it after that point;
  5  give the tile scheduler its own batch extent L = S in
     `to_underlying_arguments` (5) and in `get_grid_shape` (5b), while the
     mainloop and epilogue keep L = G. The grid must be sized by the slot
     count, but the TMA descriptors must span all G experts because the
     coordinate the kernel finally uses is the expert id;

The per-tile skip (edits 7 and 8), and why there is no whole-CTA exit
--------------------------------------------------------------------
CUTLASS's `StaticPersistentScheduler` asks for its grid with
`truncate_by_problem_size = true`, and that path clamps the CTA count to the
SM count (`tile_scheduler_params.h`). The kernel is therefore persistent:
whenever the problem has more tiles than the device has SMs, the grid is the SM
count and each CTA loops over several tiles. Liveness must therefore be decided
per TILE, and for two separate reasons.

  * A CTA that survives its first tile still WALKS every other tile the
    scheduler assigns to it, including the tiles of slots that hold no expert,
    and issues their mainloop TMA loads. Those loads stream a dead expert's
    whole weight panel from HBM for nothing. On Family B gate_up at M = 8 this
    was measured (runs b300_mxfp8_20260917/M-V1 and .../M-X1) as a DRAM read of
    1.341 times the minimum weight bytes of the live experts, against 1.039
    once the skip is in.
  * A CTA whose first tile is dead would drop every live tile assigned to it
    afterwards. Under this route's own guard that cannot happen — the slot list
    is packed, so the dead slots are a suffix of the slot range and a CTA whose
    first tile is dead has only dead tiles after it — but it is what makes a
    whole-CTA exit unsound in general.

An earlier version of this generator carried such an exit, armed by a host flag
when the launched grid happened to equal the tile count. It was deleted in run
b300_mxfp8_20260917/M-I1 and the flags with it. Run b300_mxfp8_20260917/M-I5
had already measured it as never faster beyond the pass-to-pass spread and 2 to
7 per cent slower at M <= 4, because every CTA — live ones included — paid for a
throwaway tile-scheduler probe and a slot-list read that the per-tile test then
performed again; and the flag that switched the per-tile skip off could only
ever reproduce a kernel that is wrong whenever the grid is truncated, which is
not something a library should be able to be asked for.

  7  a device helper, `fso_slot_remap_and_live`, injected into
     `namespace cutlass::gemm::kernel`. It performs edit 4's slot -> expert
     remap AND returns whether the tile is live, so that both facts are derived
     from one read of the slot list. The liveness test is the N tile index
     against the expert's routed row count, because the routed token rows sit
     on the N axis (the operands are swapped; the expert weights are on M);
     comparing the M tile index would drop valid output, since an M tile of 128
     weight rows is at or above `masked_m` for every active expert in the
     decode band.

  8  the per-tile skip itself, at eight sites across the four warp roles that
     do tile work (mainloop load, mma, epilogue load, epilogue). Each role's
     persistent loop does the tile's work only when `fso_live` is set. The
     guard in the mainloop-load role is placed at the very top of the loop
     body, AHEAD of the k-tile iterator and of both `collective_mainloop.load`
     calls, so a dead tile issues no TMA of either operand or of either scale
     factor — which is the whole point: the cost being removed is the dead
     slots' weight stream, not the tensor-core work. All four roles evaluate
     the same pure function of the same `work_tile_info`, so they skip the same
     tiles and no pipeline state is advanced by anyone for a dead tile; the CTA
     exits only when its tile range is exhausted. The one statement lifted OUT
     of the guard is the mainloop-load role's `load_order_barrier.arrive()`: a
     leading run of dead tiles would otherwise leave the epilogue-load warp
     waiting on a barrier nobody ever arrives at.

Edit 4c, and why the first liveness test is not where the others are
-------------------------------------------------------------------
The slot list may be produced by a kernel several launches earlier — the layer
gets it from `moe_build_routing`, not from the one-block builder — and this
kernel is launched as a programmatic dependent, so its PROLOGUE is free to run
while earlier grids are still on the machine. The first liveness test reads
`slot_to_expert` and `masked_m`, and CUTLASS's own `griddepcontrol.wait` sits
further down, inside each warp role. Left where edit 4b puts the coordinate, the
read would be allowed to see an unwritten list, and the consequence is not a
crash but a dropped tile: a live expert's output rows are never written and the
layer returns a finite, wrong answer. Edit 4c therefore moves the first
evaluation below `pipeline_init_wait` and puts `wait_on_dependent_grids()` ahead
of it. The pipeline and TMEM initialisation above it still overlap the producing
grid; only the read is ordered.

Edit 8i, and why the per-tile skip must not wrap the scheduler handshake
------------------------------------------------------------------------
Under a DYNAMIC scheduler (CUTLASS's cluster-launch-control one, or the compact
route's work-stealing one) the mainloop-load warp releases the scheduler warp's
next query with one throttle handshake per tile, and the stock kernel issues it
after the k-tile set-up, which edit 8a puts inside the skip. A dead tile would
then skip it, the scheduler warp would wait for it forever and the kernel would
never finish (run fso_0.2.1_20261010/S10a). Edit 8i moves the handshake ahead of
the skip. Under the static scheduler the block is `if constexpr
(IsSchedDynamicPersistent)` = false, so the masked class compiles to the same
code.

Each edit asserts its exact match count, so a CUTLASS bump that moves any of
these sites fails the build loudly instead of producing a silently different
kernel.

The compact-layout twin (fso 0.2.1)
-----------------------------------
The same generated header also carries `FsoSm100SlotGemmCompact`, the kernel the
compact activation layout runs on the slot route. It is produced from the same
CUTLASS source by the same edits plus five more (c1-c5), so the masked class
above stays exactly what it was. On the compact layout the batch coordinate has
to mean two things: the weights (A / SFA in the swap orientation) are indexed
by EXPERT, while the tokens (B / SFB) and the output (D) are indexed by the
expert's first 128-row block blk0[expert] of the one-group compact buffer.

On this route blk0[expert] needs no load: it IS the slot index. The route runs
only while the row capacity fits one token tile (m_cap <= 64), so every active
expert holds at most 64 rows and owns exactly one 128-row block; blk0 is the
exclusive prefix of the per-expert block counts in ascending expert order, and
the slot list holds the active experts in ascending order, so the expert in
slot s starts at block s. The twin therefore keeps the scheduler's batch
coordinate (the slot) as the batch index of B, SFB, the epilogue's D slice and
the fused epilogue's l_coord, and moves only the weight operands onto the
expert id, by offsetting the two weight-side partitioned tensors of the
mainloop's load parameters (`tAgA_mkl`, `tAgSFA_mkl`) along their batch mode by
(expert - slot) at the two load call sites. That is the two-batch-coordinate
design of run b300_mxfp8_20260917/M-X2 (whose rank-5 coordinate made the
collective read the A / SFA batch from a fifth mode), expressed inside the
already-forked kernel so the collective is not forked. A first version read
blk0[expert] in every warp role on every tile and measured up to 2 per cent
slower than the masked kernel on the cells with the most and shortest tiles
(Family C down at M = 16); the identity removes that read.

  c1  the class name;
  c2  `fso_num_groups` in Arguments / Params and its transfer into Params (the
      expert-id range check; the batch extent L is no longer the group count);
  c3  every remap site calls `fso_slot_live_compact`, which leaves the batch
      mode on the slot and returns the expert id beside the liveness verdict
      (the per-tile skip on masked_m[expert] is unchanged);
  c4  the declaration of the expert-id local next to `fso_live`;
  c5  the two `collective_mainloop.load` calls receive the load parameters
      with the weight-side tensors shifted by (expert - slot);
  c6  the epilogue warps' store call receives the problem shape with its token
      extent N cut to the expert's routed rows (`fso_store_shape`, the row
      count read by the liveness helper that already loads it), so both compact
      epilogues -- the plain bf16 slot store and the exact-N fused SwiGLU store
      -- write only the expert's live token columns instead of all m_cap of
      them (run fso_0.2.1_20261010/S10b-1). The mainloop and the epilogue-load
      call keep the full shape;
  c7  the MMA warp and
  c8  the epilogue warps fetch the next tile after the current tile's work, not
      before it (a dynamic scheduler's answer then overlaps the tile instead of
      delaying its start);
  c9  the load warp releases one extra scheduler query at its start, so a
      dynamic scheduler draws two tiles ahead of the mainloop.

The compact twin is instantiated with the work-stealing tile scheduler of
`slot_work_stealing.cuh` (a per-call counter over the live tiles) rather than
the static persistent one; that needs no edit here beyond 8i, because the
scheduler is a template argument of the class.

Usage: make_sm100_slot_kernel.py <output-header>; CUTLASS_DIR (or
BSGEMM_CUTLASS_DIR) names the CUTLASS checkout to read.
"""
from __future__ import annotations

import os
import re
import sys

REL_SRC = "include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"


def _cutlass_dir() -> str:
    for var in ("CUTLASS_DIR", "BSGEMM_CUTLASS_DIR"):
        v = os.environ.get(var)
        if v:
            return v
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "3rdparty", "cutlass"))


def generate(cutlass_dir: str) -> tuple[str, list[str], str]:
    """Return (generated source, per-edit report lines, source header path)."""
    src_path = os.path.join(cutlass_dir, REL_SRC)
    with open(src_path) as f:
        s = f.read()
    report: list[str] = []

    def sub(pat: str, rep: str, n: int, what: str) -> None:
        nonlocal s
        s2, cnt = re.subn(pat, rep, s, flags=re.S)
        assert cnt == n, f"[{what}] pattern applied {cnt} times, expected {n}: {pat[:70]}"
        s = s2
        report.append(f"edit {what}: {cnt} site(s)")

    # ---- 1. standalone class ---------------------------------------------
    sub(r"class GemmUniversal<\s*ProblemShape_,\s*CollectiveMainloop_,\s*CollectiveEpilogue_,\s*"
        r"TileSchedulerTag_,\s*cute::enable_if_t<.*?>>\s*\{",
        "class FsoSm100SlotGemm {", 1, "1 standalone class")

    # ---- 2. extra host arguments and device params -----------------------
    sub(r"(struct Arguments \{\s*GemmUniversalMode mode\{\};\s*ProblemShape problem_shape\{\};\s*"
        r"MainloopArguments mainloop\{\};\s*EpilogueArguments epilogue\{\};\s*"
        r"KernelHardwareInfo hw_info\{\};\s*TileSchedulerArguments scheduler\{\};)",
        r"\1\n    // fso slot route: slot -> expert id (-1 marks an unused slot), the\n"
        "    // device-resident routed row count per expert, and the host-static slot\n"
        "    // upper bound the grid was sized for.\n"
        "    int const* fso_slot_to_expert{nullptr};\n"
        "    int const* fso_masked_m{nullptr};\n"
        "    int fso_num_slots{0};", 1, "2 Arguments fields")

    sub(r"(struct Params \{\s*GemmUniversalMode mode\{\};\s*ProblemShape problem_shape\{\};\s*"
        r"MainloopParams mainloop\{\};\s*EpilogueParams epilogue\{\};\s*"
        r"TileSchedulerParams scheduler\{\};\s*KernelHardwareInfo hw_info\{\}; )",
        r"\1\n    int const* fso_slot_to_expert{nullptr};\n"
        "    int const* fso_masked_m{nullptr};\n"
        "    int fso_num_slots{0};",
        1, "2b Params fields")

    # ---- 3. carry them into Params ---------------------------------------
    sub(r"(\)\s*\n\s*,args\.hw_info\s*\n\s*\};\s*\n\s*\}\s*\n\s*static bool\s*\n\s*can_implement)",
        r"\n      )\n      ,args.hw_info\n      ,args.fso_slot_to_expert\n      ,args.fso_masked_m\n"
        "      ,args.fso_num_slots\n"
        "    };\n  }\n\n  static bool\n  can_implement", 1, "3 to_underlying_arguments")

    # ---- 7. the remap-and-liveness helper --------------------------------
    # Injected as a free template in the kernel namespace rather than as a
    # member, so that the five assignment sites -- which live in four
    # differently nested warp-role loops -- can all call it by the same name
    # without a macro.
    helper = """
// fso slot route: remap the batch coordinate from a SLOT index to an expert id
// and report whether the tile the scheduler just handed this CTA is LIVE.
//
// A tile is live when its slot holds an active expert and when its token tile
// starts before that expert's routed row count. The routed token rows sit on
// the N axis (the operands are swapped: the expert weights are on M), so it is
// the N tile index that is compared against masked_m. Comparing the M tile
// index instead would silently drop valid output, because an M tile of 128
// weight rows is >= masked_m for every active expert in the decode band.
//
// The helper must be TOTAL over whatever coordinate it is handed, including a
// coordinate the loop is about to reject. Each warp role assigns
// cta_coord_mnkl once more after the scheduler has reported that there is no
// next tile, and the slot index of that coordinate is not constrained to the
// slot range. The slot builder initialises exactly num_slots entries of a
// fixed-capacity pool, so an unchecked index can read an entry it never wrote
// and return an expert id that is not an expert id; indexing masked_m with it
// is an out-of-bounds global read, which compute-sanitizer reports and which
// becomes a fatal cudaErrorIllegalAddress under a profiler. Both the slot and
// the expert are therefore range-checked, and a coordinate that fails either
// check is reported dead and left un-remapped -- which is what the loop that
// is about to exit expects anyway.
template <class Coord>
CUTLASS_DEVICE bool
fso_slot_remap_and_live(int const* slot_to_expert, int const* masked_m, int num_slots,
                        int num_groups, int tile_n_rows, Coord& cta_coord_mnkl) {
  if (slot_to_expert == nullptr) { return true; }
  int const fso_slot = cute::get<3>(cta_coord_mnkl);
  if (fso_slot < 0 || fso_slot >= num_slots) { return false; }
  int const fso_expert = slot_to_expert[fso_slot];
  if (fso_expert < 0 || fso_expert >= num_groups) { return false; }
  cta_coord_mnkl = cute::make_coord(cute::get<0>(cta_coord_mnkl), cute::get<1>(cta_coord_mnkl),
                                    cute::get<2>(cta_coord_mnkl), fso_expert);
  return int(cute::get<1>(cta_coord_mnkl)) * tile_n_rows < masked_m[fso_expert];
}
"""
    sub(r"(namespace cutlass::gemm::kernel \{\n)", lambda m, h=helper: m.group(1) + h,
        1, "7 remap-and-liveness helper")

    # ---- 4. batch-coordinate remap + liveness at every assignment ---------
    rest_anchor = "\n        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);"
    rest = """
        cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
        fso_live = fso_slot_remap_and_live(params.fso_slot_to_expert, params.fso_masked_m,
                                           params.fso_num_slots, int(L), cute::size<1>(CtaShape_MNK{}),
                                           cta_coord_mnkl);"""
    n = s.count(rest_anchor)
    assert n == 4, f"[4 batch remap] found {n} persistent-loop sites, expected 4"
    s = s.replace(rest_anchor, rest, 4)
    report.append(f"edit 4 batch remap + liveness (loop sites): {n} site(s)")

    first_anchor = "    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);\n"
    first = """    bool fso_live = true;
    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
"""
    n = s.count(first_anchor)
    assert n == 1, f"[4b batch remap] found {n} initial sites, expected 1"
    s = s.replace(first_anchor, first, 1)
    report.append("edit 4b batch remap (initial site, declaration): 1 site")

    # ---- 4c. the first liveness test, below a dependent-grid wait ---------
    # The slot list is written by a kernel that may still be running when this
    # one's prologue starts: the launch carries the programmatic
    # stream-serialisation attribute, and the producer (moe_build_routing, or
    # the route's own one-block builder) is one or more launches back. CUTLASS's
    # `griddepcontrol.wait` sits further down, inside each warp role, so the
    # very first read of `slot_to_expert` / `masked_m` has to carry its own.
    # Everything above it -- TMEM tensor init, pipeline init, the cluster
    # barrier -- still overlaps the producing grid; only the read is ordered.
    wait_anchor = "    pipeline_init_wait(cluster_size);\n"
    wait_rep = """    pipeline_init_wait(cluster_size);

    // fso slot route, edit 4c: order the FIRST read of the slot list and of the
    // routed row counts behind the grid that writes them. Reading them in the
    // prologue of a programmatic dependent launch, as the assignment above
    // would, is allowed to see an unwritten list; the observable consequence is
    // a live tile judged dead, i.e. an expert's output rows never written and a
    // finite, wrong layer output with no error anywhere.
    cutlass::arch::wait_on_dependent_grids();
    fso_live = fso_slot_remap_and_live(params.fso_slot_to_expert, params.fso_masked_m,
                                       params.fso_num_slots, int(L), cute::size<1>(CtaShape_MNK{}),
                                       cta_coord_mnkl);
"""
    n = s.count(wait_anchor)
    assert n == 1, f"[4c first liveness] found {n} pipeline_init_wait sites, expected 1"
    s = s.replace(wait_anchor, wait_rep, 1)
    report.append("edit 4c first liveness test after a dependent-grid wait: 1 site")

    # ---- 5. the tile scheduler gets its own L = fso_num_slots -------------
    # CUTLASS sizes the grid AND builds the A / B / D / SFA / SFB TMA
    # descriptors from the same problem shape. The slot-bound grid needs L = S
    # while the descriptors must span all G experts, because the coordinate the
    # kernel finally uses is the expert id. With L = S every expert id >= S
    # falls outside the descriptors' batch bound, the loads return zeros and
    # the output is NaN.
    sub(r"(      TileScheduler::to_underlying_arguments\(\n)(        problem_shape_MNKL,)",
        r"\1        cute::make_shape(cute::get<0>(problem_shape_MNKL), cute::get<1>(problem_shape_MNKL),\n"
        "                         cute::get<2>(problem_shape_MNKL),\n"
        "                         args.fso_num_slots > 0 ? args.fso_num_slots : cute::get<3>(problem_shape_MNKL)),",
        1, "5 scheduler L in to_underlying_arguments")

    sub(r"(  static dim3\n  get_grid_shape\(Params const& params\) \{.*?)"
        r"auto problem_shape_MNKL = append<4>\(params\.problem_shape, Int<1>\{\}\);",
        r"\1auto problem_shape_MNKL_full = append<4>(params.problem_shape, Int<1>{});\n"
        "    auto problem_shape_MNKL = cute::make_shape(cute::get<0>(problem_shape_MNKL_full),\n"
        "        cute::get<1>(problem_shape_MNKL_full), cute::get<2>(problem_shape_MNKL_full),\n"
        "        params.fso_num_slots > 0 ? params.fso_num_slots : cute::get<3>(problem_shape_MNKL_full));",
        1, "5b scheduler L in get_grid_shape")

    # ---- 8. the per-tile skip in the four warp roles' persistent loops ----
    # Eight literal-anchored sites. The mainloop-load guard opens ahead of the
    # k-tile iterator and of both collective_mainloop.load calls, so a dead tile
    # issues no TMA at all; 8b lifts the load-order arrival out of it so a
    # leading run of dead tiles cannot strand the epilogue-load warp on a
    # barrier nobody arrives at.
    def lit(anchor, rep, n, what):
        nonlocal s
        cnt = s.count(anchor)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}: {anchor[:70]!r}"
        s = s.replace(anchor, rep, n)
        report.append(f"edit {what}: {cnt} site(s)")

    lit("      do {\n"
        "        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.\n"
        "        auto k_tile_iter",
        "      do {\n"
        "       if (fso_live) {   // fso per-tile skip: a dead tile costs this CTA no mainloop traffic\n"
        "        // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.\n"
        "        auto k_tile_iter", 1, "8a main_load open")

    lit("        if (do_load_order_arrive) {\n"
        "          load_order_barrier.arrive();\n"
        "          do_load_order_arrive = false;\n"
        "        }\n\n", "", 1, "8b main_load lift the load-order arrival out of the guard")

    lit("        mainloop_pipe_producer_state = mainloop_producer_state_next_;\n\n"
        "        // Sync warp to prevent non-participating threads entering next wave early",
        "        mainloop_pipe_producer_state = mainloop_producer_state_next_;\n"
        "       }\n"
        "        // fso: the load-order arrival is unconditional so that a leading run of dead\n"
        "        // tiles cannot leave the epilogue-load warp waiting on a barrier nobody arrives at.\n"
        "        if (do_load_order_arrive) {\n"
        "          load_order_barrier.arrive();\n"
        "          do_load_order_arrive = false;\n"
        "        }\n\n"
        "        // Sync warp to prevent non-participating threads entering next wave early",
        1, "8c main_load close")

    lit("        if (is_mma_leader_cta) {\n"
        "          mainloop_pipe_consumer_state = collective_mainloop.mma(",
        "        if (fso_live) {   // fso per-tile skip\n"
        "        if (is_mma_leader_cta) {\n"
        "          mainloop_pipe_consumer_state = collective_mainloop.mma(", 1, "8d mma open")

    lit("        ++accumulator_pipe_producer_state;\n"
        "        work_tile_info = next_work_tile_info;",
        "        ++accumulator_pipe_producer_state;\n"
        "        }\n"
        "        work_tile_info = next_work_tile_info;", 1, "8e mma close")

    lit("        if (compute_epilogue) {",
        "        if (compute_epilogue && fso_live) {   // fso per-tile skip", 1, "8f epi_load guard")

    lit("        // Accumulator stage slice\n"
        "        int acc_stage = [&] () {\n"
        "          if constexpr (IsOverlappingAccum) {\n"
        "            return accumulator_pipe_consumer_state.phase();",
        "        if (fso_live) {   // fso per-tile skip\n"
        "        // Accumulator stage slice\n"
        "        int acc_stage = [&] () {\n"
        "          if constexpr (IsOverlappingAccum) {\n"
        "            return accumulator_pipe_consumer_state.phase();", 1, "8g epilogue open")

    lit("          do_tail_store = true;\n"
        "        }\n"
        "        work_tile_info = next_work_tile_info;",
        "          do_tail_store = true;\n"
        "        }\n"
        "        }\n"
        "        work_tile_info = next_work_tile_info;", 1, "8h epilogue close")

    # ---- 8i. the CLC throttle handshake ahead of the per-tile skip -------------
    # The mainloop-load warp paces the scheduler warp of a DYNAMIC scheduler with
    # one throttle handshake per tile (`clc_throttle_pipeline.producer_acquire` /
    # `producer_commit`), and the scheduler warp waits for it before every query.
    # Edit 8a opens the per-tile skip ahead of that handshake, so a dead tile would
    # skip it, leave the scheduler warp waiting forever and the kernel would never
    # finish (run fso_0.2.1_20261010/S10a, hoist in S10a/scripts/apply_variants_clc2.py).
    # The handshake therefore moves in front of the skip: every tile, live or dead,
    # releases the next query. Under the static scheduler the block is
    # `if constexpr (IsSchedDynamicPersistent)` = false, so the masked class compiles
    # to the same code.
    _pat = re.compile(
        r"(       if \(fso_live\) \{   // fso per-tile skip: a dead tile costs this CTA no mainloop traffic\n)"
        r"(.*?k_tile_prologue = min\(MainloopPipeline::Stages, k_tile_count\);\n\n)"
        r"(        if constexpr \(IsSchedDynamicPersistent\) \{\n          if \(is_first_cta_in_cluster && requires_clc_query\) \{\n.*?\n          \}\n        \}\n)",
        re.S)
    s, _cnt = _pat.subn(lambda m: m.group(3) + m.group(1) + m.group(2), s)
    assert _cnt == 1, f"[8i clc throttle hoist] applied {_cnt} times, expected 1"
    report.append("edit 8i CLC throttle handshake ahead of the per-tile skip: 1 site")

    return s, report, src_path


COMPACT_HELPERS = """
// fso slot route, compact layout (0.2.1): the liveness test of the masked helper
// above, without its remap. The batch mode stays on the SLOT, which on this route
// equals the expert's first 128-row block blk0[e] of the one-group compact buffer
// (every active expert owns exactly one block, and the packed slot list and blk0
// both run over the active experts in ascending order), i.e. the batch index the
// tokens, their scales and the output use. The expert id, which the weights use,
// is returned in `a_batch`, and the expert's routed row count in `rows` (the
// epilogue's store bound, see `fso_store_shape`). Liveness is the masked helper's,
// unchanged: the N tile against the expert's routed row count. Same totality
// rules: a coordinate whose slot or expert is out of range is dead.
template <class Coord>
CUTLASS_DEVICE bool
fso_slot_live_compact(int const* slot_to_expert, int const* masked_m, int num_slots, int num_groups,
                      int tile_n_rows, Coord const& cta_coord_mnkl, int& a_batch, int& rows) {
  if (slot_to_expert == nullptr) { return false; }
  int const fso_slot = cute::get<3>(cta_coord_mnkl);
  if (fso_slot < 0 || fso_slot >= num_slots) { return false; }
  int const fso_expert = slot_to_expert[fso_slot];
  if (fso_expert < 0 || fso_expert >= num_groups) { return false; }
  a_batch = fso_expert;
  rows = masked_m[fso_expert];
  return int(cute::get<1>(cta_coord_mnkl)) * tile_n_rows < rows;
}

// fso slot route, compact layout: the problem shape the EPILOGUE is handed, with
// the token extent N cut to the expert's routed rows. Both compact slot
// epilogues bound their stores by the N they are given (the direct store of the
// plain GEMM predicates every element on it, the fused SwiGLU store every token
// column), so the tile writes the expert's live rows only, instead of all m_cap
// columns of its token tile. The mainloop keeps the full problem shape: the
// token TMA still reads its 64-row box, and the dead columns' accumulators are
// computed and dropped.
template <class Shape4>
CUTLASS_DEVICE auto
fso_store_shape(Shape4 const& shape_mnkl, int rows) {
  int const n = int(cute::get<1>(shape_mnkl));
  return cute::make_shape(cute::get<0>(shape_mnkl), rows < n ? rows : n, cute::get<2>(shape_mnkl),
                          cute::get<3>(shape_mnkl));
}

// fso slot route, compact layout: the second batch coordinate. The mainloop
// slices all four global tensors with ONE batch index (the coordinate's mode 3,
// here the slot = blk0[expert]); the weight-side tensors (A = expert weights and
// SFA = their scales, in the swap orientation) are re-based along their batch mode
// by `delta` = expert - slot, so that the same slice lands on the expert.
// The tensors are TMA coordinate tensors, so the offset moves the batch
// coordinate the TMA is issued with and nothing else. The structured binding
// fails to compile if the collective's load parameters ever change shape.
template <class LoadInputs>
CUTLASS_DEVICE auto
fso_shift_weight_batch(LoadInputs const& load_inputs, int delta) {
  auto [k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB, tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,
        mcast_mask_a, mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb] = load_inputs;
  auto const off = cute::make_coord(cute::_0{}, cute::_0{}, cute::_0{}, delta);
  return cute::make_tuple(k_tiles, cute::domain_offset(off, tAgA_mkl), tBgB_nkl, tAsA, tBsB,
                          cute::domain_offset(off, tAgSFA_mkl), tBgSFB_nkl, tAsSFA, tBsSFB,
                          mcast_mask_a, mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb);
}
"""


def _compact_class(masked_text: str, report: list[str]) -> str:
    """Apply edits c1-c5 to the masked output and return the compact class."""
    s = masked_text

    def lit(anchor, rep, n, what):
        nonlocal s
        cnt = s.count(anchor)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}: {anchor[:70]!r}"
        s = s.replace(anchor, rep, n)
        report.append(f"edit {what}: {cnt} site(s)")

    lit("class FsoSm100SlotGemm {", "class FsoSm100SlotGemmCompact {", 1, "c1 class name")
    lit("    int const* fso_slot_to_expert{nullptr};\n"
        "    int const* fso_masked_m{nullptr};\n"
        "    int fso_num_slots{0};",
        "    int const* fso_slot_to_expert{nullptr};\n"
        "    int const* fso_masked_m{nullptr};\n"
        "    int fso_num_slots{0};\n"
        "    // compact layout: the expert count (the batch extent L is max(G, B_alloc))\n"
        "    int fso_num_groups{0};", 2, "c2 Arguments / Params fields")
    lit("      ,args.fso_num_slots\n",
        "      ,args.fso_num_slots\n      ,args.fso_num_groups\n", 1,
        "c2b to_underlying_arguments")
    remap_old = ("fso_live = fso_slot_remap_and_live(params.fso_slot_to_expert, params.fso_masked_m,")
    remap_new = ("fso_live = fso_slot_live_compact(params.fso_slot_to_expert, params.fso_masked_m,")
    lit(remap_old, remap_new, 5, "c3a liveness sites")
    lit("params.fso_num_slots, int(L), cute::size<1>(CtaShape_MNK{}),\n",
        "params.fso_num_slots, params.fso_num_groups, cute::size<1>(CtaShape_MNK{}),\n", 5,
        "c3b expert-id range bound")
    s2, cnt = re.subn(r"(fso_slot_live_compact\(.*?cta_coord_mnkl)\);", r"\1, fso_a_batch, fso_rows);", s,
                      flags=re.S)
    assert cnt == 5, f"[c3c expert out-parameter] applied {cnt} times, expected 5"
    s = s2
    report.append(f"edit c3c expert and row-count out-parameters: {cnt} site(s)")
    lit("    bool fso_live = true;\n",
        "    bool fso_live = true;\n    int fso_a_batch = 0;   // compact layout: the weights' batch index\n"
        "    int fso_rows = 0;      // compact layout: the expert's routed rows (the epilogue's store bound)\n", 1,
        "c4 expert-id and row-count locals")
    lit("          load_inputs,\n          cta_coord_mnkl,\n",
        "          fso_shift_weight_batch(load_inputs, fso_a_batch - int(cute::get<3>(cta_coord_mnkl))),\n"
        "          cta_coord_mnkl,\n", 2, "c5 weight-side batch offset at the load calls")
    # c6: the epilogue store sees N = the expert's routed rows (fso_store_shape above). Only the epilogue warps'
    # store call: the epilogue-load call (no accumulator argument) and the mainloop keep the full shape.
    lit("            accumulator_pipe_consumer_state,\n"
        "            problem_shape_MNKL,\n"
        "            CtaShape_MNK{},\n"
        "            cta_coord_mnkl,\n"
        "            TileShape{},\n"
        "            TiledMma{},\n"
        "            accumulator,\n",
        "            accumulator_pipe_consumer_state,\n"
        "            fso_store_shape(problem_shape_MNKL, fso_rows),   // fso: stores bounded by the live rows\n"
        "            CtaShape_MNK{},\n"
        "            cta_coord_mnkl,\n"
        "            TileShape{},\n"
        "            TiledMma{},\n"
        "            accumulator,\n", 1, "c6 epilogue store bounded by the expert's routed rows")

    # c7 / c8: the MMA warp and the epilogue warps take the NEXT tile after the current one's work instead of
    # before it. Under a static scheduler the next tile is arithmetic, but under the work-stealing scheduler it is
    # a global atomic that the scheduler warp issues while the tile starts; asked for before the current tile's
    # MMA (or epilogue), its round trip sat in front of every tile's compute, the whole kernel's worth at M = 1
    # where a CTA has one tile. Asked for after, it overlaps the tile.
    fetch = ("        // Fetch next work tile\n"
             "        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(\n"
             "          work_tile_info,\n"
             "          clc_pipeline,\n"
             "          clc_pipe_consumer_state\n"
             "        );\n\n"
             "        if (increment_pipe) {\n"
             "          ++clc_pipe_consumer_state;\n"
             "        }\n\n")
    fetch_after = ("        // fso: the next tile is fetched AFTER this tile's work (see c7 / c8 in the generator)\n"
                   "        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(\n"
                   "          work_tile_info,\n"
                   "          clc_pipeline,\n"
                   "          clc_pipe_consumer_state\n"
                   "        );\n"
                   "        if (increment_pipe) {\n"
                   "          ++clc_pipe_consumer_state;\n"
                   "        }\n")
    lit("        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, "
        "CtaShape_MNK{});\n\n" + fetch,
        "        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, "
        "CtaShape_MNK{});\n\n", 1, "c7a MMA warp: no fetch ahead of the tile")
    lit("        ++accumulator_pipe_producer_state;\n"
        "        }\n"
        "        work_tile_info = next_work_tile_info;",
        "        ++accumulator_pipe_producer_state;\n"
        "        }\n" + fetch_after +
        "        work_tile_info = next_work_tile_info;", 1, "c7b MMA warp: fetch after the tile")
    lit("      do {\n" + fetch + "        if (fso_live) {   // fso per-tile skip\n"
        "        // Accumulator stage slice\n",
        "      do {\n        if (fso_live) {   // fso per-tile skip\n"
        "        // Accumulator stage slice\n", 1, "c8a epilogue warps: no fetch ahead of the tile")
    lit("          do_tail_store = true;\n"
        "        }\n"
        "        }\n"
        "        work_tile_info = next_work_tile_info;",
        "          do_tail_store = true;\n"
        "        }\n"
        "        }\n" + fetch_after +
        "        work_tile_info = next_work_tile_info;", 1, "c8b epilogue warps: fetch after the tile")

    # c9: the scheduler warp may draw one tile further ahead. The load warp's per-tile throttle handshake (8i)
    # releases one query per tile it starts; one extra release at the start lets the draw for tile k + 2 go out
    # while tile k loads, so the atomic's round trip under a saturated memory system is not on the load warp's
    # path from one tile to the next.
    lit("      bool do_load_order_arrive = is_epi_load_needed;\n"
        "      bool requires_clc_query = true;\n\n",
        "      bool do_load_order_arrive = is_epi_load_needed;\n"
        "      bool requires_clc_query = true;\n\n"
        "      // fso (c9): one extra throttle release, i.e. a scheduler lookahead of two tiles\n"
        "      if constexpr (IsSchedDynamicPersistent) {\n"
        "        if (is_first_cta_in_cluster) {\n"
        "          clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);\n"
        "          clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);\n"
        "          ++clc_pipe_throttle_producer_state;\n"
        "        }\n"
        "      }\n\n", 1, "c9 one extra scheduler-throttle release at the load warp's start")

    head = "class FsoSm100SlotGemmCompact {"
    start = s.index(head)
    # include the template header that precedes the class
    tmpl = s.rindex("template <", 0, start)
    i = start + len(head)
    depth = 1
    while depth:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    assert s[i] == ";", "[c extract] class body does not end in '};'"
    return s[tmpl:i + 1]


def generate_with_compact(cutlass_dir: str) -> tuple[str, list[str], str]:
    text, report, src_path = generate(cutlass_dir)
    compact_report: list[str] = []
    cls = _compact_class(text, compact_report)
    close = "} // namespace cutlass::gemm::kernel"
    assert text.count(close) == 1, "[c insert] namespace close not unique"
    text = text.replace(close, COMPACT_HELPERS + "\n" + cls + "\n\n" + close)
    report.extend(compact_report)
    return text, report, src_path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        print("usage: make_sm100_slot_kernel.py <output-header>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    cutlass_dir = _cutlass_dir()
    text, report, src_path = generate_with_compact(cutlass_dir)
    # Only rewrite when the content actually changes, so an incremental build
    # does not recompile the translation unit on every invocation.
    old = None
    if os.path.exists(dst):
        with open(dst) as f:
            old = f.read()
    if old != text:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w") as f:
            f.write(text)
        state = "wrote"
    else:
        state = "unchanged"
    print("[fso] sm_100 slot kernel: " + "; ".join(report))
    print(f"[fso] sm_100 slot kernel: {state} {dst} ({len(text)} bytes) from {src_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
