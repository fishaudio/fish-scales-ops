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
      (the per-tile skip on masked_m[expert] is unchanged); edit c11 below then
      replaces every one of these calls by the facts the scheduler resolved, so
      the helper itself is no longer emitted;
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
      dynamic scheduler draws two tiles ahead of the mainloop;
  (c10 is not used: issue #13's lever 1, the programmatic-launch trigger
      right after the kernel-wide wait instead of after the MMA loop, was
      measured and did not shorten the FC1 -> FC2 gap; run
      fso_issue13_20261031/I13-L123)
  c11 the tile facts come from the scheduler (issue #13 lever 2): at edit 4c
      every thread asks the scheduler to resolve its first tile
      (`fso_resolve_first`, which also resolves the static walk's second tile
      when the launch never draws), and at the four loop sites every role reads
      the expert, the store-bound rows and the liveness that the scheduler
      published with the tile (`fso_tile_live`) instead of loading
      slot_to_expert[slot] and masked_m[expert] itself. The two dependent global
      loads per tile per role that this removes sat between the end of one tile
      and the start of the next in the load warp and in the MMA warp. In
      tile-list mode the scheduler also rewrites the coordinate to the entry's
      block and half, so the tile-list class needs no liveness code of its own;
  c12 the load warp tells the mainloop the tile's live token rows before its
      two load calls (issue #13 lever 3, `fso_set_load_rows`). The stock
      CUTLASS mainloop has no use for them and the call compiles to nothing;
      the row-bounded mainloop of slot_rows_mainloop.cuh, which the separate
      translation unit mxfp8_sm100_slot_rows_kernel.cu instantiates for the
      launches that read rows past the expert's (tile-list mode, and slot mode
      at m_cap >= 32), loads the token operand as one box of 8, 16 or 32 rows
      covering them.

The tile-list twin (fso 0.2.1, run fso_0.2.1_20261010/S10b-3b)
---------------------------------------------------------------
Past a row capacity of one token tile (m_cap > 64) an expert spans several 64-row
token tiles and several 128-row blocks, so the slot is no longer its block. The
header therefore carries a third class, `FsoSm100SlotGemmCompactList`, made from
the compact class by four more edits (l1-l4), so the compact class stays exactly
what it was: a first version that switched modes at run time inside the compact
class measured 0.4 to 1.9 per cent slower than f6ad71a at every M from 1 to 64 of
Family C (one layer pass; the most at M = 1, where its tiles are shortest). Its scheduler's batch coordinate
indexes the routing op's packed list of LIVE token tiles (one N tile per entry),
each entry names its block b, its half h and its expert e, and the scheduler's
tile-list mode (`FsoSlotWorkStealingListScheduler`, since edit c11; before it
the class's own `fso_slot_live_compact_list`) rewrites the coordinate to
(m, h, k, b) and hands out e for the weights and 64h plus the entry's rows as
the store bound. The launcher gives
the problem a token extent of one whole block (128 rows) per batch, so the half is
an ordinary second N tile, and CUTLASS's own TileN-64 scale handling in the
collective -- N tile n reads scale atom n / 2, and the MMA shifts the TMEM scale
address by n % 2 -- gives the half-block tile the second half of its block's
128-row scale atom. On the compact layout, where one block is exactly one atom, the
forced slot route is bit-identical to the pointer-array route at every m_cap from
65 to 512 and under every routing tested (test_mxfp8_compact_sm100.py, case
`slotx`); the masked class keeps its m_cap <= 64 guard.

  l1  the class name;
  l4  a check that no per-class liveness decode is left (since edit c11 the
      class differs from the compact one by its name and by the scheduler tag
      the launcher instantiates it with; its former edits l2 / l3, the tile-list
      pointer in Params and the class's own decode, moved into the scheduler).

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
// fso slot route, compact layout: the facts about a tile that every warp role needs besides its coordinate, as the
// work-stealing scheduler resolved them when it handed the tile out (slot_work_stealing.cuh, "What a handed-out tile
// carries"): the weights' batch index (the expert id) in `a_batch`, the expert's rows that the tile stores in `rows`
// (the epilogue's store bound, see `fso_store_shape`), and the verdict. A dead or invalid tile (a slot past the
// active experts, a token tile past the expert's rows, a -1 list entry, the coordinate a loop is about to reject)
// reports false and its other fields are not used. In tile-list mode the scheduler has already moved the coordinate
// to the entry's block and half, so `work_tile_to_cta_coord` gives the coordinate the operands are sliced with.
template <class WorkTileInfo>
CUTLASS_DEVICE bool
fso_tile_live(WorkTileInfo const& w, int& a_batch, int& rows) {
  a_batch = w.a_batch;
  rows = w.rows;
  return w.is_valid_tile && w.live;
}

// fso slot route, compact layout: hand the mainloop the tile's live token rows (issue #13 lever 3) -- the expert's
// rows inside the tile, its store bound minus the N tile's first row. Only a mainloop that declares
// `kFsoRowsBounded` (FsoRowsBoundedMainloop, slot_rows_mainloop.cuh) takes them; for CUTLASS's stock collective the
// call compiles to nothing, so the kernels of the stock instantiation keep their code.
template <class T, class = void>
struct fso_is_rows_bounded : cute::false_type {};
template <class T>
struct fso_is_rows_bounded<T, cute::void_t<decltype(T::kFsoRowsBounded)>> : cute::true_type {};

template <class Mainloop, class Coord, class CtaShape>
CUTLASS_DEVICE void
fso_set_load_rows(Mainloop& mainloop, int rows, Coord const& cta_coord_mnkl, CtaShape) {
  if constexpr (fso_is_rows_bounded<Mainloop>::value) {
    int const tn = int(cute::size<1>(CtaShape{}));
    int const live = rows - int(cute::get<1>(cta_coord_mnkl)) * tn;
    mainloop.fso_set_tile_rows(live < 1 ? 1 : (live < tn ? live : tn));
  }
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

    # c11 (issue #13 lever 2): the tile facts come from the scheduler. At edit 4c's site every thread resolves its
    # first tile through the scheduler (after the kernel-wide wait); at the four loop sites every
    # role reads what the scheduler published with the tile.
    s2, cnt = re.subn(r"(    cutlass::arch::wait_on_dependent_grids\(\);\n)    fso_live = fso_slot_live_compact\(.*?fso_rows\);",
                      r"\1    scheduler.fso_resolve_first(work_tile_info);\n"
                      r"    cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);\n"
                      r"    fso_live = fso_tile_live(work_tile_info, fso_a_batch, fso_rows);", s, flags=re.S)
    assert cnt == 1, f"[c11a first tile resolved by the scheduler] applied {cnt} times, expected 1"
    s = s2
    report.append(f"edit c11a first tile resolved by the scheduler: {cnt} site(s)")
    s2, cnt = re.subn(r"fso_live = fso_slot_live_compact\(.*?fso_rows\);",
                      "fso_live = fso_tile_live(work_tile_info, fso_a_batch, fso_rows);", s, flags=re.S)
    assert cnt == 4, f"[c11b loop sites read the published tile facts] applied {cnt} times, expected 4"
    s = s2
    report.append(f"edit c11b loop sites read the published tile facts: {cnt} site(s)")
    assert "fso_slot_live_compact(" not in s, "[c11c] a per-role liveness load survived in the compact class"
    report.append("edit c11c no per-role liveness load left: checked")

    # c12 (issue #13 lever 3): the load warp hands the mainloop the tile's live rows before its two load calls.
    lit("        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, "
        "load_inputs.k_tiles);\n",
        "        fso_set_load_rows(collective_mainloop, fso_rows, cta_coord_mnkl, CtaShape_MNK{});\n"
        "        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, "
        "load_inputs.k_tiles);\n", 1, "c12 the load warp hands the mainloop the tile's live rows")

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


def _compact_list_class(compact_cls: str, report: list[str]) -> str:
    """Apply edits l1-l4 to the compact class and return its tile-list twin."""
    s = compact_cls

    def lit(anchor, rep, n, what):
        nonlocal s
        cnt = s.count(anchor)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}: {anchor[:70]!r}"
        s = s.replace(anchor, rep, n)
        report.append(f"edit {what}: {cnt} site(s)")

    lit("class FsoSm100SlotGemmCompact {", "class FsoSm100SlotGemmCompactList {", 1, "l1 class name")
    # l2 / l3 (the tile-list pointer and the class's own decode) moved into the scheduler with edit c11.
    assert "fso_slot_live_compact" not in s, "[l4] a per-class liveness decode survived in the tile-list class"
    report.append("edit l4 no per-class liveness decode left: checked")
    return s


def generate_with_compact(cutlass_dir: str) -> tuple[str, list[str], str]:
    text, report, src_path = generate(cutlass_dir)
    compact_report: list[str] = []
    cls = _compact_class(text, compact_report)
    cls_list = _compact_list_class(cls, compact_report)
    close = "} // namespace cutlass::gemm::kernel"
    assert text.count(close) == 1, "[c insert] namespace close not unique"
    text = text.replace(close, COMPACT_HELPERS + "\n" + cls + "\n\n" + cls_list + "\n\n" + close)
    report.extend(compact_report)
    return text, report, src_path


MOE_HELPERS = """
// fso fused FC1 + FC2 slot kernel (issue #13 lever 5): the tile facts of `fso_tile_live` plus the tile's phase
// (0 = an FC1 tile, 1 = an FC2 tile) and, for a drawn FC2 tile, whether the scheduler found its entry's FC1 tiles
// all released when it drew it (slot_moe_scheduler.cuh).
template <class WorkTileInfo>
CUTLASS_DEVICE bool
fso_tile_live_moe(WorkTileInfo const& w, int& a_batch, int& rows, int& phase, bool& ready) {
  a_batch = w.a_batch;
  rows = w.rows;
  phase = w.phase;
  ready = w.ready;
  return w.is_valid_tile && w.live;
}
"""


def _moe_class(compact_cls: str, report: list[str]) -> str:
    """Apply edits m1-m12 to the compact class and return the fused FC1 + FC2 class `FsoSm100SlotGemmMoe`.

    The class serves both modes of the compact slot route, like the compact class does with its tile-list twin: the
    scheduler tag the launcher instantiates it with (slot_moe_scheduler.cuh) selects slot mode or token-tile mode, and
    in token-tile mode the scheduler has already moved each tile's coordinate to its block and half, so the class's
    per-entry counter index `get<3>(cta_coord_mnkl)` is the block (issue #13 lever 5 step 3, run
    fso_issue13_20261031/I13-L5b). The two modes' kernels are instantiated in separate translation units.

    The compact class is left exactly as it is: the fused class is emitted into its own generated header, which only
    the fused kernel's translation unit (ops/mxfp8_sm100_slot_moe_kernel.cu) includes (run
    fso_issue12_20261028/R1). What the edits do (issue #13 lever 5, design note I13-D45 section 4):

      m1  the class name and a fifth template parameter, the FC2 tiles' epilogue (`CollectiveEpilogue2_`);
      m2  its types, and the asserts that the two epilogues partition the accumulator the same way;
      m3  one shared-memory region for the two epilogues (they run one after the other on the same threads);
      m4  the FC2 problem shape, mainloop arguments and epilogue arguments in Arguments and Params;
      m5  their transfer into Params, and m6 their can_implement;
      m7  a second mainloop object, a second epilogue object, the FC2 problem shape and its load partitioning;
      m8  every tile-fact site also reads the tile's phase and ready bit (`fso_tile_live_moe`);
      m9  the load warp: both phases load through one inlined copy of `fso_load_after_entry` (slot_moe_mainloop.cuh),
          on the phase's mainloop object and partitioning: an FC2 tile with the hybrid hand-off, an FC1 tile with the
          ready bit forced on; the spin target is the scheduler's for the tile (`done_target(work_tile_info)`: T1 in
          slot mode, the tile's block target in token-tile mode);
      m10 the MMA warp: the k-tile count of the tile's phase;
      m11 the epilogue warps: an FC1 tile stores through the fused SwiGLU epilogue and then, after a named barrier
          over the 128 epilogue threads, one thread releases the scheduler's count for the tile on the entry's done
          counter (`red.release.gpu`, the producer half of the hand-off: one per tile in slot mode, a tile group's
          count on its last tile in token-tile mode); an FC2 tile stores through the plain epilogue;
      m12 the TMA descriptors of the second mainloop are prefetched too.
    """
    s = compact_cls

    def lit(anchor, rep, n, what):
        nonlocal s
        cnt = s.count(anchor)
        assert cnt == n, f"[{what}] literal found {cnt} times, expected {n}: {anchor[:70]!r}"
        s = s.replace(anchor, rep, n)
        report.append(f"edit {what}: {cnt} site(s)")

    lit("  class TileSchedulerTag_\n>\nclass FsoSm100SlotGemmCompact {",
        "  class TileSchedulerTag_,\n  class CollectiveEpilogue2_\n>\nclass FsoSm100SlotGemmMoe {", 1,
        "m1 class name and the FC2 epilogue parameter")

    lit("  using EpilogueParams = typename CollectiveEpilogue::Params;\n",
        "  using EpilogueParams = typename CollectiveEpilogue::Params;\n"
        "  // fso moe: the FC2 tiles' epilogue. The two epilogues must partition the accumulator identically (the\n"
        "  // same TMEM load atom, epilogue tile and thread count) and keep nothing in a load or store pipeline.\n"
        "  using CollectiveEpilogue2 = CollectiveEpilogue2_;\n"
        "  using EpilogueArguments2 = typename CollectiveEpilogue2::Arguments;\n"
        "  using EpilogueParams2 = typename CollectiveEpilogue2::Params;\n"
        "  static_assert(cute::is_same_v<typename CollectiveEpilogue::CopyOpT2R, typename CollectiveEpilogue2::CopyOpT2R>,\n"
        "    \"fso moe: the FC1 and FC2 epilogues must load the accumulator with the same TMEM atom\");\n"
        "  static_assert(cute::is_same_v<typename CollectiveEpilogue::EpilogueTile, typename CollectiveEpilogue2::EpilogueTile>,\n"
        "    \"fso moe: the FC1 and FC2 epilogues must walk the same epilogue tile\");\n"
        "  static_assert(CollectiveEpilogue::ThreadCount == CollectiveEpilogue2::ThreadCount,\n"
        "    \"fso moe: the FC1 and FC2 epilogues must run on the same threads\");\n"
        "  static_assert(cute::is_same_v<typename CollectiveEpilogue::LoadPipeline, typename CollectiveEpilogue2::LoadPipeline>\n"
        "      && cute::is_same_v<typename CollectiveEpilogue::StorePipeline, typename CollectiveEpilogue2::StorePipeline>,\n"
        "    \"fso moe: the FC1 and FC2 epilogues must share the epilogue load and store pipeline types\");\n",
        1, "m2 the FC2 epilogue's types")

    lit("      EpilogueTensorStorage epilogue;\n      MainloopTensorStorage mainloop;\n",
        "      // fso moe: the FC1 and the FC2 epilogue run one after the other on the same 128 threads, each behind the\n"
        "      // barrier at the top of its own store, so they share one region.\n"
        "      union EpilogueUnion {\n"
        "        EpilogueTensorStorage epilogue1;\n"
        "        typename CollectiveEpilogue2::TensorStorage epilogue2;\n"
        "      } epilogue;\n"
        "      MainloopTensorStorage mainloop;\n", 1, "m3 one region for the two epilogues")

    anchor = ("    // compact layout: the expert count (the batch extent L is max(G, B_alloc))\n"
              "    int fso_num_groups{0};\n")
    parts = s.split(anchor)
    assert len(parts) == 3, f"[m4 Arguments / Params fields] anchor found {len(parts) - 1} times, expected 2"
    add_args = ("    // fso moe: the FC2 problem (N_w2, m_cap, K2, L) and its mainloop and epilogue arguments.\n"
                "    ProblemShape problem_shape2{};\n"
                "    MainloopArguments mainloop2{};\n"
                "    EpilogueArguments2 epilogue2{};\n")
    add_params = ("    // fso moe: the FC2 problem and its mainloop and epilogue parameters.\n"
                  "    ProblemShape problem_shape2{};\n"
                  "    MainloopParams mainloop2{};\n"
                  "    EpilogueParams2 epilogue2{};\n")
    s = parts[0] + anchor + add_args + parts[1] + anchor + add_params + parts[2]
    report.append("edit m4 Arguments / Params fields: 2 site(s)")

    lit("      ,args.fso_num_slots\n      ,args.fso_num_groups\n    };",
        "      ,args.fso_num_slots\n      ,args.fso_num_groups\n"
        "      ,args.problem_shape2\n"
        "      ,CollectiveMainloop::to_underlying_arguments(args.problem_shape2, args.mainloop2, nullptr, args.hw_info)\n"
        "      ,CollectiveEpilogue2::to_underlying_arguments(args.problem_shape2, args.epilogue2, nullptr)\n"
        "    };", 1, "m5 to_underlying_arguments")

    lit("    implementable &= TileScheduler::can_implement(args.scheduler);\n",
        "    implementable &= TileScheduler::can_implement(args.scheduler);\n"
        "    implementable &= CollectiveMainloop::can_implement(args.problem_shape2, args.mainloop2);\n"
        "    implementable &= CollectiveEpilogue2::can_implement(args.problem_shape2, args.epilogue2);\n"
        "    // fso moe: the FC2 epilogue gets no workspace of its own.\n"
        "    implementable &= CollectiveEpilogue2::get_workspace_size(args.problem_shape2, args.epilogue2) == 0;\n",
        1, "m6 can_implement")

    lit("    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});\n    auto [M,N,K,L] = problem_shape_MNKL;\n",
        "    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});\n    auto [M,N,K,L] = problem_shape_MNKL;\n"
        "    // fso moe: the FC2 problem.\n"
        "    auto problem_shape2_MNKL = append<4>(params.problem_shape2, Int<1>{});\n", 1, "m7a the FC2 problem shape")
    lit("    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);\n"
        "    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);\n",
        "    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);\n"
        "    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue.epilogue1);\n"
        "    // fso moe: the FC2 tiles' mainloop (same type, its own TMA descriptors) and epilogue.\n"
        "    CollectiveMainloop collective_mainloop2(params.mainloop2, cluster_shape, cta_rank_in_cluster);\n"
        "    CollectiveEpilogue2 collective_epilogue2(params.epilogue2, shared_storage.tensors.epilogue.epilogue2);\n",
        1, "m7b the FC2 mainloop and epilogue objects")
    lit("    auto load_inputs = collective_mainloop.load_init(\n        problem_shape_MNKL, shared_storage.tensors.mainloop);\n",
        "    auto load_inputs = collective_mainloop.load_init(\n        problem_shape_MNKL, shared_storage.tensors.mainloop);\n"
        "    auto load_inputs2 = collective_mainloop2.load_init(\n        problem_shape2_MNKL, shared_storage.tensors.mainloop);\n",
        1, "m7c the FC2 load partitioning")
    lit("    int fso_rows = 0;      // compact layout: the expert's routed rows (the epilogue's store bound)\n",
        "    int fso_rows = 0;      // compact layout: the expert's routed rows (the epilogue's store bound)\n"
        "    int fso_phase = 0;     // fso moe: 0 = an FC1 tile, 1 = an FC2 tile\n"
        "    bool fso_ready = false; // fso moe: an FC2 tile whose entry's FC1 tiles had all released when it was drawn\n",
        1, "m7d the phase and ready locals")
    lit("fso_live = fso_tile_live(work_tile_info, fso_a_batch, fso_rows);",
        "fso_live = fso_tile_live_moe(work_tile_info, fso_a_batch, fso_rows, fso_phase, fso_ready);", 5,
        "m8 tile facts with phase and ready bit")

    # m9: the load warp's tile body, ONE path for both phases. The two phases' mainloop objects and load partitionings
    # have the same types, so the tile selects them by reference and runs one inlined copy of the load: the FC1 tiles
    # with the ready bit forced on and no proxy fence, the FC2 tiles with the scheduler's ready bit and the fence.
    # (A first form kept the stock two-call FC1 body and a separate FC2 body: 5600 SASS instructions against 4312 for
    # today's row-bounded FC1 kernel, and 2.1 to 2.3 thousand L1.5 instruction misses per launch against 0.4 to 1.3
    # thousand; run fso_issue13_20261031/I13-L5, ncu/kern.)
    head = "       if (fso_live) {   // fso per-tile skip: a dead tile costs this CTA no mainloop traffic\n"
    tail = "       }\n        // fso: the load-order arrival is unconditional"
    i0 = s.index(head)
    assert s.count(head) == 1, "[m9] load-warp tile body head not unique"
    i1 = s.index(tail, i0)
    body = s[i0 + len(head):i1]
    assert body.count("collective_mainloop.load(") == 2, "[m9] the load-warp tile body does not hold the two stock load calls"
    one = ("        // fso moe: the tile's phase selects its operands; both phases load through one inlined copy of\n"
           "        // `fso_load_after_entry` (slot_moe_mainloop.cuh). An FC2 tile's activation (B, SFB) is what the FC1\n"
           "        // tiles of its entry write in this kernel: it loads after the ready bit the scheduler drew it with, or\n"
           "        // streams its weights first and waits for the entry itself; an FC1 tile loads as before.\n"
           "        bool const fso_fc2 = fso_phase != 0;\n"
           "        CollectiveMainloop& fso_cm = fso_fc2 ? collective_mainloop2 : collective_mainloop;\n"
           "        auto const& fso_li = fso_fc2 ? load_inputs2 : load_inputs;\n"
           "        auto const fso_ps = fso_fc2 ? problem_shape2_MNKL : problem_shape_MNKL;\n"
           "        fso_set_load_rows(fso_cm, fso_rows, cta_coord_mnkl, CtaShape_MNK{});\n"
           "        auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, fso_ps, CtaShape_MNK{}, fso_li.k_tiles);\n"
           "        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, fso_ps, CtaShape_MNK{});\n"
           "        auto [mainloop_producer_state_next, unused_] = fso_cm.fso_load_after_entry(\n"
           "          mainloop_pipeline,\n"
           "          mainloop_pipe_producer_state,\n"
           "          fso_shift_weight_batch(fso_li, fso_a_batch - int(cute::get<3>(cta_coord_mnkl))),\n"
           "          cta_coord_mnkl,\n"
           "          k_tile_iter, k_tile_count,\n"
           "          fso_fc2 ? fso_ready : true,\n"
           "          fso_fc2,\n"
           "          scheduler.done_counter(int(cute::get<3>(cta_coord_mnkl))),\n"
           "          scheduler.done_target(work_tile_info)\n"
           "        );\n"
           "        mainloop_pipe_producer_state = mainloop_producer_state_next;\n")
    s = s[:i0] + head + one + s[i1:]
    report.append("edit m9 load warp: one load path for both phases: 1 site")

    lit("      do {\n        auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});\n",
        "      do {\n        // fso moe: the k-tile count of the tile's phase\n"
        "        auto k_tile_count = fso_phase == 0\n"
        "          ? TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{})\n"
        "          : TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape2_MNKL, CtaShape_MNK{});\n",
        1, "m10 MMA warp: the phase's k-tile count")

    # m11: the epilogue warps' store, split by phase.
    st_head = ("        if (scheduler.compute_epilogue(work_tile_info)) {\n"
               "          auto [load_state_next, store_state_next, acc_state_next] = collective_epilogue.template store<IsOverlappingAccum>(\n")
    st_tail = "          do_tail_store = true;\n        }\n"
    assert s.count(st_head) == 1, "[m11] epilogue store head not unique"
    j0 = s.index(st_head)
    j1 = s.index(st_tail, j0) + len(st_tail)
    old_block = s[j0:j1]
    store_call_1 = old_block[len("        if (scheduler.compute_epilogue(work_tile_info)) {\n"):old_block.index(st_tail)]
    assert "            shared_storage.tensors.epilogue\n" in store_call_1, "[m11] the store call's shared storage"
    assert "fso_store_shape(problem_shape_MNKL, fso_rows)" in store_call_1, "[m11] the store call's store bound"
    store_call_1 = store_call_1.replace("            shared_storage.tensors.epilogue\n",
                                        "            shared_storage.tensors.epilogue.epilogue1\n")
    store_call_2 = (store_call_1.replace("collective_epilogue.template store", "collective_epilogue2.template store")
                    .replace("shared_storage.tensors.epilogue.epilogue1", "shared_storage.tensors.epilogue.epilogue2")
                    .replace("fso_store_shape(problem_shape_MNKL, fso_rows)", "fso_store_shape(problem_shape2_MNKL, fso_rows)"))
    ind = lambda t: "".join("  " + ln if ln.strip() else ln for ln in t.splitlines(keepends=True))
    new_block = ("        if (scheduler.compute_epilogue(work_tile_info)) {\n"
                 "          if (fso_phase == 0) {   // fso moe: an FC1 tile, the fused SwiGLU + MXFP8 store\n"
                 + ind(store_call_1) +
                 "            // fso moe, the producer half of the FC1 -> FC2 hand-off: every epilogue thread has issued\n"
                 "            // its stores of this tile's fp8 rows and scale bytes; after the barrier one thread releases\n"
                 "            // the tile's count on the entry's done counter, which by cumulativity orders all 128 threads'\n"
                 "            // stores before any thread that acquires the count (CUTLASS's GenericBarrier::arrive_inc).\n"
                 "            // The count is the scheduler's: one per tile in slot mode; in token-tile mode a group's last\n"
                 "            // tile releases the whole group (its earlier tiles ran on this CTA before it) and the others\n"
                 "            // nothing.\n"
                 "            uint32_t const fso_release = scheduler.fso_release_count(work_tile_info);\n"
                 "            if (fso_release != 0u) {\n"
                 "            cutlass::arch::NamedBarrier::sync(NumEpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);\n"
                 "            if (threadIdx.x % NumEpilogueThreads == 0) {\n"
                 "              cutlass::gemm::kernel::detail::fso_red_release_gpu_add(\n"
                 "                scheduler.done_counter(int(cute::get<3>(cta_coord_mnkl))), fso_release);\n"
                 "            }\n"
                 "            }\n"
                 "          } else {   // fso moe: an FC2 tile, the plain bf16 store\n"
                 + ind(store_call_2) +
                 "          }\n"
                 "          do_tail_store = true;\n"
                 "        }\n")
    s = s[:j0] + new_block + s[j1:]
    report.append("edit m11 epilogue warps: the phase's store and the FC1 release: 1 site")

    lit("      collective_mainloop.prefetch_tma_descriptors();\n",
        "      collective_mainloop.prefetch_tma_descriptors();\n      collective_mainloop2.prefetch_tma_descriptors();\n",
        1, "m12 prefetch the FC2 descriptors")
    # The epilogue-load warp is not a participant for these NoSmem epilogues, but its call must name a storage.
    lit("            shared_storage.tensors.epilogue,\n            reverse_epi_n\n",
        "            shared_storage.tensors.epilogue.epilogue1,\n            reverse_epi_n\n", 1,
        "m12b the epilogue-load call names the FC1 storage")
    assert "shared_storage.tensors.epilogue\n" not in s and "shared_storage.tensors.epilogue)" not in s \
        and "shared_storage.tensors.epilogue," not in s, "[m-check] an unqualified epilogue storage use survived"
    report.append("edit m-check every epilogue storage use names its epilogue: checked")
    return s


def generate_moe(cutlass_dir: str) -> tuple[str, list[str], str]:
    """The fused FC1 + FC2 class in a header of its own, to be included after sm100_slot_gemm_kernel.hpp (whose
    compact helpers it uses) by the fused kernel's translation unit only."""
    text, _report, src_path = generate(cutlass_dir)
    report: list[str] = []
    cls = _compact_class(text, [])
    moe = _moe_class(cls, report)
    head = ("// Generated by csrc/gemm/tools/make_sm100_slot_kernel.py --moe from CUTLASS's\n"
            "// include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp (see that script's docstring and\n"
            "// _moe_class). Include after blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp.\n"
            "#pragma once\n\n"
            "#include \"blockscale_gemm/arch/sm100/mxfp8/sm100_slot_gemm_kernel.hpp\"\n"
            "#include \"blockscale_gemm/arch/sm100/mxfp8/slot_moe_scheduler.cuh\"\n\n"
            "namespace cutlass::gemm::kernel {\n")
    out = head + MOE_HELPERS + "\n" + moe + "\n\n} // namespace cutlass::gemm::kernel\n"
    return out, report, src_path


def main() -> int:
    args = sys.argv[1:]
    moe = bool(args) and args[0] == "--moe"
    if moe:
        args = args[1:]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        print("usage: make_sm100_slot_kernel.py [--moe] <output-header>", file=sys.stderr)
        return 2
    dst = args[0]
    cutlass_dir = _cutlass_dir()
    text, report, src_path = generate_moe(cutlass_dir) if moe else generate_with_compact(cutlass_dir)
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
    what = "sm_100 fused FC1 + FC2 slot kernel" if moe else "sm_100 slot kernel"
    print(f"[fso] {what}: " + "; ".join(report))
    print(f"[fso] {what}: {state} {dst} ({len(text)} bytes) from {src_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
