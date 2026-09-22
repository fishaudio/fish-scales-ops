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

Each edit asserts its exact match count, so a CUTLASS bump that moves any of
these sites fails the build loudly instead of producing a silently different
kernel.

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

    return s, report, src_path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        print("usage: make_sm100_slot_kernel.py <output-header>", file=sys.stderr)
        return 2
    dst = sys.argv[1]
    cutlass_dir = _cutlass_dir()
    text, report, src_path = generate(cutlass_dir)
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
