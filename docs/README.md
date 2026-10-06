# fish-scales-ops documentation map

Where each kind of content lives, and the rules every page follows.

## Layout

```
README.md                  front page: what fso is, supported hardware, install, usage, a few hot-shape
                           numbers (sm_90 and sm_120), the TODO list
CHANGELOG.md               what changed in each release, known issues
docs/
  README.md                this file
  guide.md                 integration guide for a serving engine: what to call, build and install, dense
                           and MoE serving, CUDA graphs and torch.compile, deployment variables, migration
  harness.md               build the wheel once, test and measure that wheel on each machine, install the
                           tables; how CI builds the wheel
  api/
    dense.md               fish_scales_ops.dense, the stable dense linear
    moe.md                 fish_scales_ops.moe, the stable MoE layer
    attention.md           fish_scales_ops.attention
    compat.md              fish_scales_ops.compat (the explicit ops) and the reference every GEMM and MoE op
                           shares: scale layouts, constraints, torch.ops schemas, CUDA graphs, environment
                           variables
  perf/
    README.md              how the tables are measured, the shape families, the per-machine summary of
                           where fso is faster or slower than the serving libraries, environments of record
    gemm/sm{90,100,120}.md     one GEMM or grouped kernel per row, per machine
    layer/README.md            what each family's MLP / MoE block contains
    layer/sm{90,100,120}.md    the whole block per row, with the comparators and their summaries
    attention/sm{90,100,120}.md
```

## Rules

1. **Report results and how to use the library, not how it is built inside.**
   The README and the perf pages give performance conclusions, installation
   and usage. Kernel internals, tuning history and design notes stay out of
   them; the code comments and the git history keep them.
2. **Performance numbers come from a program.** Every table and every summary
   in `docs/perf/` and the README's hot-shape tables are rendered from
   `tests/baselines/` by `bench/gemm/python/render_perf_docs.py`, and
   `render_perf_docs.py --check` fails when a page has drifted. Nobody types a
   measured number into a page. `docs/api/` carries no performance numbers.
3. **The README covers sm_90 and sm_120 only and compares against no other
   library.** Comparisons live in `docs/perf/layer/` and the summary of
   `docs/perf/README.md`. There, the comparator ran on the same card under the
   same protocol, its versions appear in the generated "Environments of record"
   block, and an untuned or fallback configuration is stated next to its column.
4. **One definition per fact.** Shape families, M grids and FLOPs conventions
   are defined once, in `docs/perf/README.md` (block contents in
   `docs/perf/layer/README.md`). `docs/perf/` is split by domain (gemm, layer,
   attention), then by SM; a number lives in exactly one file.
5. **Every architecture is stated.** Where fso has no kernel for an SM, the
   page says so and names what runs instead.
