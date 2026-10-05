# Release harness

A release is built once and then tested and measured as that one artifact. This page is the order of operations,
with the script that owns each step. Nothing on it is run by hand from memory: every step is a script in this
repository, and every environment a step depends on is written down in a file the step checks.

| step | script | what it pins or records |
|---|---|---|
| 1. build the wheel | `scripts/build_wheel.sh` | the toolchain (`docker/build-wheel.Dockerfile`) and `BUILD_INFO.json` inside the wheel |
| 2. checks without a GPU | `scripts/ci/cpu_checks.sh` | — |
| 3. test on each machine | `scripts/ci/run_suite.py`, `scripts/ci/remote_suite.sh` | the machine's lock, `bench/env/<machine>.lock.json`; `result.json` |
| 4. measure on each machine | `bench/run_perf.py` | the same lock; `manifest.json` |
| 5. install the tables | `bench/gemm/python/perf_report.py install` | `tests/baselines/provenance/<machine>.json`, rendered into `docs/perf/README.md` |

The machines are `h200` (sm_90), `5090` (sm_120) and `b300` (sm_103). The development flow, `scripts/build.sh`, builds
in place for one machine and is for iteration only: no table and no release validation comes from it.

## 1. Build the wheel

```bash
scripts/build_wheel.sh --out dist            # committed tree only; --allow-dirty for a development build
```

The build runs in a fixed container: Ubuntu 22.04 (glibc 2.35, so the extension loads on every serving host), the
CUDA 13.2.1 toolkit, Python 3.12 and torch 2.13.0+cu130. One extension carries the kernels of all three
architectures.

- **What the wheel carries.** Everything the package needs at run time, so no machine needs the source tree: the
  NVRTC 13.2.78 that compiles the sm_90 kernels, the sm_90 JIT include tree, the sm_100 CuTe-DSL kernel and
  `BUILD_INFO.json`. `fish_scales_ops.build_info()` returns that file's content.
- **Outputs in `dist/`.** The wheel, `<wheel>.sha256`, `BUILD_INFO.json`, the build log and a test kit. The test kit
  is the `tests/`, `bench/`, `scripts/` and `docs/` of the same commit; steps 3 and 4 run from it.
- **Torch guard.** The wheel is tied to the torch it was built against. Importing it under another torch raises
  `ImportError`.

## 2. Checks without a GPU

```bash
scripts/ci/cpu_checks.sh --dist dist
```

It runs in the build image:
- the source-tree checks (the JIT-isolation lint, the generated doc blocks, the rendered performance tables);
- the device-independent tests against the installed wheel, with no source tree on the path.

`.github/workflows/wheel.yml` runs steps 1 and 2 on every push to `master`, on pull requests that touch code, and on
tags. On a tag it attaches the outputs to the GitHub release.

## 3. Test on each machine

On the machine itself:

```bash
python3 scripts/ci/run_suite.py --wheel W.whl --testkit KIT.tar.gz --work /path/run1 --machine h200
```

From another host, which copies the three files over ssh and fetches the result:

```bash
scripts/ci/remote_suite.sh --ssh "ssh <host>" --remote-dir /path/run1 --wheel dist/W.whl --testkit dist/KIT.tar.gz \
    -- --machine 5090
```

**What a run does.**
- It installs the wheel with `pip install --no-deps --target` into the run directory, so a shared serving venv is not
  modified.
- It takes the machine's GPU lock file and refuses a card that already runs a compute process.
- It runs every test of the kit, then the attention pytest and the render check.
- It compares the MoE layer's outputs bitwise with the reference dump the lock names
  (`tests/tools/dump_layer_outputs.py`).

`result.json` records the wheel's sha256, `build_info()`, the card and each test's exit code. The exit code is 0
only when everything passed.

## 4. Measure on each machine

```bash
python3 bench/run_perf.py --machine h200 --out /data/bench-runs/<run> --fso-path /path/run1/site --dry-run
python3 bench/run_perf.py --machine h200 --out /data/bench-runs/<run> --fso-path /path/run1/site
```

- **Preflight.** The run first compares the machine with its lock: the interpreter and the exact package versions
  of every environment a step uses, the driver, the card, the clock policy, and on sm_90 the compiler the JIT loaded.
  A difference is drift, and the run refuses it. `--dry-run` prints the plan and that verdict without touching a GPU.
- **Steps.** The steps of every table of record are listed once, in `bench/perf_suite.py`.
- **Manifest.** `manifest.json` in the run directory records what ran and on what.

`docs/perf/README.md` §7c describes the lock files and the refusals in full.

## 5. Install the tables

```bash
python bench/gemm/python/perf_report.py diff --a baselines --b /data/bench-runs/<run> --device h200
python bench/gemm/python/perf_report.py install --run /data/bench-runs/<run>
```

`install` requires the run's manifest. It refuses a smoke run, an incomplete run, and a run with drift unless
`--accept-drift`. It writes the provenance of each installed baseline file into
`tests/baselines/provenance/<machine>.json`, and `render_perf_docs.py` renders the "Environments of record" block
of `docs/perf/README.md` from it. The acceptance rule of a performance change is unchanged: every affected cell
faster than, or within ±1 % of, the installed baseline (`docs/perf/README.md`).

## Changing a pinned environment

A new torch, a new comparator version, a new driver or another card is a change to a lock file, made on purpose and
committed. The next run then measures every table on that environment, and the provenance records it. To serve
another torch, build another wheel: change the pin in `docker/build-wheel.Dockerfile`.
