# HPC Project — Completion Plan

## Context

The project implements protein secondary-structure prediction on CB513 (window=13, one-hot, 260-d input, 3 classes H/E/C) with an MLP (256→128→3), comparing serial vs multiple parallelization strategies per the course guideline. Serial baseline (test Q3 = 62.74%, 441s) and OpenMP variant (~3.3× speedup at 4 threads, accuracy matches serial) are done and logged. Remaining deliverables per `Project Guideline pdf.pdf`:

1. Shared-memory pthreads (complement to OpenMP) — **required**
2. Distributed MPI — **required**
3. Hybrid MPI+OpenMP — **required**
4. CUDA — **required** (included for full marks since it appears in the technology list)
5. Analysis report — **required**

Goal: land the three missing parallel variants with the same data-parallel gradient-reduction pattern used in OpenMP, collect a minimum-but-sufficient experiment matrix, then write the report.

## Execution Order (sequential — do not parallelize phases)

### Step 1 — Pthreads variant (`src/pthreads/train.c`, Makefile target `train_pthreads`)
- **Pattern:** master/worker pool created **once** at startup, reused across batches.
- Each worker owns: `MLPGrad` buffer + scratch (`a1`, `a2`, `logits`, `probs`). Match OpenMP per-thread layout.
- Sync: barrier-based — workers wait on a "start batch" condition, compute their sample range, signal "done"; master reduces and calls `mlp_sgd_update`.
- Partition: contiguous sample-range split of the mini-batch (same as OpenMP `schedule(static)`).
- Accuracy check: reproduce serial Q3 within tolerance at seed=42.
- Updates checklist Phase 6.

### Step 2 — MPI variant (`src/mpi/train.c`, target `train_mpi`)
- Shard training set by rank: each rank owns `N/size` contiguous samples for the epoch (shuffled identically via broadcast seed).
- Per mini-batch: each rank computes local gradient sum over its sub-batch → `MPI_Allreduce(SUM)` on the full flattened gradient buffer → every rank applies the same `mlp_sgd_update`.
- Broadcast initial weights from rank 0. Validation/test only on rank 0 (others skip).
- Timing: `MPI_Barrier` before/after epoch; log `max` per-rank epoch time.
- Ranks to benchmark: 1, 2, 3 (3 laptops per checklist, but single-node multi-rank also acceptable for demo).
- Updates checklist Phase 7.

### Step 3 — Hybrid MPI+OpenMP (`src/hybrid/train.c`, target `train_hybrid`)
- Reuse the MPI outer loop from Step 2 verbatim.
- Replace the local sub-batch loop with the OpenMP parallel-for + per-thread gradient reduction from `src/openmp/train.c`.
- Only rank 0 allocates validation buffers; others skip.
- Benchmark matrix (trim from checklist): `1×1`, `1×4`, `2×4`, `3×4` — enough to show both axes scale.
- Updates checklist Phase 8.

### Step 4 — CUDA variant (`src/cuda/train.cu`, target `train_cuda`)
- Port forward/backward to GPU. Two acceptable approaches; pick the simpler:
  - **(a) cuBLAS GEMM-based:** batch matrix multiplies for W1·X, W2·a1, W3·a2; custom kernels only for ReLU, softmax, cross-entropy gradient. Recommended.
  - **(b) Hand-written kernels** for every layer. More code, more learning, slower to land.
- Host pipeline: copy whole train set to device once, mini-batch indexing on device, accumulate gradients in device buffers, single `cublasSaxpy`-style update.
- Validate Q3 within ±1% of serial (GPU float reductions can differ slightly).
- Benchmark batch sizes: 32, 64, 128, 256.
- Record GPU model + CUDA/cuBLAS version in `results/cuda/`.
- Updates checklist Phase 9.

### Step 5 — Experiments (Phase 10)
- Run each variant with fixed config: `hidden1=256 hidden2=128 epochs=80 lr=0.01 seed=42` (same as serial baseline).
- Sweeps:
  - OpenMP: threads ∈ {1, 2, 4, 8, 16}
  - Pthreads: threads ∈ {1, 2, 4, 8, 16}
  - MPI single-node: ranks ∈ {1, 2, 3}
  - **MPI 3-laptop run:** see "Multi-laptop MPI runbook" below — ranks ∈ {1, 2, 3} across physical machines
  - Hybrid: {1×1, 1×8, 1×16, 2×8, 3×8} per checklist Phase 8.3
  - CUDA: batch sizes {32, 64, 128, 256}
- Run each timing config 3 times for mean/std (drop to 1 only if time is critical).
- All JSON logs land under `results/<variant>/`.

#### Multi-laptop MPI runbook (delivered as `scripts/cluster_setup.md` + `scripts/hosts.txt`)
Document and (where possible) script:
1. Install matching OpenMPI version on all 3 laptops (`apt install openmpi-bin libopenmpi-dev`).
2. Same Linux user on each, passwordless SSH from launcher to the other two (`ssh-copy-id`).
3. Identical project path on every laptop (rsync the repo + `data/processed/` to each).
4. `hosts.txt` with one line per machine, e.g. `laptop1 slots=1`, `laptop2 slots=1`, `laptop3 slots=1`.
5. Smoke test: `mpirun --hostfile hosts.txt -np 3 hostname`.
6. Real run: `mpirun --hostfile hosts.txt -np 3 ./train_mpi --data data/processed/cb513/binary --epochs 80 --batch 64 --lr 0.01 --seed 42 --out results/mpi/cluster_3node`.
7. Logging: each rank writes `rank<r>_*.json`; rank 0 writes the aggregated metrics. Capture `max` epoch time across ranks.

### Step 6 — Plots & report (Phases 11–12)
- Plots (`plots/`): time-vs-threads and speedup-vs-threads for OpenMP & pthreads on one chart; time-vs-ranks for MPI (single-node + 3-laptop bars); hybrid grouped bar chart; CUDA time-vs-batch + GPU-vs-CPU comparison.
- Report sections per checklist 12.1. Include four diagrams: OpenMP gradient-reduction flow, pthreads master/worker, MPI/hybrid Allreduce, CUDA host/device dataflow. Accuracy table showing every variant's test Q3 matches serial within ±0.5% (CUDA within ±1%).

### Step 7 — Checklist + README
- Mark Phase 6/7/8/10/11/12 items in `docs/roadmap_n_progress/PROJECT_TODO_CHECKLIST.md` as code lands.
- Add `README.md` build/run instructions (Phase 0.1).

## Critical Files to Create / Modify

- Create: [src/pthreads/train.c](src/pthreads/train.c), [src/mpi/train.c](src/mpi/train.c), [src/hybrid/train.c](src/hybrid/train.c), [src/cuda/train.cu](src/cuda/train.cu), [scripts/cluster_setup.md](scripts/cluster_setup.md), [scripts/hosts.txt](scripts/hosts.txt)
- Reuse (no changes needed): [src/models/mlp.c](src/models/mlp.c), [src/common/*](src/common/), [include/models/mlp.h](include/models/mlp.h)
- Reference implementation to mirror: [src/openmp/train.c](src/openmp/train.c)
- Modify: [Makefile](Makefile) (targets already stubbed — verify they build), [docs/roadmap_n_progress/PROJECT_TODO_CHECKLIST.md](docs/roadmap_n_progress/PROJECT_TODO_CHECKLIST.md)

## Reusable APIs (from exploration)

- `mlp_forward/backward/sgd_update` — backward only writes into caller-provided `MLPGrad`, so thread/rank-local buffers are safe without locks.
- `mlpgrad_alloc/zero` — use for per-worker grad buffers.
- `grad_reduce()` (in `src/openmp/train.c`) — element-wise grad sum, lift into a shared helper if needed by pthreads/MPI.
- CLI already exposes `--threads`; add `--data --epochs --batch --lr --seed --out` are already wired.

## Known Caveat

`mlp_backward()` does internal per-call mallocs for `da1/da2/dz2`. This is OK for correctness in pthreads/MPI, but can be lifted to pre-allocated per-worker scratch later if we see malloc contention in timing profiles. Do not refactor preemptively — finish variants first, optimize only if needed.

## Verification

After each variant:
1. `make <target>` builds clean.
2. Short smoke run (2 epochs, seed=42, threads=1 or rank=1): loss decreases, final val_q3 close to serial's epoch-2 value (within ±0.5%).
3. Full 80-epoch run: test Q3 within ±0.5% of serial's 62.74%; JSON log written to `results/<variant>/`.
4. Scaling run: report wall-clock train time vs thread/rank count.

## Out of Scope

- BLOSUM62 encoding comparison (already coded but not required for final report).
- `mlp_backward` malloc refactor — only revisit if profiling shows it dominates.

## Risks / Fallbacks

- **CUDA execution:** user will run on lab or friend's machine — code must be self-contained and buildable with just `nvcc` + cuBLAS. Add a short `scripts/cuda_setup.md` with build/run commands and the env vars needed (CUDA_HOME, library paths) so the remote run is one command.
- **3-laptop MPI:** if SSH/network setup blocks on the day, ship the runbook + a single-node 3-rank result and note the constraint in the report.
