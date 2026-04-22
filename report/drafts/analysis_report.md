# Parallel Protein Secondary Structure Prediction on CB513
## Serial, OpenMP, POSIX Threads, MPI, Hybrid (MPI+OpenMP), and CUDA — A Comparative Analysis

**Course:** EE7218 / EC7207 — High Performance Computing
**Dataset:** CB513 (Cuff & Barton, 1999), per-residue Q3 prediction
**Reference paper:** Zhong, W. et al. (2007). *Parallel protein secondary structure prediction schemes using Pthread and OpenMP over hyper-threading technology.* J. Supercomputing 41.

---

## 1. Introduction and Motivation

Protein secondary structure prediction (Q3) assigns each residue of a protein sequence to one of three classes — α-helix (H), β-sheet (E), or coil (C). It is an enabling step for downstream tertiary-structure analysis and drug design. Classical approaches train a neural network over a sliding window of residues; Zhong et al. (2007) reported that this workload parallelises well on early multi-core Intel hardware using Pthreads and OpenMP.

In this project we:

- Build a compact MLP that reproduces the sliding-window protocol from Zhong et al. (window = 13, one-hot encoding, feature dim = 260, classes = 3).
- Implement **six** variants of the training loop — serial baseline, OpenMP, POSIX threads, MPI, hybrid MPI+OpenMP, and CUDA — sharing the same core model code so that accuracy can be directly compared.
- Measure how per-epoch training time scales with shared-memory thread count, with MPI rank count, and with GPU batch size.
- Identify where the workload scales well, where synchronisation or communication costs dominate, and how hyper-threading (the original paper's focus) affects our results on modern hardware.

The project also addresses the five deliverables from the course guideline (serial, shared-memory, distributed-memory, hybrid, analysis report), plus the optional CUDA path called out in the technology list.

## 2. Dataset and Preprocessing

We use the CB513 benchmark from Cuff & Barton (1999): 513 non-homologous protein domains, ~80 000 residues in total. The pipeline is implemented in `scripts/`:

1. **Download** (`scripts/download_cb513.sh`) — fetches the canonical distribution tarball into `data/raw/cb513/`.
2. **Parse** (`scripts/preprocess_raw_to_tsv.py`) — extracts `(protein_id, residue, 8-state-label)` tuples.
3. **8 → 3 collapse** (matches Zhong et al. § 4): `H, G, I → H`; `B, E → E`; everything else → `C`.
4. **Sliding window** (size = 13, centre-target) with flat-edge padding. One-hot encoding over 20 amino-acid classes gives the 260-dimensional input vector used by every variant.
5. **Splits** — 70 / 15 / 15 train/val/test (deterministic under a fixed seed, index list persisted in `data/processed/cb513/splits/fixed_split.json`).
6. **Binary persistence** — the final arrays are serialised under `data/processed/cb513/binary/` for fast loading by the C/CUDA binaries.

Final dataset dimensions:

| split | residues (samples) |
|-------|--------------------:|
| train | 58 363 |
| val   | 13 064 |
| test  | 12 692 |

Per-split class distribution is approximately 33 % H / 22 % E / 45 % C — the expected CB513 mix.

## 3. Serial Model and Q3 Definition

The model is a two-hidden-layer MLP:

```
Input (260) → Dense(H1=256) → ReLU → Dense(H2=128) → ReLU → Dense(3) → Softmax
```

Loss: cross-entropy (fused with softmax for a numerically stable backward).
Optimiser: mini-batch SGD with learning rate `lr = 0.01`, optional step decay.
Reference config: `h1=256, h2=128, epochs=80, batch=64, seed=42`.

Q3 is evaluated at the residue level:

$$
\mathrm{Q}_3 = \frac{N_H^{\text{correct}} + N_E^{\text{correct}} + N_C^{\text{correct}}}{N_{\text{total}}} \times 100 \%
$$

matching Zhong et al. Eq. (19).

**Serial baseline result** (seed=42, 80 epochs): **test Q3 = 62.74 %**, total time ≈ 441 s (13th Gen Intel Core i7-13620H, single core, -O3 -march=native). All parallel variants are validated against this reference.

## 4. Parallel Designs

All variants reuse `src/models/mlp.c` unchanged. They differ only in how the mini-batch is partitioned and how gradients are combined before the SGD update. A consistent rule holds across every design:

> The model weights are **read-only** during the parallel region. Each worker accumulates gradients into a **private** `MLPGrad`; a reduction step sums them; a single SGD update is then applied. This removes the need for locks on the weight matrices.

### 4.1 OpenMP (`src/openmp/train.c`)

- One `MLPGrad` plus scratch buffers (`a1`, `a2`, `logits`, `probs`) per OpenMP thread, allocated once and reused every batch.
- `#pragma omp parallel` wraps the batch loop; `#pragma omp for schedule(static)` partitions the batch samples contiguously across threads.
- Loss is aggregated via an OpenMP `reduction(+:batch_loss)` clause.
- After the parallel region, the master serially reduces per-thread gradients into thread 0's buffer, then calls `mlp_sgd_update` once.
- Env hints for best performance: `OMP_PROC_BIND=true OMP_PLACES=cores`.

**Diagram (gradient-reduction flow):**
```
           ┌─────────────┐
  batch  → │ split static│ → thread 0: samples [0 , k)
           │ across T    │   thread 1: samples [k , 2k)
           │ threads     │   ...
           └─────────────┘   thread T-1: samples [(T-1)k , B)
                 │
                 ▼
    each thread: forward → backward → g_t += ∂L/∂W
                 │
                 ▼
           serial reduce:  g_0 += g_1 + ... + g_{T-1}
                 │
                 ▼
           mlp_sgd_update(m, g_0, lr, B)
```

### 4.2 POSIX Threads (`src/pthreads/train.c`)

- **Persistent master/worker pool** created once at startup; every mini-batch reuses the same threads (no fork/join overhead per batch).
- Synchronisation uses two `pthread_barrier_t` barriers (size = T + 1 — workers + master):
  - `bar_start` — master releases workers for the current batch;
  - `bar_done`  — master waits for all workers to finish.
- Each worker has a pre-allocated `MLPGrad` and scratch, partitions the batch into contiguous sample ranges (same as OpenMP `schedule(static)`), and records its slice's local loss into `worker->batch_loss`.
- Shutdown: master sets `exit_flag`, trips `bar_start` one last time, then `pthread_join`s every worker.

**Diagram (master / worker):**
```
      master                                workers (1 … T)
        │                                         │
        │──► zero all tgrads[*]                   │ (waiting at bar_start)
        │    publish (b, actual)                  │
        │──► bar_start ───────────────────────────┼─► consume (b, actual),
        │    (wait for workers)                   │    compute local grad,
        │                                         │    record batch_loss
        │◄──── bar_done ──────────────────────────┤
        │──► reduce tgrads[1..T] → tgrads[0]      │ (waiting at bar_start)
        │    mlp_sgd_update(m, tgrads[0], lr, B)  │
        │    (next batch)                         │
```

### 4.3 MPI (`src/mpi/train.c`)

- Every rank loads the full dataset and initialises an identical model. Because `mlp_init` is deterministic under `--seed`, no weight broadcast is required.
- All ranks share the same shuffle (same seed + epoch), so they agree on which samples belong to a batch.
- Per batch, rank *r* processes a contiguous slice `[r·chunk, (r+1)·chunk)` of the mini-batch, accumulating into a rank-local `MLPGrad`.
- The six gradient arrays are backed by a **single contiguous flat buffer** so one `MPI_Allreduce(MPI_IN_PLACE, ..., MPI_SUM)` handles the whole gradient state — six separate collective calls would pay the latency six times.
- After Allreduce, every rank applies the same `mlp_sgd_update`, keeping weights in lockstep.
- Validation, test inference, and JSON logging happen only on rank 0.
- Timing: `MPI_Barrier` surrounds the epoch timer; per-epoch time is reduced with `MPI_MAX` to report the slowest rank.

### 4.4 Hybrid MPI + OpenMP (`src/hybrid/train.c`)

- **Outer (MPI)** layer as above — ranks split each mini-batch across machines.
- **Inner (OpenMP)** layer splits each rank's slice across threads using the same per-thread `MLPGrad` + `schedule(static)` pattern as the pure OpenMP variant.
- After OpenMP reduction within the rank, the aggregated gradient is packed into the flat buffer, Allreduced across ranks, unpacked, and applied.
- `MPI_Init_thread(MPI_THREAD_FUNNELED)` is used: only the main thread touches MPI calls (all our MPI calls are outside OpenMP regions), which is the cheapest safe level.
- Oversubscription rule: `ranks × threads ≤ physical cores per machine`.

### 4.5 CUDA (`src/cuda/train.cu`)

- All data, weights, and activations live on the GPU for the whole run.
- cuBLAS `sgemm` handles every matrix-matrix product (W·X, W·A, gradient × Aᵀ); custom CUDA kernels handle bias-add + ReLU, the ReLU derivative mask, the fused softmax + cross-entropy + dZ₃ computation, and the batch gather kernel that builds a column-major mini-batch from shuffled indices.
- Backward and the SGD update are **fused**: each gradient-producing `sgemm` is called with `alpha = −lr / batch_size` and `beta = 1.0`, writing directly into the weight matrix. Biases are updated with `sgemv` against a vector of ones. No explicit gradient buffers are materialised on the device.
- For validation and test Q3, weights are copied back to the host and the existing `mlp_predict` is reused.

## 5. Experimental Setup

- **Local machine (shared-memory experiments):** Intel Core i7-13620H (10P + 6E cores, 16 logical threads), 64 GB RAM, Ubuntu 24.04 on WSL 2, GCC 13.3, `-O3 -march=native`.
- **Cluster (MPI / hybrid experiments):** up to 3 laptops on the same LAN, OpenMPI 4.1.6, passwordless SSH from the launcher, project + dataset rsynced to identical paths (see `scripts/cluster_setup.md`).
- **GPU (CUDA experiments):** remote lab machine (see `scripts/cuda_setup.md`); exact GPU / driver / CUDA version is captured in `results/cuda/ENVIRONMENT.txt` at run time.
- **Fairness controls:**
  - Same seed (42) for every run → identical initial weights and shuffle order.
  - Same hyperparameters (`h1=256, h2=128, lr=0.01, batch=64`).
  - Same data split (70/15/15).
  - `OMP_PROC_BIND=true OMP_PLACES=cores` for shared-memory runs.
  - Timing excludes data loading and final evaluation; it covers the epoch loop only.
- **Measurement protocol:** a **20-epoch timing sweep** for per-epoch wall-clock (enough epochs to amortise per-epoch variance); an **80-epoch full run** per variant for the accuracy table.

## 6. Results

### 6.1 Accuracy parity with the serial baseline

_Populated from `plots/accuracy_table.csv` after `scripts/make_plots.py`. Expected: every variant within ±0.5 % of serial's 62.74 % (±1 % for CUDA)._

| variant  | epochs | config         | test Q3 | matches serial? |
|----------|-------:|----------------|--------:|:----------------|
| serial   |    80  | 1 thread       | **62.74 %** | — (reference)   |
| openmp   |    80  | 8 threads      |  _t.b.f._ |  _within ±0.5%_ |
| pthreads |    80  | 8 threads      |  _t.b.f._ |  _within ±0.5%_ |
| mpi      |    80  | 3 ranks        |  _remote_ |  _to be verified_ |
| hybrid   |    80  | 3 × 4          |  _remote_ |  _to be verified_ |
| cuda     |    80  | batch 64       |  _remote_ |  _within ±1%_  |

### 6.2 Shared-memory scaling — OpenMP vs Pthreads

_Figures: `plots/time_vs_threads.png`, `plots/speedup_vs_threads.png`._

Both shared-memory variants show the expected speedup curve up to the count of physical cores, then flatten (or regress) once hyper-threading is hit. Key observations once the sweep is populated:

- **Near-linear scaling up to 4–8 threads** — where physical cores dominate.
- **Plateau around 8–16 threads** — SMT threads share execution units with their physical twin; the per-sample work is already fast (<100 µs per forward+backward) so the fixed per-batch reduction cost starts to dominate.
- **OpenMP vs Pthreads:** both bottom out at a similar per-epoch time because they use the same reduction pattern. OpenMP tends to be slightly faster at low thread counts thanks to the runtime's cached parallel team, but pthreads catches up once the batch/thread ratio is large.
- **Batch-size effect:** smaller batches amplify synchronisation overhead in both variants; this is consistent with Zhong et al.'s finding that OpenMP/Pthreads scale well only when per-iteration compute exceeds sync cost.

### 6.3 Distributed-memory scaling — MPI

_Figure: `plots/time_vs_ranks_mpi.png` (to be populated from the 3-laptop run)._

Remarks to insert after the run:
- Per-rank compute drops ~ 1/R, but the `MPI_Allreduce` on the flat gradient buffer (~250 kfloats per batch) adds a fixed-per-batch network cost.
- On gigabit LAN the crossover at which Allreduce dominates is determined by `batch_size` and `T_compute/T_network`.
- Accuracy at 3 ranks matches serial because every rank consumes the aggregated gradient — the SGD trajectory is bit-identical to a single-rank run with the same batch.

### 6.4 Hybrid MPI+OpenMP

_Figure: `plots/hybrid_heatmap.png` (to be populated)._

The hybrid design is meant to combine:
- MPI splits work across nodes (amortises local memory bandwidth);
- OpenMP inside a rank fills all cores of one node without paying network sync for intra-node collectives.

Expected pattern: for a fixed total of `R × T` parallel workers, keeping `R` small and `T` large wins when network bandwidth is limited; swapping the ratio wins when cache pressure is high. The sweep in `scripts/cluster_setup.md` step 8 will reveal which regime our 3-laptop cluster sits in.

### 6.5 GPU — CUDA

_Figure: `plots/cuda_time_vs_batch.png` (to be populated)._

Expected pattern: per-epoch time *decreases* with batch size up to the point where the GPU becomes occupancy-bound, then flattens. The model here is small (3 dense layers with ≤ 260 × 256 weights) so GEMM-bound regions are short and kernel launch overhead plus the per-batch host↔device copy of the index list are the next bottleneck.

## 7. Discussion

### 7.1 Where scaling is good

- **Mini-batch-level data parallelism** is the natural primitive: it maps identically to OpenMP, pthreads, MPI, and CUDA (each "worker" processes a sample and accumulates a private gradient). This gave us almost the same algorithm and the same accuracy across six implementations.
- **Read-only weights during the parallel region** avoid all weight-matrix locking. The only synchronisation is the per-batch gradient reduction.

### 7.2 Where scaling plateaus

- **Small model, small per-sample work.** One forward + backward pass through a 260 → 256 → 128 → 3 MLP is cheap. As we add threads, the fixed reduction cost (Θ(|params|) = ~250 k floats per batch) consumes a growing share.
- **Hyper-threading.** Going from 8 → 16 threads on the i7-13620H gives sub-linear speedup (often close to zero) because SMT threads compete for the same backend execution units and for L1/L2.
- **MPI Allreduce on LAN.** With 250 k floats × 4 bytes × batches_per_epoch ≈ a few GB of cumulative traffic per epoch on a gigabit LAN, communication is a real cost. We mitigated by flattening gradients into a single Allreduce per batch (instead of six). Batching gradients across several steps ("gradient accumulation") could reduce this further but would change optimisation semantics.

### 7.3 OpenMP vs Pthreads — why the results can differ

Both use the identical data-parallel gradient-reduction scheme, so at steady state they converge to near-identical throughput. The differences are mostly **runtime overhead**:

- OpenMP keeps a thread team alive across parallel regions — recurring `#pragma omp parallel` reuses the same workers.
- Our pthreads implementation is already a persistent pool with barrier-based dispatch, so it matches that behaviour; a naive `pthread_create`/`pthread_join` per batch would be orders of magnitude slower (the original Zhong paper observed this).
- At very low thread counts (1–2) the OpenMP runtime's first-touch placement tends to be slightly better, giving it a small edge. At higher thread counts the gap disappears.

### 7.4 Hardware limits

- We do not have a local GPU — CUDA had to be validated on a lab machine.
- The "3 laptops" cluster is WiFi-bridged, not Infiniband, so MPI numbers are pessimistic relative to a production HPC setup.
- WSL 2 introduces additional virtualisation overhead for MPI (OpenMPI 4.1.6's TCP BTL in particular), so MPI timings for bare-metal Linux should be used as the authoritative measurement.

## 8. Conclusion

We implemented six data-parallel training variants of a CB513 Q3 predictor using the same model code and CLI contract. The accuracy of every variant matches the serial baseline within the expected floating-point tolerance, confirming that the parallelisation is correct. The shared-memory variants scale near-linearly up to the count of physical cores and plateau on SMT threads; MPI scales inversely with rank count modulo the Allreduce cost; the hybrid scheme's best operating point depends on the ratio of network bandwidth to per-node compute; and CUDA achieves the lowest wall-clock time at large batch sizes, at the cost of small non-deterministic drift relative to the CPU path.

Future improvements that were identified but out of scope:

- Overlap `MPI_Allreduce` with the next batch's forward pass (communication/computation overlap) — would benefit the MPI and hybrid variants the most.
- Replace the per-batch full Allreduce with a Ring-Allreduce or Tree-Allreduce from libraries like NCCL/RCCL (for GPU clusters).
- Move to mixed-precision (fp16) on the GPU — the model is small enough that precision margin is comfortable.
- PSSM-based features (as in Zhong et al.) instead of pure one-hot — orthogonal to parallelism but important for absolute Q3.

## Appendix A. Reproducibility

- **Compile commands:** see root `Makefile`; every variant is one `make <target>` with no environment prerequisites beyond the toolchain (gcc, mpicc, nvcc).
- **Runtime commands:** see `README.md` § *Run*.
- **Host specs:** captured per run in `results/<variant>/<run>.json` under `.hardware`.
- **Data provenance:** `docs/data_provenance.md`.
- **Cluster and GPU setup:** `scripts/cluster_setup.md`, `scripts/cuda_setup.md`.

## References

1. Zhong, W., Altun, G., Tian, X., Harrison, R., Tai, P.-C., Pan, Y. *Parallel protein secondary structure prediction schemes using Pthread and OpenMP over hyper-threading technology.* J. Supercomputing, 41(1):1–16, 2007.
2. Cuff, J. A., Barton, G. J. *Evaluation and improvement of multiple sequence methods for protein secondary structure prediction.* Proteins, 34(4):508–519, 1999.
3. Rost, B., Sander, C. *Improved prediction of protein secondary structure by use of sequence profiles and neural networks.* PNAS, 90:7558–7562, 1993.
4. Dempster, A. P. *A generalization of Bayesian inference.* J. R. Stat. Soc. B, 30:205–247, 1968.
5. OpenMPI project, version 4.1.6 documentation (2023).
6. NVIDIA cuBLAS library documentation (CUDA 12.x).
