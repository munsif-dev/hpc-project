# HPC Project End-to-End TODO Checklist
## Topic
Protein secondary-structure prediction on CB513 with:
Serial + OpenMP + POSIX threads + MPI + Hybrid (MPI+OpenMP) + CUDA, plus final analysis report.

## Source Alignment
- `research-paper.pdf` (Zhong et al., 2007): CB513, window size 13, Q3 metric, OpenMP vs pthread comparison.
- `HPC_CB513_NN_OpenMP_Pthreads_Roadmap.pdf`: implementation roadmap and experiment matrix.
- `Project Guideline.pdf`: required deliverables and report contents.

## How To Use This File
- Mark each task as done: `[ ]` -> `[x]`.
- Do tasks in section order unless a task is marked optional.
- Do not start performance comparisons before serial accuracy and preprocessing checks are stable.

---

## Phase 0: Project Setup And Planning
### 0.1 Repository structure
- [x] Create folders: `src/`, `include/`, `data/raw/`, `data/processed/`, `scripts/`, `results/`, `plots/`, `report/`, `docs/`.
- [ ] Add `README.md` with build and run instructions.
- [x] Add `Makefile` with separate targets:
  - [x] `train_serial`
  - [ ] `train_omp`
  - [ ] `train_pthreads`
  - [ ] `train_mpi`
  - [ ] `train_hybrid_mpi_omp`
  - [ ] `train_cuda`

### 0.2 Environment and toolchain check
- [x] Confirm C/C++ compiler and flags (`-O3 -march=native`).
- [x] Confirm OpenMP support (`-fopenmp`).
- [x] Confirm pthread support (`-pthread`).
- [ ] Confirm MPI install (`mpicc`, `mpirun`).
- [ ] Confirm remote CUDA environment access (lab/cloud/Colab).

### 0.3 Team execution plan
- [ ] Assign owners for modules:
  - [ ] Data pipeline
  - [ ] Serial baseline
  - [ ] Shared-memory (OpenMP/pthreads)
  - [ ] MPI/Hybrid
  - [ ] CUDA
  - [ ] Report and plots
- [ ] Define weekly checkpoints and internal deadlines.

Done criteria:
- Build system can compile at least placeholder binaries for all required targets.

---

## Phase 1: Data Gathering And Provenance
### 1.1 Choose CB513 source
- [x] Select one primary CB513 source and document exact URL.
- [x] Save source citation in `docs/data_provenance.md`.
- [x] Record download date, file names, and checksums.

### 1.2 Download and store raw files
- [x] Create script: `scripts/download_cb513.sh`.
- [x] Download dataset into `data/raw/`.
- [x] Keep raw files unchanged (read-only copy for reproducibility).

### 1.3 Label format confirmation
- [x] Verify whether labels are already 3-state (`H/E/C`) or 8-state DSSP.
- [x] If 8-state, implement mapping from paper:
  - [x] `H,G,I -> H`
  - [x] `B,E -> E`
  - [x] all others -> `C`

Done criteria:
- Raw dataset is available locally with documented provenance and label format.

---

## Phase 2: Preprocessing Pipeline
### 2.1 Parser and canonical data model
- [x] Implement parser for sequences and per-residue labels.
- [x] Validate sequence length equals label length for each protein.
- [x] Store canonical representation (protein ID, sequence, labels).

### 2.2 Sliding window generation (paper-aligned)
- [x] Implement window size `13` (center residue target).
- [x] Implement edge handling (padding policy) and document it.
- [x] Confirm input dimension for one-hot: `20 x 13 = 260`.

### 2.3 Feature encoding
- [x] MVP encoding: one-hot amino acid encoding.
- [x] Optional encoding: BLOSUM62 mode with same window pipeline.
- [x] Keep encoding selectable by CLI flag.

### 2.4 Dataset split protocol
- [x] Implement either:
  - [x] Fixed train/val/test split 70/15/15 (documented in metadata.json).
- [x] Ensure split is deterministic with fixed seed.
- [x] Save split indices for reproducibility.

### 2.5 Persist processed dataset
- [x] Write processed data to binary format for fast training.
- [x] Save metadata file (feature dim, class mapping, split info, counts).

### 2.6 Preprocessing sanity checks
- [x] Print total proteins and total residues.
- [x] Print class distribution (`H/E/C`) for train/val/test.
- [x] Confirm no NaN/Inf in features.
- [x] Confirm sample count after windowing matches expectation.

Done criteria:
- `data/processed/` contains deterministic, reusable training data + metadata.

---

## Phase 3: Metrics, Logging, And CLI Contract
### 3.1 Metrics
- [x] Implement Q3 metric exactly:
  - [x] `Q3 = (correct_H + correct_E + correct_C) / total_residues * 100`.
- [x] Implement per-class accuracy and confusion matrix.

### 3.2 Logging and output schema
- [x] Define JSON schema for each run:
  - [x] model/variant name
  - [x] seed, epochs, batch size, learning rate
  - [x] thread/rank config
  - [x] train/val/test Q3
  - [x] epoch times and total time
  - [x] hardware/compiler info
- [x] Save logs under `results/<variant>/`.

### 3.3 Unified CLI
- [x] Enforce common flags across all binaries:
  - [x] `--data`
  - [x] `--epochs`
  - [x] `--batch`
  - [x] `--lr`
  - [x] `--seed`
  - [x] `--threads` (where applicable)
  - [x] `--out`

Done criteria:
- All variants produce machine-readable logs in a common format.

---

## Phase 4: Serial Baseline (Reference Truth)
### 4.1 Model implementation
- [x] Implement compact MLP (two hidden layers + 3-class softmax).
- [x] Use cross-entropy loss.
- [x] Use SGD (mini-batch, with optional lr decay).

### 4.2 Training loop
- [x] Forward pass.
- [x] Backward pass.
- [x] Weight update.
- [x] Validation at each epoch.
- [x] Final test evaluation (Q3).

### 4.3 Stability checks
- [x] Fix RNG seed and confirm repeatability.
- [x] Run short smoke test (1-2 epochs) to verify loss decreases.
- [x] Run full baseline and save final metrics/time.
  - Config: hidden1=256, hidden2=128, epochs=80, lr=0.01, seed=42
  - Result: test Q3=62.74%, total_time=441s
  - Reference JSON: results/serial/serial_t1_s42_20260308_124640.json

Done criteria:
- Serial version is stable, reproducible, and used as accuracy reference for all parallel versions.

---

## Phase 5: OpenMP Version
### 5.1 Parallel strategy
- [x] Parallelize sample or mini-batch gradient computation.
- [x] Use per-thread local gradient buffers.
- [x] Reduce local gradients into global gradient.
- [x] Apply single-thread model update after reduction.

### 5.2 Performance-sensitive implementation
- [x] Keep parallel region persistent when possible (avoid frequent fork/join).
- [x] Document OpenMP schedule and chunk policy (`schedule(static)` on inner k-loop).
- [x] Add optional pinning settings in experiment notes:
  - [x] `OMP_PROC_BIND=true`
  - [x] `OMP_PLACES=cores`

### 5.3 Correctness validation
- [x] Compare OpenMP Q3 vs serial (same seed and hyperparameters).
- [x] Ensure difference stays within small tolerance.
  - Verified: loss/val_q3/test_q3 identical at epoch 1-2 (seed=42, 256/128 config)
  - Speedup: ~3.3× with 4 threads (~4.9s/epoch serial → ~1.5s/epoch OMP)
- [x] Validate no race conditions with thread sanitizing checks (if available).

Done criteria:
- OpenMP code produces matching accuracy and measurable speedup.

---

## Phase 6: POSIX Threads Version
### 6.1 Architecture (master/worker)
- [ ] Implement one master thread for update control.
- [ ] Implement worker thread pool created once at startup.
- [ ] Partition batch among workers.
- [ ] Aggregate worker gradients at synchronization points.

### 6.2 Synchronization and memory safety
- [ ] Use `pthread_mutex` + `pthread_cond` or barrier.
- [ ] Keep per-thread buffers isolated to avoid false sharing.
- [ ] Add clean shutdown via exit flag and thread joins.

### 6.3 Correctness and performance checks
- [ ] Match serial Q3 within tolerance.
- [ ] Measure time vs thread counts (1,2,4,8,16).
- [ ] Compare pthread and OpenMP overhead behavior.

Done criteria:
- Pthreads variant is correct, stable, and benchmarkable against OpenMP.

---

## Phase 7: MPI Version (Distributed Across 3 Laptops)
### 7.1 Cluster setup
- [ ] Install same OpenMPI version on all laptops.
- [ ] Configure passwordless SSH from launcher node.
- [ ] Create and test `hosts.txt`.
- [ ] Run MPI hello-world on 3 nodes.

### 7.2 Data-parallel SGD
- [ ] Shard dataset by rank.
- [ ] Compute local gradients per rank.
- [ ] Use `MPI_Allreduce` to sum gradients.
- [ ] Update weights consistently on all ranks.
- [ ] Use barrier before timing blocks.

### 7.3 Validation and timing
- [ ] Compare MPI Q3 to serial reference.
- [ ] Benchmark ranks: 1, 2, 3.
- [ ] Log max per-rank epoch time (not just rank 0 local time).

Done criteria:
- MPI variant scales across laptops with verified accuracy consistency.

---

## Phase 8: Hybrid MPI + OpenMP
### 8.1 Implementation
- [ ] Use MPI across machines, OpenMP within each rank.
- [ ] Keep one rank per laptop as initial configuration.
- [ ] Sweep thread counts per rank carefully.

### 8.2 Oversubscription control
- [ ] Ensure `ranks x threads <= 16` per laptop.
- [ ] Validate CPU utilization and thread placement behavior.

### 8.3 Validation
- [ ] Confirm hybrid Q3 matches serial baseline.
- [ ] Benchmark requested matrix:
  - [ ] `1x1`, `1x8`, `1x16`, `2x8`, `2x16`, `3x8`, `3x16`.

Done criteria:
- Hybrid variant produces both correct results and an interpretable performance profile.

---

## Phase 9: CUDA Version (Remote GPU)
### 9.1 Environment setup
- [ ] Choose execution platform (lab machine/cloud/Colab).
- [ ] Record GPU model, CUDA version, driver, and cuBLAS version.

### 9.2 GPU implementation
- [ ] Port dense layer operations to CUDA (custom kernels or cuBLAS).
- [ ] Handle data transfer efficiently (minimize host-device copies).
- [ ] Implement/validate softmax + loss path.

### 9.3 Accuracy and timing
- [ ] Verify CUDA Q3 on fixed subset vs serial.
- [ ] Benchmark batch sizes: 32, 64, 128, 256.
- [ ] Report GPU vs CPU timing on same workload.

Done criteria:
- CUDA run is reproducible remotely and included in final comparison table.

---

## Phase 10: Experiment Automation And Results Collection
### 10.1 Automation scripts
- [ ] Create scripts to run all experiment sweeps without manual edits.
- [ ] Standardize output naming (`variant_threads_seed_timestamp.json`).
- [ ] Save raw timing logs and parsed summary CSV files.

### 10.2 Required experiment matrix
- [ ] Accuracy table for:
  - [ ] Serial
  - [ ] OpenMP
  - [ ] Pthreads
  - [ ] MPI
  - [ ] Hybrid
  - [ ] CUDA
- [ ] Timing sweeps:
  - [ ] OpenMP threads: 1,2,4,8,16
  - [ ] Pthreads threads: 1,2,4,8,16
  - [ ] MPI ranks: 1,2,3
  - [ ] Hybrid rank-thread combinations
  - [ ] CUDA batch sizes: 32,64,128,256

### 10.3 Statistical reliability
- [ ] Repeat each timing run at least 3 times.
- [ ] Report mean and standard deviation.
- [ ] Exclude setup/preprocessing time from training benchmark.

Done criteria:
- Complete raw data exists for every table/plot needed in the report.

---

## Phase 11: Plotting And Analysis
### 11.1 Required visuals
- [ ] Time vs threads (OpenMP, pthreads).
- [ ] Speedup vs threads (OpenMP, pthreads).
- [ ] Time vs ranks (MPI).
- [ ] Hybrid performance table/heatmap.
- [ ] GPU vs CPU time comparison.

### 11.2 Required analysis points
- [ ] Accuracy parity with serial baseline.
- [ ] Where scaling improves and where it plateaus.
- [ ] Overheads: synchronization, communication, thread management.
- [ ] Why OpenMP and pthread results differ in your implementation.
- [ ] Practical limits from hardware constraints (3 laptops, no local GPU).

Done criteria:
- Every claim in report text is backed by a saved figure or table.

---

## Phase 12: Final Report Writing
### 12.1 Report structure (write in this order)
- [ ] Introduction and motivation.
- [ ] Dataset and preprocessing details.
- [ ] Serial model and Q3 definition.
- [ ] Parallel designs (OpenMP, pthreads, MPI, hybrid, CUDA).
- [ ] Experimental setup and fairness controls.
- [ ] Results: accuracy table + timing/speedup plots.
- [ ] Discussion of bottlenecks and tradeoffs.
- [ ] Conclusion and future improvements.

### 12.2 Diagrams required for high marks
- [ ] OpenMP data-parallel gradient + reduction flow.
- [ ] Pthreads master/worker synchronization flow.
- [ ] MPI/Hybrid diagram with ranks, threads, and `Allreduce`.

### 12.3 Reproducibility appendix
- [ ] Compile commands and flags.
- [ ] Runtime commands for each executable.
- [ ] Machine specs and software versions.
- [ ] Data source links and citation list.

Done criteria:
- Report is internally consistent with code, logs, and plots.

---

## Phase 13: Submission Readiness Checklist
- [ ] All required executables build from clean checkout.
- [ ] All required deliverables are present:
  - [ ] Serial code
  - [ ] Shared-memory code
  - [ ] Distributed-memory code
  - [ ] Hybrid code
  - [ ] Analysis report PDF
- [ ] Final quick rerun confirms no broken paths.
- [ ] Package submission artifact (zip/tar) with clear folder structure.

Done criteria:
- You can submit immediately without last-minute manual fixes.

---

## Minimum Viable Execution Order (If Time Is Short)
- [ ] Phase 0
- [ ] Phase 1
- [ ] Phase 2
- [ ] Phase 3
- [ ] Phase 4
- [ ] Phase 5
- [ ] Phase 6
- [ ] Phase 7
- [ ] Phase 11 (basic plots)
- [ ] Phase 12 (report)

Optional if schedule allows:
- [ ] Phase 8 (hybrid tuning)
- [ ] Phase 9 (CUDA depth improvements)
