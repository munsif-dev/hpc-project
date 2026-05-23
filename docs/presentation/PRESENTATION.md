# PRESENTATION — 10-Slide Build Script

**Project:** Parallel Training of an MLP Neural Network for Protein Secondary
Structure Classification (CB513)

**Variants compared:** Serial · OpenMP · POSIX Threads · MPI · Hybrid MPI+OpenMP · CUDA

**Group 36** — EG/2021/4684 MUNSIF M.F.A. · EG/2021/4685 MUTHUKUMARI H.M.S. · EG/2021/4686 NADHA M.J.

**One-sentence thesis:** *We trained the same MLP using six execution models
and measured how much training time dropped while accuracy stayed close to
the serial baseline.*

**Total target time:** 10 minutes · 10 slides · one idea + one visual per slide.

| # | Slide                              | Sec |
|---|------------------------------------|----:|
| 1 | Title & contribution               |  45 |
| 2 | Problem: Q3 classification         |  60 |
| 3 | Data → features → model            |  60 |
| 4 | The parallel rule                  |  75 |
| 5 | Shared memory: OpenMP + Pthreads   |  75 |
| 6 | Shared-memory result               |  60 |
| 7 | Distributed memory: MPI + Hybrid   |  75 |
| 8 | Distributed result                 |  60 |
| 9 | GPU: CUDA workflow + result        |  90 |
| 10 | Accuracy parity + takeaway        |  80 |

---

## Slide 1 — Title & Contribution

**Visual:** none (clean title layout).
Bottom strip of labels: `Serial | OpenMP | Pthreads | MPI | Hybrid | CUDA`.

**On-slide text:**
- Parallel Training of an MLP Neural Network
- Protein Secondary Structure Classification on CB513
- Group 36 — EG/2021/4684, 4685, 4686

**Speaker note:**
This project is about high-performance training of a small neural network.
The ML task is protein secondary structure classification, but the
contribution is the parallelisation of the MLP training loop. We built six
implementations of the same training algorithm and compared them under
identical dataset, model, and hyperparameters.

---

## Slide 2 — Problem: Q3 Classification

**Visual:** small H / E / C illustration on the right (no PNG required —
Canva text + colour blocks).

**On-slide text:**
- Per residue, predict one of three local structures
- H = alpha helix · E = beta strand · C = coil
- Metric: Q3 = correctly classified residues ÷ total

**Speaker note:**
Proteins are chains of amino acids. Each position locally forms a helix, a
strand, or a coil. Our model receives a sequence window for one residue and
outputs one of these three labels. Q3 is just the percentage of residues
classified correctly across the three classes — the standard metric for
this task.

---

## Slide 3 — Data → Features → Model

**Visual:** `docs/diagrams/png/01_preprocessing_pipeline.png`

**On-slide text:**
- CB513 dataset — 513 proteins, 70 / 15 / 15 split (deterministic, seed = 42)
- Sliding window = 13 residues, one-hot over 20 amino acids → **260 floats** per sample
- Output classes: H, E, C
- Model: MLP **260 → 256 → 128 → 3** (≈100 k parameters)

**Speaker note:**
For each residue we extract a 13-residue window — 6 left, the centre, 6
right. Each amino acid is one-hot encoded over 20 types, so every sample is
13 × 20 = 260 floats. The same preprocessing and the same MLP shape are
used by every variant, which is what makes the timing comparison fair.

---

## Slide 4 — The Parallel Rule

**Visual:** `docs/diagrams/png/02_serial_training_loop.png` on the left;
pseudocode block on the right.

**On-slide text:**
- Weights are **read-only** during the parallel computation
- Each worker writes to its **private gradient buffer**
- After workers finish: **reduce gradients → one SGD update**

```text
for each epoch:
  shuffle samples
  for each mini-batch:
    parallel workers: forward + backward on batch slice
                      → private gradient buffer
    global_grad = reduce_sum(private_grads)
    weights -= (lr / batch) * global_grad
```

**Speaker note:**
This is the common pattern behind every variant. During a mini-batch,
workers process different samples, but no worker writes to the weights.
Each one writes to a private gradient buffer. After the parallel region we
sum the buffers and apply one SGD update. This avoids locks on the weight
matrices and keeps the training maths identical to the serial baseline.
What changes across variants is *how* the workers are created and *how*
the gradients are reduced.

---

## Slide 5 — Shared Memory: OpenMP + Pthreads

**Visual:** `docs/diagrams/png/03_openmp_shared_memory.png`

**On-slide text:**
- One process, multiple CPU threads, **shared weights**
- **OpenMP** — compiler-managed team, `#pragma omp for schedule(static)`, per-thread private gradients
- **Pthreads** — manual persistent worker pool, start/done barriers, master reduces
- Best OpenMP: **2.49× at 16 threads**
- Best Pthreads: **2.72× at 16 threads**

**Speaker note:**
Both versions use shared memory: the weights sit in one process and every
thread reads them. OpenMP uses the runtime to create and manage the thread
team — the implementation is short. Pthreads spawns a persistent worker
pool once and reuses it; the master publishes the batch, workers compute
into thread-local gradient buffers, barriers synchronise start and finish.
Pthreads is slightly faster because there is no thread-team teardown
between batches.

---

## Slide 6 — Shared-Memory Result

**Visual:** `plots/time_vs_threads.png`

**On-slide text:**
- Speedup grows with threads but **flattens past 8–16**
- Cause: sync overhead + small MLP → little work per sample
- Both OpenMP and Pthreads track each other closely

**Speaker note:**
The curve shows total training time as we add threads. Speedup improves
quickly up to 8 threads and starts to flatten by 16. After that, the model
is small enough that per-sample work no longer hides the cost of
synchronisation and memory traffic. This is the first HPC trade-off in the
talk: parallel work has diminishing returns when the per-worker task is
cheap.

---

## Slide 7 — Distributed Memory: MPI + Hybrid

**Visual:** `docs/diagrams/png/05_mpi_distributed_memory.png`

**On-slide text:**
- **MPI** — each rank holds its own model copy; identical seed → synchronised state
- One `MPI_Allreduce(SUM)` per batch reduces gradients across ranks
- **Hybrid (MPI + OpenMP)** — MPI splits across ranks, OpenMP splits inside each rank, then Allreduce
- Best MPI: **2.76× at 4 ranks**
- Best Hybrid: **3.04× at 4 ranks × 4 threads** *(best CPU result)*

**Speaker note:**
Each MPI rank has its own address space, so we cannot share weights through
memory. Every rank starts from the same model state and processes a
different slice of every mini-batch. Then `MPI_Allreduce` sums the
gradients across all ranks; every rank receives the same global gradient
and applies the same SGD update, which keeps the models in lock-step. The
hybrid version combines both ideas: MPI handles the outer split between
ranks, OpenMP handles the inner split between threads inside a rank. This
keeps the rank count low and reduces Allreduce pressure.

---

## Slide 8 — Distributed Result

**Visual:** `plots/time_vs_ranks_mpi.png`

**On-slide text:**
- MPI improves up to ~4 ranks
- Past ~8 ranks: **Allreduce communication dominates**
- Hybrid wins on CPU because it balances ranks and threads

**Speaker note:**
The MPI curve shows the second HPC trade-off: when the model is small,
collective communication starts to cost as much as the work it parallelises.
We get useful scaling up to about 4 ranks; beyond 8 the Allreduce overhead
overtakes the gain from extra workers. The hybrid version sidesteps this by
keeping the rank count low and adding threads inside each rank — which is
why it produces the best CPU timing in the project.

---

## Slide 9 — GPU: CUDA Workflow + Result

**Visual (top half):** `docs/diagrams/png/07_cuda_gpu_workflow.png`
**Visual (bottom half):** `plots/cuda_time_vs_batch.png`

**On-slide text:**
- Training tensors and weights stay on the **GPU**
- Mini-batch gather kernel · cuBLAS GEMM · custom ReLU / softmax / loss kernels
- Backward + SGD update **fused** via cuBLAS with `alpha = -lr / batch`
- **6.27 s vs 236.81 s → 37.8× speedup** *(80 epochs, batch = 64)*

**Speaker note:**
The CUDA version keeps the data and the weights resident on the GPU. The
CPU mostly shuffles indices and launches kernels. Dense matrix
multiplications use cuBLAS, and we wrote custom kernels for ReLU, softmax,
and the loss. The backward pass and the SGD update are fused — we pass a
negative scaled alpha into the cuBLAS call so the gradient is written
straight into the weights. This is the most aggressive parallel version
and it gives the largest speedup by a wide margin: roughly 38× faster than
serial for the full 80-epoch run.

---

## Slide 10 — Accuracy Parity + Takeaway

**Visual:** `plots/accuracy_by_variant.png`

**On-slide text:**

| Variant   | Time (s) | Speedup | Q3 (%) |
|-----------|---------:|--------:|-------:|
| Serial    |   236.81 |   1.00× |  59.28 |
| OpenMP    |    94.97 |   2.49× |  59.42 |
| Pthreads  |    87.17 |   2.72× |  59.23 |
| MPI       |    85.77 |   2.76× |  59.23 |
| Hybrid    |    77.97 |   3.04× |  59.30 |
| CUDA      |     6.27 |  37.82× |  59.37 |

- Every variant within **±0.15 %** of serial Q3
- Same mini-batch SGD logic — six execution models, one learning curve

**Speaker note:**
The headline is that parallelism preserved accuracy. Every variant lands
within roughly 0.15 % of the serial Q3 of 59.28 %, so faster training did
not change the model the network learns. The biggest jump is CUDA at
nearly 38× — dense matrix operations map cleanly to the GPU. The lesson
of the project is not a new biological predictor; it is a clean comparison
of parallel training strategies for the same MLP. The correctness rule that
made every variant land at the same accuracy is the same in every case:
keep weights read-only during the parallel region and update them only
after the gradients are reduced.

End the talk with: *"Same MLP, six execution models, 38× faster — and the
model still learns the same thing."*

---

## Backup Q&A (not on slides)

- **Why does the speedup flatten on CPU?** The MLP is small, so per-sample
  work is cheap. Past a point, synchronisation and gradient reduction take
  longer than the work they parallelise.
- **Why is CUDA Q3 slightly different?** cuBLAS reorders floating-point
  reductions. The algorithm is unchanged but FP addition is not
  associative, so the trajectory differs by a fraction of a percent.
- **Why Allreduce and not a parameter server?** Allreduce is the standard
  collective for synchronous SGD; every rank gets the same global gradient
  in one call with no central bottleneck.
- **Why is Hybrid the best CPU number?** Fewer MPI ranks → less
  communication pressure; OpenMP soaks up the remaining cores inside each
  rank.

## Build assets

- Diagrams: `docs/diagrams/png/01_preprocessing_pipeline.png`,
  `02_serial_training_loop.png`, `03_openmp_shared_memory.png`,
  `05_mpi_distributed_memory.png`, `07_cuda_gpu_workflow.png`
- Plots: `plots/time_vs_threads.png`, `plots/time_vs_ranks_mpi.png`,
  `plots/cuda_time_vs_batch.png`, `plots/accuracy_by_variant.png`
- Numbers cross-checked against `plots/timing_summary.csv` and
  `plots/accuracy_table.csv`.
- Claude Design upload bundle: `docs/presentation/claude-design/`
  (contains `brief.md` + 9 PNGs with `slideNN_` filenames).
