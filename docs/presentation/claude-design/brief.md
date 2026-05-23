# Claude Design Brief — 10-Minute HPC Presentation

**Title:** Parallel Training of an MLP for Protein Secondary Structure Classification (CB513)

**Subtitle:** A comparison of Serial · OpenMP · Pthreads · MPI · Hybrid · CUDA

**Audience:** Final HPC course evaluation (technical reviewers).

**Single thesis (state on slide 1 and slide 10):** Same MLP, six execution
models. We measured how much training time dropped while accuracy stayed close
to the serial baseline.

---

## Design System

Apply these rules across every slide.

- **Deck length:** exactly 10 slides.
- **One idea per slide.** One headline, one visual, ≤ 6 short lines of
  body text. No paragraphs on slides.
- **Palette (used consistently):**
  - Primary: deep blue (`#1F3A93`) — headers and structural elements.
  - Secondary: slate gray (`#3B4252`) — body text.
  - Accents (used sparingly, one per variant where helpful):
    - OpenMP — green (`#2ECC71`)
    - Pthreads — orange (`#E67E22`)
    - MPI — purple (`#8E44AD`)
    - Hybrid — teal (`#16A085`)
    - CUDA — red (`#C0392B`)
- **Typography:** Heading 34–42 pt · Body 20–26 pt · Caption / labels 16–18 pt.
- **Whitespace > density.** Leave breathing room around every figure.
- **Numbers are protagonists.** When a slide has a headline number (e.g.
  `37.8×`, `±0.15 %`), set it large enough to read from the back row.
- **Figures occupy ≥ 50 % of the slide** when present. Crop figures to
  remove dead white margins before placing.
- **Footer (every slide except 1 and 10):** small right-aligned label
  `Group 36 · CB513 · MLP 260-256-128-3`.

---

## Slide 1 — Title & Contribution

**Layout:** centered title block; small variant strip across the bottom.

**On-slide text:**
- *(big)* Parallel Training of an MLP Neural Network
- *(medium)* Protein Secondary Structure Classification on CB513
- *(small)* Group 36 — EG/2021/4684 · 4685 · 4686
- *(bottom strip)* `Serial · OpenMP · Pthreads · MPI · Hybrid · CUDA`

**Visual:** none (clean title).

**Speaker note:** This project is high-performance training of a small MLP.
The ML task is protein secondary structure classification, but the
contribution is the parallel training loop. Six implementations, same data
and hyperparameters, head-to-head.

---

## Slide 2 — Problem: Q3 Classification

**Layout:** left half — short sequence example; right half — three labelled
class blocks (H, E, C).

**On-slide text:**
- Predict one label per residue: **H · E · C**
- H = alpha helix · E = beta strand · C = coil
- Metric: **Q3** = correctly classified residues ÷ total

**Visual:** none required. A simple Canva illustration: one row of letters
with coloured tags above (H green, E purple, C gray).

**Speaker note:** Each residue locally takes one of three shapes. The model
sees a window around a residue and predicts its label. Q3 is the percentage
of residues classified correctly across all three classes.

---

## Slide 3 — Data → Features → Model

**Layout:** image fills 60 % of the slide; text bullets on the right.

**Image:** `slide03_data_model.png`

**On-slide text:**
- CB513 — 513 proteins · 70 / 15 / 15 split (seed = 42, deterministic)
- Sliding window = 13 residues, one-hot across 20 amino acids → **260 floats**
- Output classes: H, E, C
- Model: MLP **260 → 256 → 128 → 3** (≈ 100 k parameters)

**Speaker note:** For each residue we extract a 13-residue window — 6 left,
the centre, 6 right. Each amino acid is one-hot encoded over 20 types, so
every sample is 260 floats. The same preprocessing and the same model shape
are used by every variant.

---

## Slide 4 — The Parallel Rule

**Layout:** image on the left (50 %); pseudocode block on the right (50 %).

**Image:** `slide04_serial_loop.png`

**On-slide text (right column):**
- Weights are **read-only** during the parallel region
- Each worker writes to a **private gradient buffer**
- After workers finish: **reduce gradients → one SGD update**

**Pseudocode (monospace, small font):**
```
for each epoch:
  shuffle samples
  for each mini-batch:
    parallel workers: forward + backward
                      → private gradient buffer
    global_grad = reduce_sum(private_grads)
    weights -= (lr / batch) * global_grad
```

**Speaker note:** This is the rule behind every variant. Workers process
different samples but never write to the weights — they only write to
private gradient buffers. After the parallel region we sum the buffers and
apply one SGD update. Locks are avoided and the training maths matches the
serial baseline. What changes across variants is *how* the workers are
created and *how* the gradients are reduced.

---

## Slide 5 — Shared Memory: OpenMP + Pthreads

**Layout:** image left (50 %); two-column text right (OpenMP / Pthreads).

**Image:** `slide05_openmp_shared.png`

**On-slide text:**
- One process, multiple CPU threads, **shared weights**
- **OpenMP (green):** `#pragma omp for schedule(static)` · per-thread private gradients · runtime-managed team
- **Pthreads (orange):** persistent worker pool · start/done barriers · master reduces
- Best OpenMP: **2.49× at 16 threads** · Best Pthreads: **2.72× at 16 threads**

**Speaker note:** Both versions use shared memory: the weights live in one
process and every thread can read them. OpenMP relies on the runtime to
create and manage the thread team. Pthreads creates a worker pool once and
reuses it for every mini-batch, which saves the teardown cost — that is why
Pthreads is slightly faster on this hardware.

---

## Slide 6 — Shared-Memory Result

**Layout:** plot fills 65 % of the slide; small caption block below.

**Image:** `slide06_time_vs_threads.png`

**On-slide text:**
- Speedup grows quickly to 8 threads, then **flattens**
- Small MLP → per-sample work no longer hides sync overhead
- OpenMP and Pthreads track each other closely

**Speaker note:** This is the first HPC trade-off in the talk: parallel work
has diminishing returns when the per-worker task is cheap. By 16 threads,
sync and memory traffic are eating the gain.

---

## Slide 7 — Distributed Memory: MPI + Hybrid

**Layout:** image top (60 % height); two-column text below (MPI / Hybrid).

**Image:** `slide07_mpi_distributed.png`

**On-slide text:**
- **MPI (purple):** each rank holds its own model · identical seed → synchronised state · `MPI_Allreduce(SUM)` once per batch
- **Hybrid (teal):** MPI splits across ranks · OpenMP splits inside each rank · two-level reduction
- Best MPI: **2.76× at 4 ranks**
- Best Hybrid: **3.04× at 4 ranks × 4 threads** — **best CPU result**

**Speaker note:** Each MPI rank has a separate address space, so weights
cannot be shared through memory. Every rank starts from the same state,
processes its own slice of the batch, and `MPI_Allreduce` sums the
gradients across all ranks. Hybrid keeps the rank count low and lets
OpenMP soak up the cores inside each rank.

---

## Slide 8 — Distributed Result

**Layout:** plot fills 65 % of the slide; caption block below.

**Image:** `slide08_time_vs_ranks_mpi.png`

**On-slide text:**
- Useful scaling up to ~4 ranks
- Past ~8 ranks: **Allreduce communication dominates**
- Hybrid wins on CPU by balancing ranks and threads

**Speaker note:** The second HPC trade-off: when the model is small,
collective communication starts to cost as much as the work it parallelises.
That is why Hybrid (few ranks × multiple threads) beats pure MPI at high
worker counts.

---

## Slide 9 — GPU: CUDA Workflow + Result

**Layout:** top half — workflow diagram; bottom half — batch-size scaling plot.

**Image (top):** `slide09_cuda_workflow.png`
**Image (bottom):** `slide09b_cuda_time_vs_batch.png`

**On-slide text (between the two images):**
- Training tensors and weights stay on the **GPU**
- Mini-batch gather kernel · cuBLAS GEMM · custom ReLU / softmax / loss kernels
- Backward + SGD update **fused** through cuBLAS (`alpha = -lr / batch`)
- *(huge accent number, red)* **37.8× faster** — 6.27 s vs 236.81 s, 80 epochs

**Speaker note:** The CUDA version keeps data and weights on the GPU; the
CPU mostly shuffles indices and launches kernels. Dense matrix
multiplications use cuBLAS, and we wrote custom kernels for ReLU, softmax,
and the loss. We fuse the backward pass and the SGD update by passing a
scaled negative alpha into the cuBLAS call, so the gradient is written
straight into the weights. This is the most aggressive parallel version
and it produces the largest speedup by a wide margin.

---

## Slide 10 — Accuracy Parity + Takeaway

**Layout:** plot on the left (45 %); small results table + takeaway on the right.

**Image:** `slide10_accuracy_by_variant.png`

**On-slide text — results table:**

| Variant   | Time (s) | Speedup | Q3 (%) |
|-----------|---------:|--------:|-------:|
| Serial    |   236.81 |   1.00× |  59.28 |
| OpenMP    |    94.97 |   2.49× |  59.42 |
| Pthreads  |    87.17 |   2.72× |  59.23 |
| MPI       |    85.77 |   2.76× |  59.23 |
| Hybrid    |    77.97 |   3.04× |  59.30 |
| CUDA      |     6.27 |  37.82× |  59.37 |

**Takeaway block (right column, big):**
- All variants within **±0.15 %** of the serial Q3
- Same mini-batch SGD logic, six execution models, **one learning curve**
- *"Same MLP, six execution models, 38× faster — and the model still learns the same thing."*

**Speaker note:** Parallelism preserved the model's learning behaviour.
Every variant lands within roughly 0.15 % of the serial Q3 of 59.28 %, so
the faster training does not change what the network learns. The headline
is CUDA at nearly 38× — dense matrix operations map cleanly to the GPU.
The take-home rule is: keep weights read-only during the parallel region,
reduce gradients, then apply one SGD update.

---

## Asset list (already attached / uploaded with this brief)

- `slide03_data_model.png`
- `slide04_serial_loop.png`
- `slide05_openmp_shared.png`
- `slide06_time_vs_threads.png`
- `slide07_mpi_distributed.png`
- `slide08_time_vs_ranks_mpi.png`
- `slide09_cuda_workflow.png`
- `slide09b_cuda_time_vs_batch.png`
- `slide10_accuracy_by_variant.png`

Slides 1 and 2 do not require uploaded images; build them from text and
simple Canva-style shapes per the layout notes above.
