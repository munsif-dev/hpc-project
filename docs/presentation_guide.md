# Presentation Guide: 10-Minute Final Evaluation

Project title:

**Parallel Training of an MLP Neural Network for Protein Secondary Structure Classification**

Subtitle:

**A CB513 Evaluation Using Serial, OpenMP, POSIX Threads, MPI, Hybrid MPI+OpenMP, and CUDA**

Use this guide to build the final Canva presentation. Keep the deck visual and avoid putting full report paragraphs on slides. The talk should make one clear point:

> We trained the same MLP using different parallel programming models and measured how much training speed improved while accuracy stayed close to the serial baseline.

## Canva Setup

- Deck length: 10 slides.
- Target time: 10 minutes.
- Style: technical, clean, not crowded.
- Use one main visual per slide.
- Use the report colors where possible: blue, slate/gray, green, orange, purple, red.
- Font suggestion: heading 34-42 pt, body 20-26 pt, small labels 16-18 pt.
- Keep slide text short; explain details verbally.

## Slide Timing Overview

| Slide | Topic | Time |
|---|---:|---:|
| 1 | Title and contribution | 45 sec |
| 2 | Problem and Q3 task | 60 sec |
| 3 | Dataset and preprocessing | 60 sec |
| 4 | MLP model and serial training | 60 sec |
| 5 | Main HPC idea | 75 sec |
| 6 | Shared-memory CPU: OpenMP and Pthreads | 75 sec |
| 7 | Distributed and hybrid CPU: MPI and MPI+OpenMP | 75 sec |
| 8 | GPU: CUDA training workflow | 75 sec |
| 9 | Results: time, speedup, accuracy | 120 sec |
| 10 | Final takeaway and Q&A bridge | 75 sec |

Total: about 10 minutes.

---

## Slide 1: Title and Contribution

### Canva Layout

Use a centered title layout. Add a small row of labels at the bottom:

`Serial | OpenMP | Pthreads | MPI | Hybrid | CUDA`

### On-Slide Text

**Parallel Training of an MLP Neural Network**

Protein Secondary Structure Classification on CB513

Group 36

EG/2021/4684 MUNSIF M.F.A.  
EG/2021/4685 MUTHUKUMARI H.M.S.  
EG/2021/4686 NADHA M.J.

### Speaker Notes

This project is about high-performance training of a neural network. The machine learning task is protein secondary structure classification, but the main contribution is the parallelisation of the MLP training loop. We implemented and compared serial, OpenMP, POSIX Threads, MPI, Hybrid MPI+OpenMP, and CUDA versions under the same dataset, model, and hyperparameters.

### Transition

Before discussing the parallel programming part, I will first explain the classification task.

---

## Slide 2: Problem and Q3 Classification Task

### Canva Layout

Left side: simple protein sequence example.  
Right side: three class labels H, E, C.

### On-Slide Text

**Problem**

Classify each amino acid residue into one of three secondary-structure classes:

- H: alpha helix
- E: beta strand
- C: coil

**Metric**

Q3 = correctly classified residues / total residues

### Speaker Notes

Proteins are chains of amino acids. Each position in the chain can locally form a helix, a strand, or a coil. Our model receives sequence-window features and predicts one of these three labels for the center residue. The evaluation metric is Q3 accuracy, which is simply the percentage of residues classified correctly across the three classes.

Important wording: do not say the project is mainly about proposing a new biology model. Say it is about accelerating the training of a baseline MLP for this classification task.

### Transition

To train the model, we first convert the CB513 protein dataset into numeric features.

---

## Slide 3: Dataset and Preprocessing

### Canva Layout

Use the preprocessing diagram as the main visual.

### Figure

`docs/diagrams/png/01_preprocessing_pipeline.png`

### On-Slide Text

**CB513 Dataset**

- 513 protein domains
- Train / validation / test split: 70 / 15 / 15
- Input feature size: 260
- Output classes: H, E, C

**Feature extraction**

13-residue sliding window x 20 amino-acid one-hot vector = 260 floats

### Speaker Notes

We used the CB513 benchmark. The raw labels are collapsed from 8 DSSP states into the 3 Q3 classes. For each residue, we take a local window of 13 residues: 6 on the left, the center residue, and 6 on the right. Each amino acid is one-hot encoded over 20 amino-acid types, so each sample becomes 13 times 20, which is 260 input features.

This preprocessing is deterministic and shared by all implementations, so the comparison is fair.

### Transition

After preprocessing, every implementation trains the same MLP model.

---

## Slide 4: MLP Model and Serial Training Loop

### Canva Layout

Use the serial training-loop diagram on one side. Put the model shape on the other side.

### Figure

`docs/diagrams/png/02_serial_training_loop.png`

### On-Slide Text

**Model**

260 -> 256 -> 128 -> 3

**Training**

- Forward pass
- Cross-entropy loss
- Backpropagation
- SGD weight update
- Q3 evaluation

### Speaker Notes

The model is a two-hidden-layer MLP. The input has 260 features. The hidden layers have 256 and 128 neurons, and the output layer has 3 neurons for H, E, and C. In the serial version, each mini-batch runs forward propagation, computes loss, runs backpropagation, and then updates the weights using SGD.

This serial training loop is the baseline. Every parallel version preserves the same mathematical training logic.

### Transition

The main HPC question is: which part of this training loop can we safely parallelise?

---

## Slide 5: Main HPC Idea: Mini-Batch Data Parallelism

### Canva Layout

Use a large flow block:

`Mini-batch -> split samples -> private gradients -> reduce gradients -> one SGD update`

Add the pseudocode in a small box.

### On-Slide Text

**Key parallel rule**

Weights are read-only during parallel computation.

Each worker computes private gradients.

Gradients are reduced once.

Then one SGD update changes the weights.

### Pseudocode for Slide

```text
for each epoch:
    shuffle training samples
    for each mini-batch:
        parallel workers:
            compute forward + backward on batch slice
            store private gradients

        global_gradient = reduce_sum(private_gradients)
        weights = weights - learning_rate * global_gradient
```

### Speaker Notes

The main parallel programming concept is mini-batch data parallelism. During a mini-batch, workers process different samples. The important design choice is that weights are not updated inside the parallel region. Workers only read the current weights and write to private gradient buffers. After all workers finish, we reduce the gradients and apply one SGD update. This avoids locking on the weight matrices and keeps the training logic close to the serial baseline.

This pattern is the common idea behind OpenMP, Pthreads, MPI, Hybrid, and CUDA. The difference is how workers are created and how gradients are reduced.

### Transition

First, I will explain the shared-memory CPU versions: OpenMP and Pthreads.

---

## Slide 6: Shared-Memory CPU Parallelism: OpenMP and Pthreads

### Canva Layout

Use two columns:

- Left: OpenMP workflow
- Right: Pthreads workflow

### Figures

`docs/diagrams/png/03_openmp_shared_memory.png`  
`docs/diagrams/png/04_pthreads_master_worker.png`

### On-Slide Text

**Shared-memory idea**

One process, multiple CPU threads, shared model weights.

**OpenMP**

- Compiler/runtime manages thread team
- Static mini-batch split
- Thread-private gradients

**Pthreads**

- Manual persistent worker pool
- Start and done barriers
- Master reduces gradients

### Speaker Notes

OpenMP and Pthreads both use shared memory. The model weights are stored once, and multiple threads can read them. In OpenMP, the runtime creates and manages the thread team, so the implementation is shorter. In Pthreads, we manually build a persistent worker pool. The master thread publishes the mini-batch, workers compute private gradients, and barriers synchronize the start and finish of each mini-batch.

The key point is that both versions avoid updating weights from multiple threads at the same time. Only after the private gradients are combined does the master perform the SGD update.

### Transition

Shared-memory works inside one machine. For distributed memory, we use MPI.

---

## Slide 7: Distributed and Hybrid CPU Parallelism: MPI and MPI+OpenMP

### Canva Layout

Use two rows:

- Top: MPI distributed-memory workflow
- Bottom: Hybrid MPI+OpenMP workflow

### Figures

`docs/diagrams/png/05_mpi_distributed_memory.png`  
`docs/diagrams/png/06_hybrid_mpi_openmp.png`

### On-Slide Text

**MPI distributed memory**

- Each rank has its own model copy
- Each rank computes local gradients
- `MPI_Allreduce(SUM)` combines gradients
- Every rank applies the same SGD update

**Hybrid MPI+OpenMP**

- MPI splits work across ranks
- OpenMP splits each rank's slice across threads
- Intra-rank reduce, then MPI Allreduce

### Speaker Notes

MPI is different from OpenMP and Pthreads because each rank has a separate address space. Each rank loads the same data split and starts from the same model state. For every mini-batch, ranks process different slices and compute rank-local gradients. Then we pack the gradient arrays and use `MPI_Allreduce(SUM)` so that every rank receives the same global gradient. After that, each rank applies the same SGD update, keeping the model copies synchronized.

The hybrid version combines both ideas. MPI handles the outer distributed split, and OpenMP handles parallelism inside each rank. This reduces per-rank sample work while avoiding too many MPI ranks.

### Transition

The final implementation moves the training loop to the GPU using CUDA.

---

## Slide 8: GPU Parallelism: CUDA Training Workflow

### Canva Layout

Use the CUDA workflow diagram as the main visual. Add a small side note with "GPU resident data".

### Figure

`docs/diagrams/png/07_cuda_gpu_workflow.png`

### On-Slide Text

**CUDA implementation**

- Training tensors and weights stay on GPU
- Mini-batch gather is parallel
- Matrix operations use cuBLAS GEMM
- Custom kernels handle ReLU, softmax, and loss
- SGD updates happen on GPU

### Speaker Notes

The CUDA version keeps the training data and model weights on the GPU. For each mini-batch, the CPU mainly shuffles indices and launches kernels. The GPU gathers the mini-batch, runs matrix multiplications using cuBLAS, applies ReLU and softmax kernels, computes gradients, and updates weights in GPU memory.

This is the most aggressive parallel version because the dense matrix operations and per-sample operations are highly suitable for the GPU.

### Transition

Now I will compare the performance and accuracy of these implementations.

---

## Slide 9: Results: Time, Speedup, and Accuracy

### Canva Layout

Use four result tiles. Do not overload the slide with all graphs at full size. Pick the most important visuals:

1. CUDA timing or total speedup callout
2. Shared-memory speedup
3. MPI/Hybrid timing
4. Accuracy parity

### Figures

Use these figures:

`plots/speedup_vs_threads.png`  
`plots/time_vs_ranks_mpi.png`  
`plots/hybrid_heatmap.png`  
`plots/cuda_time_vs_batch.png`  
`plots/accuracy_by_variant.png`

If the slide becomes crowded, use only:

`plots/speedup_vs_threads.png`  
`plots/cuda_time_vs_batch.png`  
`plots/accuracy_by_variant.png`

### On-Slide Text

**Performance**

- OpenMP best: 5.34x speedup at 8 threads
- MPI best useful point: 8 ranks, then communication overhead increases
- Hybrid best timing: 8 ranks x 2 threads
- CUDA 80-epoch run: 6.3 s vs serial 236.8 s, about 37.8x faster

**Accuracy**

- Serial test Q3: 59.28%
- All parallel variants stay close to serial

### Speaker Notes

The results show the main HPC tradeoff. OpenMP improves up to 8 threads, but after that synchronization and memory effects reduce benefit. MPI improves up to a useful point, but too many ranks increase collective communication overhead. Hybrid gives the best CPU timing because it balances rank-level and thread-level parallelism. CUDA is the fastest by a large margin because the matrix operations and parallel kernels run efficiently on the GPU.

Accuracy is important because a faster implementation is not useful if it changes the training result. The final Q3 values are close to the serial baseline, and the convergence curves in the report show that training follows nearly the same learning path.

Recommended spoken numbers:

- Serial 80 epochs: 236.8 s
- CUDA 80 epochs: 6.3 s
- CUDA speedup: about 37.8x
- Serial test Q3: 59.28%
- CUDA test Q3: 59.37%

### Transition

Finally, I will summarize what we learned from the implementations.

---

## Slide 10: Final Takeaway and Q&A Bridge

### Canva Layout

Use a clean three-column summary:

1. What we built
2. What scaled
3. What limited scaling

### On-Slide Text

**What we built**

Same MLP training loop implemented using six execution models.

**What scaled well**

Mini-batch data parallelism with private gradients and reduction.

**Main bottlenecks**

CPU synchronization, gradient reduction, MPI Allreduce, and small model size.

**Final takeaway**

Parallel programming significantly reduced training time while preserving Q3 accuracy.

### Speaker Notes

The main lesson is that mini-batch data parallelism is a good match for MLP training. The same training logic can be expressed using shared-memory threads, distributed-memory ranks, hybrid parallelism, and GPU kernels. The most important correctness rule is to keep weights read-only during the parallel computation and update them only after reducing gradients.

The CPU versions show useful speedups but eventually hit synchronization and communication limits. The CUDA version gives the largest improvement because dense matrix operations map well to GPU parallelism. Overall, the project demonstrates how parallel programming concepts can accelerate neural network training while preserving the model's accuracy behavior.

End with:

"That is the main contribution of our project: not a new biological predictor, but a high-performance comparison of parallel training strategies for the same MLP model."

---

## Backup Q&A Notes

Use these only if evaluators ask.

### Why Q3 and not RMSE?

This is a classification task with three labels: H, E, and C. Q3 is the standard percentage accuracy for three-state protein secondary structure classification. RMSE is more suitable for regression, not this task.

### Why does CUDA accuracy differ slightly?

CUDA and cuBLAS can reorder floating-point reductions. The algorithm is the same, but floating-point addition is not perfectly associative, so the exact weight trajectory can differ slightly.

### Why does speedup reduce after many CPU workers?

The MLP is small, so per-sample computation is limited. At higher thread or rank counts, synchronization, memory traffic, and gradient reduction consume a larger fraction of runtime.

### Why use private gradients?

Private gradients avoid many workers writing to the same gradient buffers at the same time. This avoids locks and race conditions. The private gradients are summed once before the SGD update.

### Why Hybrid MPI+OpenMP?

Hybrid parallelism reduces the number of MPI processes while still using multiple CPU cores inside each process. This can reduce MPI communication pressure and improve CPU utilization.

### What is the biggest result?

CUDA completed the 80-epoch reference run in about 6.3 seconds, compared with 236.8 seconds for serial training, while keeping test Q3 close to the serial baseline.

---

## Canva Asset Checklist

Copy or upload these images into Canva:

- `docs/diagrams/png/01_preprocessing_pipeline.png`
- `docs/diagrams/png/02_serial_training_loop.png`
- `docs/diagrams/png/03_openmp_shared_memory.png`
- `docs/diagrams/png/04_pthreads_master_worker.png`
- `docs/diagrams/png/05_mpi_distributed_memory.png`
- `docs/diagrams/png/06_hybrid_mpi_openmp.png`
- `docs/diagrams/png/07_cuda_gpu_workflow.png`
- `plots/speedup_vs_threads.png`
- `plots/time_vs_ranks_mpi.png`
- `plots/hybrid_heatmap.png`
- `plots/cuda_time_vs_batch.png`
- `plots/accuracy_by_variant.png`
- `plots/convergence_curves.png`

For a simpler 10-minute deck, use only the boldest visuals:

- `01_preprocessing_pipeline.png`
- `02_serial_training_loop.png`
- `03_openmp_shared_memory.png`
- `05_mpi_distributed_memory.png`
- `07_cuda_gpu_workflow.png`
- `speedup_vs_threads.png`
- `cuda_time_vs_batch.png`
- `accuracy_by_variant.png`

---

## Final Delivery Checklist

Before presenting:

- Keep the deck to 10 slides.
- Practice once with a timer.
- Do not explain every line of the diagrams; explain the flow.
- Use the phrase "private gradients, reduction, one SGD update" several times.
- Keep code/pseudocode to one slide only.
- Do not over-focus on biology; the evaluation is for HPC.
- End with the speedup plus accuracy-preservation message.
