# One-Page Project Proposal
## Module Details
EC7207: High Performance Computing

## Group Number
36

## Project Title
Parallel Protein Secondary Structure Prediction on CB513 using Serial, OpenMP, POSIX Threads, MPI, and Hybrid (MPI+OpenMP)

## Team Members
- EG/2021/4684 MUNSIF M.F.A.
- EG/2021/4685 MUTHUKUMARI H.M.S.
- EG/2021/4686 NADHA M.J.

## Project Summary
This project addresses protein secondary structure prediction as a residue-level 3-class classification task (H/E/C) on the CB513 benchmark. We will build one common training pipeline and implement it across Serial, OpenMP, POSIX Threads, MPI, and Hybrid (MPI+OpenMP) models to evaluate performance and scalability on available CPU resources.

The dataset contains 513 protein files and 84,119 residues. DSSP labels from the raw CB513 files will be mapped to 3-state labels using the paper-aligned rule (H/G/I to H, B/E to E, others to C). Samples will be generated using a sliding window of length 13 with one-hot encoding (20 x 13 = 260 features), ensuring a fair, shared data pipeline for all implementations.

## Problem Statement
Serial neural-network training on CB513 is too slow for repeated experiments and parameter sweeps. The key challenge is to parallelize training across shared-memory, distributed-memory, and hybrid settings while keeping prediction behavior consistent with the serial baseline. The project must therefore balance correctness (accuracy parity) and performance (timing/speedup) under real hardware constraints.

## Objectives
1. Build a correct serial baseline with reproducible preprocessing, training, and Q3 evaluation on CB513.
2. Implement all required HPC variants with the same model/data settings:
   - OpenMP
   - POSIX Threads
   - MPI
   - Hybrid MPI+OpenMP
3. Verify correctness by comparing each parallel variant against the serial baseline using Q3 accuracy.
4. Measure runtime, speedup, and scaling by varying threads/ranks/other parameters under fixed hyperparameters.
5. Prepare the final analysis report with:
   - parallelization diagram and implementation description,
   - accuracy comparison against serial,
   - timing comparison plots for serial vs parallel versions.

## Technical Plan
1. **Data and preprocessing**
   - Parse CB513 `.all` files (`RES`, `DSSP`).
   - Apply DSSP-to-Q3 mapping.
   - Generate residue-wise samples with window size 13.
   - Store processed binary data for repeatable timing experiments.
2. **Serial baseline**
   - Implement a compact feedforward neural network (2 hidden layers, 3-class softmax).
   - Train with cross-entropy loss and SGD-style updates.
   - Record Q3 accuracy and per-epoch timing.
3. **Parallel implementations**
   - OpenMP: thread-parallel gradient computation with reduction.
   - Pthreads: master-worker pattern with persistent thread pool and synchronization barriers.
   - MPI: data-parallel training across three laptops with `MPI_Allreduce`.
   - Hybrid: MPI across nodes and OpenMP within each node.
4. **Experiment design**
   - Keep dataset split, seed, epochs, and batch size fixed for fair comparison.
   - Run thread/rank/batch sweeps and compute speedup vs serial.
   - Verify that parallel Q3 remains close to serial.
