# HPC Project Current Plan

## Status

The fresh HPC experiment run is complete on `blackbox-Z790-EAGLE-AX`.

Implemented binaries:

- `train_serial`
- `train_omp`
- `train_pthreads`
- `train_mpi`
- `train_hybrid`
- `train_cuda`

Fresh generated artifacts:

- JSON run logs under `results/serial`, `results/openmp`, `results/pthreads`, `results/mpi`, `results/hybrid`, `results/cuda`, and `results/smoke`
- Timing summary: `plots/timing_summary.csv`
- Accuracy summary: `plots/accuracy_table.csv`
- Plots:
  - `plots/time_vs_threads.png`
  - `plots/speedup_vs_threads.png`
  - `plots/time_vs_ranks_mpi.png`
  - `plots/hybrid_heatmap.png`
  - `plots/cuda_time_vs_batch.png`
- Report source updated with the fresh numbers: `docs/report/analysis_report.tex`

## Host

- CPU: Intel Core i9-14900K, 32 logical CPUs.
- GPU: NVIDIA GeForce RTX 5090, visible via `nvidia-smi`.
- MPI: Open MPI 5.0.8, run locally on this HPC host.
- CUDA: `nvcc` 12.4, NVIDIA driver 595.71.05.

## Experiment Policy

- Timing sweeps use 20 epochs and 3 repeats.
- Accuracy runs use 80 epochs and one representative configuration per variant.
- Timing metric: mean `epoch_time_s` from JSON `epoch_log`; report standard deviation across repeated runs.
- Accuracy baseline: serial 80-epoch test Q3 is `59.2814%`.
- CPU accuracy tolerance: within +/- `0.5` percentage points of serial.
- CUDA accuracy tolerance: within +/- `1.0` point of serial.

## Key Results

- Accuracy parity passes for all variants:
  - serial: `59.2814%`
  - OpenMP 16 threads: `59.4233%`
  - Pthreads 16 threads: `59.2263%`
  - MPI 4 ranks: `59.2263%`
  - Hybrid 4 ranks x 4 threads: `59.2972%`
  - CUDA batch 64: `59.3681%`
- Best 20-epoch timing points:
  - OpenMP: 8 threads, `0.5032 s/epoch`
  - Pthreads: 8 threads, `0.6879 s/epoch`
  - MPI: 8 ranks, `0.4274 s/epoch`
  - Hybrid: 8 ranks x 2 threads, `0.3779 s/epoch`
  - CUDA: batch 512, `0.0132 s/epoch`, but with lower Q3; batch 64 is used for fair accuracy comparison.
- CUDA 80-epoch batch-64 run: `6.3 s` total, about `37.8x` faster than the serial 80-epoch baseline.

## Remaining Work

- Compile `docs/report/analysis_report.tex` into PDF on a machine with LaTeX installed.
- Optional: export `docs/diagrams/hpc_workflows.drawio` to image/PDF if the presentation needs separate diagram files.
- Optional: do a final wording pass on the report after PDF compilation to check page breaks and figure placement.

## Reproduction Commands

```bash
make train_serial train_omp train_pthreads train_mpi train_hybrid train_cuda
bash scripts/run_hpc_experiments.sh smoke
bash scripts/run_hpc_experiments.sh timing-cpu
bash scripts/run_hpc_experiments.sh timing-cuda
bash scripts/run_hpc_experiments.sh accuracy
python3 scripts/make_plots.py
```
