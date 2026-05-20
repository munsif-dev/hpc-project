# HPC Project Current Plan

## Status

This repo now contains all six implementations for the CB513 Q3 project:

- `train_serial`
- `train_omp`
- `train_pthreads`
- `train_mpi`
- `train_hybrid`
- `train_cuda`

The serial/OpenMP/Pthreads result logs already exist. MPI, Hybrid, and CUDA still need complete 20-epoch timing sweeps and 80-epoch accuracy runs before the report can be finalized.

The project has moved from the old multi-laptop plan to a local HPC-first plan:

- CPU: Intel Core i9-14900K, 32 logical CPUs.
- GPU: NVIDIA GeForce RTX 5090, visible from a normal shell via `nvidia-smi`.
- MPI: run local multi-rank jobs on this HPC host first; keep multi-laptop MPI as optional backup/demo material.
- CUDA: run from a shell that can access `/dev/nvidia*`; Codex sandboxed commands may not see the GPU.

## Execution Order

1. Run quick validation:
   ```bash
   bash scripts/run_hpc_experiments.sh smoke
   ```

2. Run CPU/MPI/Hybrid timing:
   ```bash
   bash scripts/run_hpc_experiments.sh timing-cpu
   ```

3. Run CUDA timing from a GPU-visible shell:
   ```bash
   bash scripts/run_hpc_experiments.sh timing-cuda
   ```

4. Run final 80-epoch accuracy:
   ```bash
   bash scripts/run_hpc_experiments.sh accuracy
   ```

5. Regenerate plots and summary CSV files:
   ```bash
   python3 scripts/make_plots.py
   ```

6. Finalize `docs/report/analysis_report.tex` by replacing placeholder values with generated values from:
   - `plots/accuracy_table.csv`
   - `plots/timing_summary.csv`
   - `plots/*.png`

## Experiment Policy

- Accuracy baseline: serial 80-epoch test Q3 is `62.7403%`.
- CPU accuracy tolerance: within +/- `0.5` percentage points of serial.
- CUDA accuracy tolerance: within +/- `1.0` point of serial.
- Timing metric: mean `epoch_time_s` from JSON `epoch_log`; report standard deviation across repeated runs.
- Timing sweeps use 20 epochs and 3 repeats.
- Accuracy runs use 80 epochs and one representative configuration per variant.

## Report Artifacts

- Main report: `docs/report/analysis_report.tex`
- Editable diagrams: `docs/diagrams/hpc_workflows.drawio`
- Generated plots: `plots/*.png`
- Generated tables: `plots/accuracy_table.csv`, `plots/timing_summary.csv`

## Remaining Work

- Run full experiment profiles.
- Confirm CUDA smoke/accuracy outside the sandbox.
- Generate final plots/tables.
- Fill final numerical values into the report.
- Compile the report PDF.
