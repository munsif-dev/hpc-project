# Protein Secondary Structure Prediction on CB513 — HPC Variants

Serial, OpenMP, POSIX threads, MPI, Hybrid (MPI+OpenMP), and CUDA implementations of a small MLP (Input 260 → Hidden1 → Hidden2 → Softmax 3) for per-residue Q3 prediction on the CB513 benchmark. Reproduces the experimental setup from *Zhong et al., J. Supercomputing 2007* ("Parallel protein secondary structure prediction schemes using Pthread and OpenMP…") with sliding-window = 13 and one-hot input.

## Directory layout

```
src/
  common/       metrics, CLI, logger, timer, data loader (reused by every variant)
  models/       MLP (forward / backward / SGD update) — shared across variants
  serial/       train_serial
  openmp/       train_omp
  pthreads/     train_pthreads
  mpi/          train_mpi        (OpenMPI)
  hybrid/       train_hybrid     (OpenMPI + OpenMP)
  cuda/         train_cuda       (cuBLAS + custom kernels)
data/
  raw/cb513/                     CB513 distribution tarball + .all files
  processed/cb513/               tsv -> binary + metadata + splits
scripts/
  download_cb513.sh              one-time dataset fetch
  preprocess_*.py                feature pipeline
  run_sweeps.sh                  local timing sweeps (serial + omp + pthreads)
  run_hpc_experiments.sh         full HPC experiment runner
  make_plots.py                  generate all report figures
  cluster_setup.md               multi-laptop MPI runbook
  cuda_setup.md                  CUDA build/run guide for the HPC GPU
docs/diagrams/
  hpc_workflows.drawio           editable workflow diagrams
results/<variant>/               per-run JSON logs (epoch trajectory + final)
plots/                           PNG figures + accuracy CSV
report/                          final analysis report
```

## Build

```bash
make train_serial     # baseline
make train_omp        # needs gcc with -fopenmp
make train_pthreads   # needs libpthread
make train_mpi        # needs mpicc (OpenMPI)
make train_hybrid     # mpicc + -fopenmp
make train_cuda       # needs nvcc + cuBLAS (link -lcublas)
```

## Run — common CLI

Every binary accepts the same flags:

```
--data    PATH   processed CB513 binary directory (default: data/processed/cb513/binary)
--epochs  N      number of SGD epochs (default: 80)
--batch   N      mini-batch size (default: 64)
--lr      F      learning rate (default: 0.01)
--seed    N      RNG seed — controls weight init and shuffle (default: 42)
--threads N      shared-memory thread count (OpenMP / Pthreads / Hybrid inner)
--hidden1 N      first hidden layer size (default: 256)
--hidden2 N      second hidden layer size (default: 128)
--out     DIR    where to write the JSON log (default: results/<variant>)
--verbose 0|1    per-epoch stdout
```

Examples:

```bash
./train_serial     --epochs 80 --batch 64 --seed 42 --out results/serial
./train_omp        --epochs 80 --batch 64 --seed 42 --threads 8 --out results/openmp
./train_pthreads   --epochs 80 --batch 64 --seed 42 --threads 8 --out results/pthreads

# MPI: single-machine ranks on the HPC host
mpirun -np 4 ./train_mpi --epochs 80 --batch 64 --seed 42 --out results/mpi

# Hybrid: MPI ranks plus OpenMP threads per rank
OMP_PROC_BIND=true OMP_PLACES=cores \
mpirun -np 4 ./train_hybrid --threads 4 --epochs 80 --batch 64 --seed 42 \
       --out results/hybrid

# CUDA: run from a shell where nvidia-smi can see the RTX GPU
./train_cuda --epochs 80 --batch 64 --seed 42 --out results/cuda
```

## Reference accuracy (serial baseline)

| config (seed=42, h1=256, h2=128, 80 epochs) | test Q3 |
|---|---|
| serial | **62.74 %** |

Every parallel variant should land within ±0.5 % of this value (±1 % for CUDA because of float-reduction ordering differences in cuBLAS). See `plots/accuracy_table.csv` after running the sweeps.

## Reproducing the figures

```bash
bash scripts/run_hpc_experiments.sh smoke       # quick 1-epoch validation
bash scripts/run_hpc_experiments.sh timing-cpu  # CPU/MPI/hybrid timing sweeps
bash scripts/run_hpc_experiments.sh timing-cuda # CUDA batch-size sweep
bash scripts/run_hpc_experiments.sh accuracy    # 80-epoch accuracy runs
python3 scripts/make_plots.py                  # results/*.json → plots/*.png + CSV
```

`scripts/make_plots.py` writes `plots/timing_summary.csv` with mean/std across repeated runs and `plots/accuracy_table.csv` for the final Q3 comparison. The full `all` profile can take a long time; run the split profiles above when you want checkpoints.

## Data

CB513 (Cuff & Barton, 1999) was retrieved via `scripts/download_cb513.sh`. Labels are 8-state DSSP collapsed to 3 states:

- H, G, I → **H** (helix)
- B, E   → **E** (sheet)
- all others → **C** (coil)

Window size 13 with flat-edge padding, one-hot encoding (20 × 13 = 260 dims), fixed 70/15/15 train/val/test split (`scripts/split_dataset.py`), all deterministic under the seed.
