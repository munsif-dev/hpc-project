# CUDA Setup & Run Guide

The CUDA variant (`train_cuda`) uses cuBLAS for all weight/activation GEMMs and a handful of custom kernels for bias+ReLU, the ReLU derivative mask, and the fused softmax+cross-entropy+gradient computation. Forward and backward passes, and the SGD update, all stay on the GPU.

## Prerequisites (on the lab / friend's machine)

- NVIDIA GPU with CUDA capability >= 3.5.
- CUDA toolkit installed (any recent version ≥ 10.1 should work; 11.x or 12.x recommended):
  ```bash
  nvcc --version
  nvidia-smi
  ```
- cuBLAS (bundled with the CUDA toolkit — no separate install).
- `gcc`, `make` (to build the common C objects).

## Copy the project

```bash
# From your laptop:
rsync -av --exclude='.git' ~/hpc-project/ user@lab:~/hpc-project/
```

Make sure `data/processed/cb513/binary/` is present.

## Build

```bash
cd ~/hpc-project
make clean
make train_cuda          # also builds common/*.o with gcc
```

If `nvcc` isn't on PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Run

```bash
./train_cuda --data data/processed/cb513/binary \
             --epochs 80 --batch 64 --lr 0.01 --seed 42 \
             --hidden1 256 --hidden2 128 \
             --out results/cuda --verbose 1
```

## Required experiment sweep (Phase 10)

Batch-size sweep — same seed, same other hyperparameters:

```bash
for bs in 32 64 128 256; do
    ./train_cuda --data data/processed/cb513/binary \
                 --epochs 80 --batch $bs --lr 0.01 --seed 42 \
                 --hidden1 256 --hidden2 128 \
                 --out results/cuda --verbose 1
done
```

Each run writes a JSON log named `cuda_t1_s42_<timestamp>.json` into `results/cuda/`.

## Recording the environment

Add this to the top of your run so the analysis report can quote exact versions:
```bash
nvcc --version        >  results/cuda/ENVIRONMENT.txt
nvidia-smi            >> results/cuda/ENVIRONMENT.txt
gcc --version          >> results/cuda/ENVIRONMENT.txt
```

## Copy results back

```bash
# On your laptop:
rsync -av user@lab:~/hpc-project/results/cuda/ ~/hpc-project/results/cuda/
```

## Accuracy expectation

Serial baseline test Q3 ≈ 62.74% (seed=42, hidden=256/128, 80 epochs).
CUDA Q3 should be within ±1% of that. Small drift is normal because cuBLAS
GEMM ordering and fused softmax+gradient introduce different float rounding
than the sequential CPU path — the *algorithm* is identical.

## Troubleshooting

- **"CUDA out of memory":** drop `--hidden1`/`--hidden2`, or batch size. With 256/128/64 the model fits easily in < 100 MB.
- **"invalid device function":** nvcc compiled for a different compute capability. Add `-arch=sm_XX` to the nvcc line in the Makefile (check with `nvidia-smi --query-gpu=compute_cap --format=csv`).
- **Silent zero predictions / loss blowup:** verify `--seed` is consistent and that the serial baseline runs fine on the same machine first.
