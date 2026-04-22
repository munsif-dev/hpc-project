#!/usr/bin/env bash
# scripts/run_sweeps.sh — run all timing sweeps we can do on the local machine.
# Sweeps run at 20 epochs (enough for stable per-epoch time measurements and
# a meaningful Q3 parity check); for the full-accuracy table we rely on the
# existing 80-epoch serial baseline in results/serial/.
#
# Usage:   bash scripts/run_sweeps.sh
# Outputs: JSON logs under results/<variant>/

set -euo pipefail
cd "$(dirname "$0")/.."   # project root

COMMON="--data data/processed/cb513/binary --epochs 20 --batch 64 --lr 0.01 \
        --seed 42 --hidden1 256 --hidden2 128 --verbose 0"

echo "=== Building ==="
make train_serial train_omp train_pthreads >/dev/null

echo "=== Serial (1 thread, reference) ==="
./train_serial $COMMON --out results/serial

echo "=== OpenMP sweep ==="
for t in 1 2 4 8 16; do
    echo "-- omp threads=$t --"
    OMP_PROC_BIND=true OMP_PLACES=cores \
        ./train_omp $COMMON --threads $t --out results/openmp
done

echo "=== Pthreads sweep ==="
for t in 1 2 4 8 16; do
    echo "-- pthreads threads=$t --"
    ./train_pthreads $COMMON --threads $t --out results/pthreads
done

echo "=== Done. Results in results/{serial,openmp,pthreads}/ ==="
