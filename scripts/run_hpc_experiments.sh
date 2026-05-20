#!/usr/bin/env bash
# Run the reproducible experiment matrix for the CB513 HPC project.
#
# Profiles:
#   smoke        quick 1-epoch correctness/runtime checks under results/smoke/
#   timing-cpu   serial/openmp/pthreads/mpi/hybrid 20-epoch timing sweeps
#   timing-cuda  CUDA batch-size timing sweep
#   accuracy     80-epoch accuracy runs for every variant
#   all-cpu      smoke + timing-cpu + CPU accuracy runs
#   all          smoke + timing-cpu + timing-cuda + all accuracy runs
#
# Override defaults with environment variables, e.g.:
#   REPEATS=1 TIMING_EPOCHS=10 bash scripts/run_hpc_experiments.sh timing-cpu

set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${1:-smoke}"

DATA="${DATA:-data/processed/cb513/binary}"
SEED="${SEED:-42}"
HIDDEN1="${HIDDEN1:-256}"
HIDDEN2="${HIDDEN2:-128}"
LR="${LR:-0.01}"
BATCH="${BATCH:-64}"
TIMING_EPOCHS="${TIMING_EPOCHS:-20}"
ACCURACY_EPOCHS="${ACCURACY_EPOCHS:-80}"
REPEATS="${REPEATS:-3}"

OMP_THREADS="${OMP_THREADS:-1 2 4 8 16 24 32}"
PTHREADS_THREADS="${PTHREADS_THREADS:-1 2 4 8 16 24 32}"
MPI_RANKS="${MPI_RANKS:-1 2 3 4 8 16}"
HYBRID_CONFIGS="${HYBRID_CONFIGS:-1x1 1x8 1x16 1x24 2x4 2x8 4x4 4x8 8x2 8x4}"
CUDA_BATCHES="${CUDA_BATCHES:-32 64 128 256 512}"

COMMON_ARGS=(
  --data "$DATA"
  --batch "$BATCH"
  --lr "$LR"
  --seed "$SEED"
  --hidden1 "$HIDDEN1"
  --hidden2 "$HIDDEN2"
  --verbose 0
)

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_logged() {
  echo
  echo "[$(timestamp)] $*"
  "$@"
}

build_cpu() {
  run_logged make train_serial train_omp train_pthreads train_mpi train_hybrid
}

build_cuda() {
  run_logged make train_cuda
}

capture_environment() {
  mkdir -p results/environment results/cuda
  {
    echo "Captured at: $(timestamp)"
    echo
    echo "== hostname =="
    hostname
    echo
    echo "== lscpu =="
    lscpu
    echo
    echo "== gcc =="
    gcc --version
    echo
    echo "== mpirun =="
    mpirun --version
    echo
    echo "== nvcc =="
    if command -v nvcc >/dev/null 2>&1; then nvcc --version; else echo "nvcc not found"; fi
    echo
    echo "== nvidia-smi =="
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; else echo "nvidia-smi not found"; fi
  } > results/environment/HPC_ENVIRONMENT.txt
  cp results/environment/HPC_ENVIRONMENT.txt results/cuda/ENVIRONMENT.txt
}

run_smoke() {
  build_cpu
  mkdir -p results/smoke/{serial,openmp,pthreads,mpi,hybrid,cuda}

  run_logged ./train_serial "${COMMON_ARGS[@]}" --epochs 1 --out results/smoke/serial
  run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
    ./train_omp "${COMMON_ARGS[@]}" --epochs 1 --threads 2 --out results/smoke/openmp
  run_logged ./train_pthreads "${COMMON_ARGS[@]}" --epochs 1 --threads 2 --out results/smoke/pthreads
  run_logged mpirun -np 2 ./train_mpi "${COMMON_ARGS[@]}" --epochs 1 --out results/smoke/mpi
  run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
    mpirun -np 2 ./train_hybrid "${COMMON_ARGS[@]}" --epochs 1 --threads 2 --out results/smoke/hybrid

  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    build_cuda
    run_logged ./train_cuda "${COMMON_ARGS[@]}" --epochs 1 --out results/smoke/cuda
  else
    echo "[warn] CUDA smoke skipped: nvidia-smi is not visible in this shell."
  fi
}

run_timing_cpu() {
  build_cpu

  for rep in $(seq 1 "$REPEATS"); do
    echo
    echo "=== CPU timing repeat $rep/$REPEATS ==="

    run_logged ./train_serial "${COMMON_ARGS[@]}" \
      --epochs "$TIMING_EPOCHS" --out results/serial

    for threads in $OMP_THREADS; do
      run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
        ./train_omp "${COMMON_ARGS[@]}" --epochs "$TIMING_EPOCHS" \
        --threads "$threads" --out results/openmp
    done

    for threads in $PTHREADS_THREADS; do
      run_logged ./train_pthreads "${COMMON_ARGS[@]}" --epochs "$TIMING_EPOCHS" \
        --threads "$threads" --out results/pthreads
    done

    for ranks in $MPI_RANKS; do
      run_logged mpirun -np "$ranks" ./train_mpi "${COMMON_ARGS[@]}" \
        --epochs "$TIMING_EPOCHS" --out results/mpi
    done

    for cfg in $HYBRID_CONFIGS; do
      ranks="${cfg%x*}"
      threads="${cfg#*x}"
      run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
        mpirun -np "$ranks" ./train_hybrid "${COMMON_ARGS[@]}" \
        --epochs "$TIMING_EPOCHS" --threads "$threads" --out results/hybrid
    done
  done
}

run_timing_cuda() {
  build_cuda

  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi is not visible. Run this profile from a shell that can access /dev/nvidia*."
    exit 1
  fi

  for rep in $(seq 1 "$REPEATS"); do
    echo
    echo "=== CUDA timing repeat $rep/$REPEATS ==="
    for cuda_batch in $CUDA_BATCHES; do
      run_logged ./train_cuda "${COMMON_ARGS[@]}" --epochs "$TIMING_EPOCHS" \
        --batch "$cuda_batch" --out results/cuda
    done
  done
}

run_accuracy_cpu() {
  build_cpu

  run_logged ./train_serial "${COMMON_ARGS[@]}" \
    --epochs "$ACCURACY_EPOCHS" --out results/serial
  run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
    ./train_omp "${COMMON_ARGS[@]}" --epochs "$ACCURACY_EPOCHS" \
    --threads 16 --out results/openmp
  run_logged ./train_pthreads "${COMMON_ARGS[@]}" --epochs "$ACCURACY_EPOCHS" \
    --threads 16 --out results/pthreads
  run_logged mpirun -np 4 ./train_mpi "${COMMON_ARGS[@]}" \
    --epochs "$ACCURACY_EPOCHS" --out results/mpi
  run_logged env OMP_PROC_BIND=true OMP_PLACES=cores \
    mpirun -np 4 ./train_hybrid "${COMMON_ARGS[@]}" --epochs "$ACCURACY_EPOCHS" \
    --threads 4 --out results/hybrid
}

run_accuracy_cuda() {
  build_cuda

  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi is not visible. Run CUDA accuracy from a shell that can access /dev/nvidia*."
    exit 1
  fi

  run_logged ./train_cuda "${COMMON_ARGS[@]}" \
    --epochs "$ACCURACY_EPOCHS" --batch "$BATCH" --out results/cuda
}

case "$PROFILE" in
  smoke)
    capture_environment
    run_smoke
    ;;
  timing-cpu)
    capture_environment
    run_timing_cpu
    ;;
  timing-cuda)
    capture_environment
    run_timing_cuda
    ;;
  accuracy)
    capture_environment
    run_accuracy_cpu
    run_accuracy_cuda
    ;;
  all-cpu)
    capture_environment
    run_smoke
    run_timing_cpu
    run_accuracy_cpu
    ;;
  all)
    capture_environment
    run_smoke
    run_timing_cpu
    run_timing_cuda
    run_accuracy_cpu
    run_accuracy_cuda
    ;;
  *)
    echo "Usage: $0 {smoke|timing-cpu|timing-cuda|accuracy|all-cpu|all}" >&2
    exit 2
    ;;
esac

echo
echo "[$(timestamp)] Experiment profile '$PROFILE' completed."
echo "Next: python3 scripts/make_plots.py"
