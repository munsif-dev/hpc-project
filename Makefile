CC      = gcc
MPICC   = mpicc
NVCC    = nvcc
CFLAGS  = -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS = -lm

COMMON_SRCS = src/common/metrics.c \
              src/common/logger.c  \
              src/common/cli.c     \
              src/common/timer.c   \
              src/common/data_loader.c \
              src/models/mlp.c
COMMON_OBJS = $(COMMON_SRCS:.c=.o)

.PHONY: all clean test_metrics

all: train_serial train_omp train_pthreads

# Serial baseline
train_serial: $(COMMON_OBJS) src/serial/train.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# OpenMP variant
src/openmp/train.o: src/openmp/train.c
	$(CC) $(CFLAGS) -fopenmp -c -o $@ $<

train_omp: $(COMMON_OBJS) src/openmp/train.o
	$(CC) $(CFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# POSIX Threads variant
train_pthreads: $(COMMON_OBJS) src/pthreads/train.o
	$(CC) $(CFLAGS) -pthread -o $@ $^ $(LDFLAGS)

# MPI variant (requires OpenMPI)
src/mpi/train.o: src/mpi/train.c
	$(MPICC) $(CFLAGS) -c -o $@ $<

train_mpi: $(COMMON_OBJS) src/mpi/train.o
	$(MPICC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Hybrid MPI + OpenMP
src/hybrid/train.o: src/hybrid/train.c
	$(MPICC) $(CFLAGS) -fopenmp -c -o $@ $<

train_hybrid: $(COMMON_OBJS) src/hybrid/train.o
	$(MPICC) $(CFLAGS) -fopenmp -o $@ $^ $(LDFLAGS)

# CUDA variant (uses cuBLAS for GEMMs + custom kernels for activations)
src/cuda/train.o: src/cuda/train.cu
	$(NVCC) -O3 -Iinclude -c -o $@ $<

train_cuda: $(COMMON_OBJS) src/cuda/train.o
	$(NVCC) -O3 -Iinclude -o $@ $^ -lcublas $(LDFLAGS)

# C test driver for Phase 3 verification
test_metrics: $(COMMON_OBJS) tests/test_metrics.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./test_metrics --data data/processed/cb513/binary --out results/serial

# Generic compile rule for all .c files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	find src -name "*.o" -delete
	find tests -name "*.o" -delete
	rm -f train_serial train_omp train_pthreads train_mpi train_hybrid train_cuda test_metrics
