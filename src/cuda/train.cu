/* src/cuda/train.cu — CUDA + cuBLAS MLP training for protein secondary structure (Q3).
 *
 * Parallel strategy:
 *   - Whole training / val / test tensors live on the GPU for the entire run.
 *   - For each mini-batch:
 *       1. A gather kernel builds a column-major batch matrix X_batch [IN x B]
 *          from the shuffled host index list.
 *       2. Forward pass uses cuBLAS sgemm for W·X matmuls plus custom kernels
 *          for bias-add+ReLU and the fused softmax + cross-entropy + dZ3
 *          (output becomes P - onehot(y) in-place).
 *       3. Backward pass is fused with the SGD update: the gradient of each
 *          weight matrix is produced by one cuBLAS sgemm with
 *          alpha = -lr/batch_size and beta = 1, writing directly into the
 *          weight buffer. Biases are updated via cuBLAS sgemv with a vector
 *          of ones. No explicit gradient buffers are materialised.
 *   - Parallelism is at two levels:
 *       DATA:    every sample in the batch is processed in parallel inside
 *                each GEMM (and inside the custom kernels, one thread per
 *                (feature, sample) pair).
 *       WEIGHTS: every weight element is updated in parallel inside the
 *                gradient-accumulation GEMM.
 *
 * Weight / activation layout is COLUMN-MAJOR throughout (cuBLAS native).
 * Host data (Dataset.X) is row-major [N x IN]; the gather kernel converts
 * on the fly so no global transpose is needed.
 *
 * Validation and final test Q3 are computed on the HOST with the existing
 * `mlp_predict()` — after copying updated weights back from device to host.
 * This avoids duplicating prediction code on the device.
 *
 * Architecture: Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Build:        nvcc -O3 -Iinclude src/cuda/train.cu  (see Makefile target train_cuda)
 * Usage:        ./train_cuda [--data PATH] [--epochs N] [--batch N]
 *                            [--lr F] [--seed N] [--hidden1 N] [--hidden2 N]
 *                            [--out DIR] [--verbose 0|1]
 */

extern "C" {
#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"
#include "models/mlp.h"
}

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t err__ = (call);                                             \
    if (err__ != cudaSuccess) {                                             \
        fprintf(stderr, "[cuda] %s at %s:%d\n",                             \
                cudaGetErrorString(err__), __FILE__, __LINE__);             \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define CUBLAS_CHECK(call) do {                                             \
    cublasStatus_t s__ = (call);                                            \
    if (s__ != CUBLAS_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "[cublas] error %d at %s:%d\n",                     \
                (int)s__, __FILE__, __LINE__);                              \
        exit(1);                                                            \
    }                                                                       \
} while (0)

/* -----------------------------------------------------------------------
 * Host-side shuffle (same LCG as every other variant so results are
 * comparable across runs / implementations).
 * ----------------------------------------------------------------------- */
static unsigned int shuf_state;
static void shuf_seed(unsigned int s) { shuf_state = s; }
static unsigned int shuf_rand(void) {
    shuf_state = shuf_state * 1664525u + 1013904223u;
    return shuf_state;
}
static void shuffle(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(shuf_rand() % (unsigned int)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* -----------------------------------------------------------------------
 * Kernels
 * ----------------------------------------------------------------------- */

/* Build a column-major batch matrix X_batch [IN x B] from the row-major
 * device copy X_all [N x IN] using host-provided sample indices.
 *   X_batch[i + j*IN] = X_all[idx[j] * IN + i]
 */
__global__ void gather_batch_kernel(const float *X_all,
                                    const int   *d_idx,   /* length B */
                                    int IN, int B,
                                    float *X_batch)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;   /* feature index */
    int j = blockIdx.x * blockDim.x + threadIdx.x;   /* sample in batch */
    if (i < IN && j < B) {
        int sample = d_idx[j];
        X_batch[i + (size_t)j * IN] = X_all[(size_t)sample * IN + i];
    }
}

/* A[m + j*M] += b[m]; if (A<0) A=0;   (bias add + ReLU, column-major) */
__global__ void bias_relu_kernel(float *A, const float *b, int M, int B)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && j < B) {
        float v = A[m + (size_t)j * M] + b[m];
        A[m + (size_t)j * M] = v > 0.0f ? v : 0.0f;
    }
}

/* Mask gradient with ReLU derivative:   dZ[m,j] *= (A[m,j] > 0) ? 1 : 0 */
__global__ void relu_mask_kernel(float *dZ, const float *A, int M, int B)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && j < B) {
        size_t off = m + (size_t)j * M;
        if (A[off] <= 0.0f) dZ[off] = 0.0f;
    }
}

/* Gather labels for the current batch:   d_ybatch[j] = d_y_all[d_idx[j]]  */
__global__ void gather_labels_kernel(const int *d_y_all, const int *d_idx,
                                     int B, int *d_y_batch)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < B) d_y_batch[j] = d_y_all[d_idx[j]];
}

/* Fused:  add b3 to every column of Z3, then apply softmax along OUT axis,
 * accumulate cross-entropy loss per sample, and convert Z3 to dZ3 = P - onehot(y).
 * One CUDA block per batch column. OUT is tiny (3), so a single thread per
 * block is enough; we keep threads=32 for warp efficiency but only thread 0 acts.
 */
__global__ void softmax_xent_grad_kernel(float *Z3, const float *b3,
                                         const int *d_ybatch,
                                         float *d_loss_accum,
                                         int OUT, int B)
{
    int j = blockIdx.x;
    if (j >= B) return;
    if (threadIdx.x != 0) return;

    float *z = Z3 + (size_t)j * OUT;
    /* bias-add */
    float max_z = -1e30f;
    for (int k = 0; k < OUT; k++) { z[k] += b3[k]; if (z[k] > max_z) max_z = z[k]; }
    /* softmax */
    float sum = 0.0f;
    float p[16];   /* OUT is small (3); 16 is a safe upper bound */
    for (int k = 0; k < OUT; k++) { p[k] = expf(z[k] - max_z); sum += p[k]; }
    float inv = 1.0f / sum;
    int y = d_ybatch[j];
    float p_true = p[y] * inv;
    if (p_true < 1e-9f) p_true = 1e-9f;

    atomicAdd(d_loss_accum, -logf(p_true));

    /* Convert Z3 column in-place: z := P - onehot(y) */
    for (int k = 0; k < OUT; k++) z[k] = p[k] * inv;
    z[y] -= 1.0f;
}

/* -----------------------------------------------------------------------
 * Batched inference on the device (used for val/test Q3 each epoch).
 * Returns predictions in host y_pred[] (size n).
 * Uses the same cuBLAS handle and device weights; reuses workspace d_A1/d_A2/d_Z3
 * from the main training loop (caller provides).
 * ----------------------------------------------------------------------- */
static void cuda_predict(cublasHandle_t cublas,
                         const float *d_W1, const float *d_b1,
                         const float *d_W2, const float *d_b2,
                         const float *d_W3, const float *d_b3,
                         const float *d_X, int n, int IN,
                         int H1, int H2, int OUT,
                         int batch,
                         float *d_A1_big, float *d_A2_big, float *d_Z3_big,
                         int *y_pred_host)
{
    const float one = 1.0f, zero = 0.0f;
    for (int b0 = 0; b0 < n; b0 += batch) {
        int B = batch; if (b0 + B > n) B = n - b0;

        /* X slice starts at offset b0 * IN (row-major host side, but we
         * uploaded a column-major version for inference — see caller). */
        const float *d_Xb = d_X + (size_t)b0 * IN;

        /* A1 = W1 * Xb */
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 H1, B, IN, &one,
                                 d_W1, H1, d_Xb, IN,
                                 &zero, d_A1_big, H1));
        dim3 blk(16, 16), grd((B+15)/16, (H1+15)/16);
        bias_relu_kernel<<<grd, blk>>>(d_A1_big, d_b1, H1, B);

        /* A2 = W2 * A1 */
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 H2, B, H1, &one,
                                 d_W2, H2, d_A1_big, H1,
                                 &zero, d_A2_big, H2));
        dim3 grd2((B+15)/16, (H2+15)/16);
        bias_relu_kernel<<<grd2, blk>>>(d_A2_big, d_b2, H2, B);

        /* Z3 = W3 * A2 */
        CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                 OUT, B, H2, &one,
                                 d_W3, OUT, d_A2_big, H2,
                                 &zero, d_Z3_big, OUT));

        /* Pull Z3 (logits + b3 not applied to logits yet for inference).
         * Simpler: apply b3 on host via argmax, since OUT=3 is tiny. */
        float *h_Z3 = (float *)malloc((size_t)OUT * B * sizeof(float));
        float *h_b3 = (float *)malloc((size_t)OUT * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_Z3, d_Z3_big, (size_t)OUT*B*sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b3, d_b3,     (size_t)OUT*sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int j = 0; j < B; j++) {
            int best = 0; float bv = h_Z3[0 + j*OUT] + h_b3[0];
            for (int k = 1; k < OUT; k++) {
                float v = h_Z3[k + j*OUT] + h_b3[k];
                if (v > bv) { bv = v; best = k; }
            }
            y_pred_host[b0 + j] = best;
        }
        free(h_Z3); free(h_b3);
    }
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
    Args args;
    parse_args(argc, argv, &args, "cuda");

    if (args.verbose) {
        printf("=== train_cuda ===\n");
        cudaDeviceProp prop; int dev = 0;
        cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        printf("  device      : %s (cc %d.%d, %zu MB)\n",
               prop.name, prop.major, prop.minor,
               (size_t)(prop.totalGlobalMem >> 20));
        print_args(&args);
    }

    /* --- Load data (host) --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
           train.n, val.n, test.n, train.feature_dim);

    int IN = train.feature_dim, H1 = args.hidden1, H2 = args.hidden2, OUT = 3;

    /* --- Run log --- */
    RunLog log;
    runlog_init(&log);
    strncpy(log.variant, "cuda", sizeof(log.variant) - 1);
    snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
    log.seed          = args.seed;
    log.epochs        = args.epochs;
    log.batch_size    = args.batch_size;
    log.learning_rate = args.lr;
    log.hidden1       = H1;
    log.hidden2       = H2;
    log.threads       = 1;
    log.mpi_ranks     = 1;
    runlog_set_dataset(&log, train.n, val.n, test.n, IN, OUT);
    runlog_set_compiler(&log, "nvcc", "-O3");

    /* --- Initialise shared model on host (deterministic) --- */
    MLP m;
    mlp_init(&m, IN, H1, H2, OUT, (unsigned int)args.seed);

    /* --- cuBLAS --- */
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    /* --- Device buffers: weights (column-major) --- */
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    CUDA_CHECK(cudaMalloc(&d_W1, (size_t)H1*IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, (size_t)H1    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, (size_t)H2*H1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, (size_t)H2    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W3, (size_t)OUT*H2* sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, (size_t)OUT   * sizeof(float)));

    /* mlp_init produced W1 as row-major [H1 x IN]. In column-major with
     * leading dim H1, the same byte layout represents [H1 x IN] correctly
     * (W1[i + j*H1] = W_row_major[i*IN + j]? NO — they differ). We need
     * to transpose on upload so cuBLAS sees W1 as column-major [H1 x IN].
     * Instead of transposing, observe:  the host W1 stored row-major as
     * W1_row[i, k] = weights[i*IN + k]. We want d_W1 column-major with
     * d_W1[i + k*H1] = W1_row[i, k] = weights[i*IN + k].
     * So the upload is NOT memcpy — it's a transpose. Do it on host.     */
    auto upload_T = [&](float *d, const float *h, int rows, int cols) {
        /* h is row-major [rows x cols]; d receives column-major [rows x cols]. */
        float *tmp = (float *)malloc((size_t)rows*cols * sizeof(float));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                tmp[r + (size_t)c * rows] = h[(size_t)r*cols + c];
        CUDA_CHECK(cudaMemcpy(d, tmp, (size_t)rows*cols*sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);
    };
    auto download_T = [&](float *h, const float *d, int rows, int cols) {
        /* d is column-major [rows x cols]; h receives row-major [rows x cols]. */
        float *tmp = (float *)malloc((size_t)rows*cols * sizeof(float));
        CUDA_CHECK(cudaMemcpy(tmp, d, (size_t)rows*cols*sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                h[(size_t)r*cols + c] = tmp[r + (size_t)c * rows];
        free(tmp);
    };

    upload_T(d_W1, m.W1, H1, IN);
    upload_T(d_W2, m.W2, H2, H1);
    upload_T(d_W3, m.W3, OUT, H2);
    CUDA_CHECK(cudaMemcpy(d_b1, m.b1, (size_t)H1 *sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, m.b2, (size_t)H2 *sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b3, m.b3, (size_t)OUT*sizeof(float), cudaMemcpyHostToDevice));

    /* --- Full data on device: row-major for training gather, column-major for inference --- */
    float *d_X_row = NULL;       /* [train.n * IN] row-major, used by gather_batch_kernel */
    int   *d_y_all = NULL;
    CUDA_CHECK(cudaMalloc(&d_X_row, (size_t)train.n*IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_all, (size_t)train.n    * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_X_row, train.X, (size_t)train.n*IN*sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_all, train.y, (size_t)train.n*sizeof(int),
                          cudaMemcpyHostToDevice));

    /* Column-major copies of val and test for cuda_predict */
    auto upload_col = [&](float **d_col, const float *h_row, int n, int feat) {
        float *tmp = (float *)malloc((size_t)n*feat*sizeof(float));
        for (int r = 0; r < n; r++)
            for (int c = 0; c < feat; c++)
                tmp[c + (size_t)r*feat] = h_row[(size_t)r*feat + c];
        CUDA_CHECK(cudaMalloc(d_col, (size_t)n*feat*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(*d_col, tmp, (size_t)n*feat*sizeof(float),
                              cudaMemcpyHostToDevice));
        free(tmp);
    };
    float *d_Xval_col=NULL, *d_Xtest_col=NULL, *d_Xtrain_col=NULL;
    upload_col(&d_Xval_col,   val.X,   val.n,   IN);
    upload_col(&d_Xtest_col,  test.X,  test.n,  IN);
    upload_col(&d_Xtrain_col, train.X, train.n, IN);

    /* --- Batch work buffers --- */
    int B = args.batch_size;
    float *d_X_batch, *d_A1, *d_A2, *d_Z3, *d_dA2, *d_dA1, *d_ones_B, *d_loss;
    int   *d_idx_batch, *d_y_batch;
    CUDA_CHECK(cudaMalloc(&d_X_batch,   (size_t)IN*B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A1,        (size_t)H1*B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A2,        (size_t)H2*B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z3,        (size_t)OUT*B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA2,       (size_t)H2*B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dA1,       (size_t)H1*B  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones_B,    (size_t)B     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss,      sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_idx_batch, (size_t)B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_y_batch,   (size_t)B * sizeof(int)));

    /* Fill d_ones_B with 1.0 */
    {
        float *h_ones = (float *)malloc((size_t)B*sizeof(float));
        for (int i = 0; i < B; i++) h_ones[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d_ones_B, h_ones, (size_t)B*sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_ones);
    }

    /* Inference workspace reused across val/train/test predicts */
    int pred_batch = 512;   /* large batch for inference */
    float *d_predA1, *d_predA2, *d_predZ3;
    CUDA_CHECK(cudaMalloc(&d_predA1, (size_t)H1 *pred_batch*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predA2, (size_t)H2 *pred_batch*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predZ3, (size_t)OUT*pred_batch*sizeof(float)));

    int *indices      = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_val   = (int *)malloc((size_t)val.n   * sizeof(int));
    int *y_pred_train = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_test  = (int *)malloc((size_t)test.n  * sizeof(int));
    for (int i = 0; i < train.n; i++) indices[i] = i;

    /* --- Training loop --- */
    Timer total_timer; timer_start(&total_timer);
    float last_val_q3 = 0.0f;

    const float one = 1.0f, zero = 0.0f;

    for (int epoch = 1; epoch <= args.epochs; epoch++) {
        shuf_seed((unsigned int)(args.seed + epoch));
        shuffle(indices, train.n);

        Timer epoch_timer; timer_start(&epoch_timer);
        float loss_accum_host = 0.0f;

        for (int b = 0; b < train.n; b += B) {
            int actual = B; if (b + actual > train.n) actual = train.n - b;

            /* Upload this batch's indices */
            CUDA_CHECK(cudaMemcpy(d_idx_batch, indices + b,
                                  (size_t)actual*sizeof(int),
                                  cudaMemcpyHostToDevice));

            /* Build X_batch [IN x actual] col-major via gather */
            dim3 blk(16, 16), grd((actual+15)/16, (IN+15)/16);
            gather_batch_kernel<<<grd, blk>>>(d_X_row, d_idx_batch,
                                              IN, actual, d_X_batch);
            gather_labels_kernel<<<(actual+127)/128, 128>>>(d_y_all, d_idx_batch,
                                                             actual, d_y_batch);

            /* Forward */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     H1, actual, IN, &one,
                                     d_W1, H1, d_X_batch, IN,
                                     &zero, d_A1, H1));
            dim3 grdA1((actual+15)/16, (H1+15)/16);
            bias_relu_kernel<<<grdA1, blk>>>(d_A1, d_b1, H1, actual);

            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     H2, actual, H1, &one,
                                     d_W2, H2, d_A1, H1,
                                     &zero, d_A2, H2));
            dim3 grdA2((actual+15)/16, (H2+15)/16);
            bias_relu_kernel<<<grdA2, blk>>>(d_A2, d_b2, H2, actual);

            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     OUT, actual, H2, &one,
                                     d_W3, OUT, d_A2, H2,
                                     &zero, d_Z3, OUT));

            /* Zero loss accumulator, then fused softmax + xent + dZ3 */
            float zeroF = 0.0f;
            CUDA_CHECK(cudaMemcpy(d_loss, &zeroF, sizeof(float),
                                  cudaMemcpyHostToDevice));
            softmax_xent_grad_kernel<<<actual, 32>>>(d_Z3, d_b3, d_y_batch,
                                                     d_loss, OUT, actual);

            /* Pull batch loss back (could defer to epoch end — cheap enough) */
            float batch_loss;
            CUDA_CHECK(cudaMemcpy(&batch_loss, d_loss, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            loss_accum_host += batch_loss;

            /* Backward + fused SGD update.
             * Scale is -lr/actual (matches mlp_sgd_update semantics).
             * Order matters: compute dA? BEFORE updating the weight it came from. */
            float neg_scale = -args.lr / (float)actual;

            /* dA2 = W3^T * dZ3   (dZ3 currently lives in d_Z3) */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     H2, actual, OUT, &one,
                                     d_W3, OUT, d_Z3, OUT,
                                     &zero, d_dA2, H2));

            /* W3 += neg_scale * dZ3 * A2^T  (fused gradient + update) */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                     OUT, H2, actual, &neg_scale,
                                     d_Z3, OUT, d_A2, H2,
                                     &one, d_W3, OUT));
            /* b3 += neg_scale * dZ3 * ones  */
            CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                     OUT, actual, &neg_scale,
                                     d_Z3, OUT, d_ones_B, 1,
                                     &one, d_b3, 1));

            /* dZ2 = dA2 masked by (A2 > 0) */
            dim3 grdH2((actual+15)/16, (H2+15)/16);
            relu_mask_kernel<<<grdH2, blk>>>(d_dA2, d_A2, H2, actual);

            /* dA1 = W2^T * dZ2 */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                     H1, actual, H2, &one,
                                     d_W2, H2, d_dA2, H2,
                                     &zero, d_dA1, H1));

            /* W2 += neg_scale * dZ2 * A1^T */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                     H2, H1, actual, &neg_scale,
                                     d_dA2, H2, d_A1, H1,
                                     &one, d_W2, H2));
            /* b2 += neg_scale * dZ2 * ones */
            CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                     H2, actual, &neg_scale,
                                     d_dA2, H2, d_ones_B, 1,
                                     &one, d_b2, 1));

            /* dZ1 = dA1 masked by (A1 > 0) */
            dim3 grdH1((actual+15)/16, (H1+15)/16);
            relu_mask_kernel<<<grdH1, blk>>>(d_dA1, d_A1, H1, actual);

            /* W1 += neg_scale * dZ1 * X_batch^T */
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                     H1, IN, actual, &neg_scale,
                                     d_dA1, H1, d_X_batch, IN,
                                     &one, d_W1, H1));
            /* b1 += neg_scale * dZ1 * ones */
            CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
                                     H1, actual, &neg_scale,
                                     d_dA1, H1, d_ones_B, 1,
                                     &one, d_b1, 1));
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        double epoch_time = timer_elapsed_s(&epoch_timer);
        float avg_loss = loss_accum_host / (float)train.n;

        /* Validation Q3 */
        cuda_predict(cublas, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                     d_Xval_col, val.n, IN, H1, H2, OUT,
                     pred_batch, d_predA1, d_predA2, d_predZ3, y_pred_val);
        float val_q3 = compute_q3(val.y, y_pred_val, val.n);
        last_val_q3 = val_q3;

        runlog_add_epoch(&log, epoch, avg_loss, val_q3, epoch_time);

        if (args.verbose) {
            printf("Epoch %3d  loss=%.4f  val_q3=%6.2f%%  lr=%.6f  %.2fs\n",
                   epoch, avg_loss, val_q3, args.lr, epoch_time);
        }

        if (args.lr_decay != 1.0f && epoch % args.lr_decay_every == 0)
            args.lr *= args.lr_decay;
    }

    /* --- Final evaluation --- */
    cuda_predict(cublas, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                 d_Xtrain_col, train.n, IN, H1, H2, OUT,
                 pred_batch, d_predA1, d_predA2, d_predZ3, y_pred_train);
    cuda_predict(cublas, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                 d_Xtest_col,  test.n,  IN, H1, H2, OUT,
                 pred_batch, d_predA1, d_predA2, d_predZ3, y_pred_test);

    float train_q3 = compute_q3(train.y, y_pred_train, train.n);
    float test_q3  = compute_q3(test.y,  y_pred_test,  test.n);
    float per_class[3];
    int   conf[3][3];
    compute_per_class_accuracy(test.y, y_pred_test, test.n, per_class);
    compute_confusion_matrix(test.y,  y_pred_test, test.n, conf);

    double total_s = timer_elapsed_s(&total_timer);
    runlog_finalize(&log, train_q3, last_val_q3, test_q3, per_class, conf, total_s);
    runlog_write(&log, args.out_dir);

    printf("\n=== Final Results ===\n");
    printf("train Q3 = %.2f%%\n", train_q3);
    printf("val   Q3 = %.2f%%\n", last_val_q3);
    printf("test  Q3 = %.2f%%\n", test_q3);
    printf("per-class (H/E/C): %.2f%% / %.2f%% / %.2f%%\n",
           per_class[0], per_class[1], per_class[2]);
    printf("total time = %.1f s\n", total_s);

    /* --- Cleanup --- */
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_X_row); cudaFree(d_y_all);
    cudaFree(d_Xval_col); cudaFree(d_Xtest_col); cudaFree(d_Xtrain_col);
    cudaFree(d_X_batch); cudaFree(d_A1); cudaFree(d_A2); cudaFree(d_Z3);
    cudaFree(d_dA2); cudaFree(d_dA1); cudaFree(d_ones_B); cudaFree(d_loss);
    cudaFree(d_idx_batch); cudaFree(d_y_batch);
    cudaFree(d_predA1); cudaFree(d_predA2); cudaFree(d_predZ3);
    cublasDestroy(cublas);

    free(indices); free(y_pred_val); free(y_pred_train); free(y_pred_test);
    mlp_free(&m);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);
    (void)download_T;   /* suppress unused-warning; kept for future debugging */
    return 0;
}
