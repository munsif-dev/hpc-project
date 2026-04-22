/* src/hybrid/train.c — Hybrid MPI + OpenMP MLP training (data-parallel SGD).
 *
 * Two-level parallelism:
 *   OUTER (MPI):     ranks split each mini-batch into contiguous slices and
 *                    combine per-rank gradients with one MPI_Allreduce.
 *   INNER (OpenMP):  inside each rank's slice, OpenMP threads further split
 *                    the work, each accumulating into a thread-local MLPGrad,
 *                    then reducing serially into a per-rank gradient before
 *                    the cross-rank Allreduce.
 *
 * Layout:
 *   - Each rank owns N OpenMP-thread MLPGrad buffers (per-thread) PLUS one
 *     contiguous flat MLPGrad buffer (rank-level) used for Allreduce.
 *   - Per batch:
 *       1. Zero all per-thread grads + the flat rank grad.
 *       2. OpenMP parallel-for over the rank's slice -> per-thread grads.
 *       3. Serial reduce per-thread grads into thread-0's grad.
 *       4. Copy thread-0 grad -> flat rank grad (memcpy from union of arrays).
 *       5. MPI_Allreduce(SUM) on the flat rank grad.
 *       6. Apply mlp_sgd_update on every rank.
 *
 * Validation/test inference happens only on rank 0.
 *
 * Architecture: Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Usage:        mpirun -np R ./train_hybrid --threads T \
 *                       [--data PATH] [--epochs N] [--batch N]
 *                       [--lr F] [--seed N]
 *                       [--hidden1 N] [--hidden2 N]
 *                       [--out DIR] [--verbose 0|1]
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"
#include "models/mlp.h"

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Fisher-Yates shuffle — identical permutation on every rank.
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

/* Element-wise gradient sum (used inside a rank to reduce per-thread grads). */
static void grad_reduce(MLPGrad *dst, const MLPGrad *src, const MLP *m) {
    int H1 = m->hidden1, H2 = m->hidden2, IN = m->input_dim, OUT = m->output_dim;
    for (int i = 0; i < H1 * IN;  i++) dst->dW1[i] += src->dW1[i];
    for (int i = 0; i < H1;       i++) dst->db1[i] += src->db1[i];
    for (int i = 0; i < H2 * H1;  i++) dst->dW2[i] += src->dW2[i];
    for (int i = 0; i < H2;       i++) dst->db2[i] += src->db2[i];
    for (int i = 0; i < OUT * H2; i++) dst->dW3[i] += src->dW3[i];
    for (int i = 0; i < OUT;      i++) dst->db3[i] += src->db3[i];
}

/* Pack an MLPGrad into a flat contiguous buffer (for MPI_Allreduce). */
static void grad_pack(float *flat, const MLPGrad *g, const MLP *m) {
    size_t H1 = m->hidden1, H2 = m->hidden2, IN = m->input_dim, OUT = m->output_dim;
    size_t off = 0;
    memcpy(flat + off, g->dW1, H1*IN  * sizeof(float)); off += H1*IN;
    memcpy(flat + off, g->db1, H1     * sizeof(float)); off += H1;
    memcpy(flat + off, g->dW2, H2*H1  * sizeof(float)); off += H2*H1;
    memcpy(flat + off, g->db2, H2     * sizeof(float)); off += H2;
    memcpy(flat + off, g->dW3, OUT*H2 * sizeof(float)); off += OUT*H2;
    memcpy(flat + off, g->db3, OUT    * sizeof(float));
}

/* Unpack a flat buffer back into an MLPGrad. */
static void grad_unpack(MLPGrad *g, const float *flat, const MLP *m) {
    size_t H1 = m->hidden1, H2 = m->hidden2, IN = m->input_dim, OUT = m->output_dim;
    size_t off = 0;
    memcpy(g->dW1, flat + off, H1*IN  * sizeof(float)); off += H1*IN;
    memcpy(g->db1, flat + off, H1     * sizeof(float)); off += H1;
    memcpy(g->dW2, flat + off, H2*H1  * sizeof(float)); off += H2*H1;
    memcpy(g->db2, flat + off, H2     * sizeof(float)); off += H2;
    memcpy(g->dW3, flat + off, OUT*H2 * sizeof(float)); off += OUT*H2;
    memcpy(g->db3, flat + off, OUT    * sizeof(float));
}

static size_t flat_grad_size(const MLP *m) {
    size_t H1 = m->hidden1, H2 = m->hidden2, IN = m->input_dim, OUT = m->output_dim;
    return H1*IN + H1 + H2*H1 + H2 + OUT*H2 + OUT;
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args args;
    parse_args(argc, argv, &args, "hybrid");

    omp_set_num_threads(args.threads);

    if (args.verbose && rank == 0) {
        printf("=== train_hybrid ===\n");
        printf("  mpi_ranks   : %d\n", size);
        printf("  omp_threads : %d (per rank)\n", args.threads);
        print_args(&args);
    }

    /* --- Load data --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    if (rank == 0)
        printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
               train.n, val.n, test.n, train.feature_dim);

    /* --- Run log (rank 0) --- */
    RunLog log;
    if (rank == 0) {
        runlog_init(&log);
        strncpy(log.variant, "hybrid", sizeof(log.variant) - 1);
        snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
        log.seed          = args.seed;
        log.epochs        = args.epochs;
        log.batch_size    = args.batch_size;
        log.learning_rate = args.lr;
        log.hidden1       = args.hidden1;
        log.hidden2       = args.hidden2;
        log.threads       = args.threads;
        log.mpi_ranks     = size;
        runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
        runlog_set_compiler(&log, "mpicc " __VERSION__, "-O3 -march=native -fopenmp");
    }

    /* --- Identical model on every rank --- */
    MLP m;
    mlp_init(&m, train.feature_dim, args.hidden1, args.hidden2, 3,
             (unsigned int)args.seed);

    /* --- Per-thread gradient + scratch buffers --- */
    int nthreads = args.threads;
    if (nthreads < 1) nthreads = 1;
    MLPGrad *tgrads = (MLPGrad *)malloc((size_t)nthreads * sizeof(MLPGrad));
    if (!tgrads) { fprintf(stderr, "[rank %d] OOM tgrads\n", rank); MPI_Abort(MPI_COMM_WORLD,1); }
    for (int t = 0; t < nthreads; t++) mlpgrad_alloc(&tgrads[t], &m);

    int H1 = args.hidden1, H2 = args.hidden2;
    float **ta1     = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **ta2     = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **tlogits = (float **)malloc((size_t)nthreads * sizeof(float *));
    float **tprobs  = (float **)malloc((size_t)nthreads * sizeof(float *));
    for (int t = 0; t < nthreads; t++) {
        ta1[t]     = (float *)malloc((size_t)H1 * sizeof(float));
        ta2[t]     = (float *)malloc((size_t)H2 * sizeof(float));
        tlogits[t] = (float *)malloc(3 * sizeof(float));
        tprobs[t]  = (float *)malloc(3 * sizeof(float));
    }

    /* Flat rank-level gradient buffer for Allreduce */
    size_t gsize = flat_grad_size(&m);
    float *flat_grad = (float *)calloc(gsize, sizeof(float));
    if (!flat_grad) { fprintf(stderr, "[rank %d] OOM flat\n", rank); MPI_Abort(MPI_COMM_WORLD,1); }

    int *indices      = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_val   = (rank == 0) ? (int *)malloc((size_t)val.n   * sizeof(int)) : NULL;
    int *y_pred_train = (rank == 0) ? (int *)malloc((size_t)train.n * sizeof(int)) : NULL;
    int *y_pred_test  = (rank == 0) ? (int *)malloc((size_t)test.n  * sizeof(int)) : NULL;
    for (int i = 0; i < train.n; i++) indices[i] = i;

    /* --- Training loop --- */
    Timer total_timer;
    if (rank == 0) timer_start(&total_timer);

    float last_val_q3 = 0.0f;

    for (int epoch = 1; epoch <= args.epochs; epoch++) {
        shuf_seed((unsigned int)(args.seed + epoch));
        shuffle(indices, train.n);

        double local_epoch_loss = 0.0;
        MPI_Barrier(MPI_COMM_WORLD);
        double epoch_t0 = MPI_Wtime();

        for (int b = 0; b < train.n; b += args.batch_size) {
            int actual = args.batch_size;
            if (b + actual > train.n) actual = train.n - b;

            /* MPI partition: rank's slice */
            int rchunk = (actual + size - 1) / size;
            int rs = rank * rchunk;
            int re = rs + rchunk;
            if (rs > actual) rs = actual;
            if (re > actual) re = actual;
            int rank_local = re - rs;

            /* Zero per-thread grads */
            for (int t = 0; t < nthreads; t++) mlpgrad_zero(&tgrads[t], &m);

            double batch_loss = 0.0;

            /* OpenMP parallel over the rank's slice */
            #pragma omp parallel reduction(+:batch_loss)
            {
                int tid = omp_get_thread_num();
                MLPGrad *g    = &tgrads[tid];
                float *a1     = ta1[tid];
                float *a2     = ta2[tid];
                float *logits = tlogits[tid];
                float *probs  = tprobs[tid];

                #pragma omp for schedule(static)
                for (int k = 0; k < rank_local; k++) {
                    int idx = indices[b + rs + k];
                    const float *x = train.X + (size_t)idx * train.feature_dim;

                    mlp_forward(&m, x, a1, a2, logits, probs);

                    float p_true = probs[train.y[idx]];
                    if (p_true < 1e-9f) p_true = 1e-9f;
                    batch_loss -= (double)logf(p_true);

                    mlp_backward(&m, g, x, a1, a2, probs, train.y[idx]);
                }
            }
            local_epoch_loss += batch_loss;

            /* Reduce per-thread grads into tgrads[0] */
            for (int t = 1; t < nthreads; t++) grad_reduce(&tgrads[0], &tgrads[t], &m);

            /* Pack -> Allreduce -> Unpack */
            grad_pack(flat_grad, &tgrads[0], &m);
            MPI_Allreduce(MPI_IN_PLACE, flat_grad, (int)gsize,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            grad_unpack(&tgrads[0], flat_grad, &m);

            /* Synchronous SGD update on every rank */
            mlp_sgd_update(&m, &tgrads[0], args.lr, actual);
        }

        double epoch_t1 = MPI_Wtime();
        double local_epoch_time = epoch_t1 - epoch_t0;
        double max_epoch_time = 0.0;
        double total_epoch_loss = 0.0;
        MPI_Reduce(&local_epoch_time, &max_epoch_time, 1, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_epoch_loss, &total_epoch_loss, 1, MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            float avg_loss = (float)(total_epoch_loss / train.n);

            mlp_predict(&m, val.X, val.n, val.feature_dim, y_pred_val);
            float val_q3 = compute_q3(val.y, y_pred_val, val.n);
            last_val_q3 = val_q3;

            runlog_add_epoch(&log, epoch, avg_loss, val_q3, max_epoch_time);

            if (args.verbose) {
                printf("Epoch %3d  loss=%.4f  val_q3=%6.2f%%  lr=%.6f  %.2fs (max-rank)\n",
                       epoch, avg_loss, val_q3, args.lr, max_epoch_time);
            }
        }

        if (args.lr_decay != 1.0f && epoch % args.lr_decay_every == 0)
            args.lr *= args.lr_decay;
    }

    /* --- Final evaluation (rank 0) --- */
    if (rank == 0) {
        mlp_predict(&m, train.X, train.n, train.feature_dim, y_pred_train);
        mlp_predict(&m, test.X,  test.n,  test.feature_dim,  y_pred_test);

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
        printf("mpi_ranks = %d, omp_threads = %d (per rank)\n", size, args.threads);
        printf("train Q3  = %.2f%%\n", train_q3);
        printf("val   Q3  = %.2f%%\n", last_val_q3);
        printf("test  Q3  = %.2f%%\n", test_q3);
        printf("per-class (H/E/C): %.2f%% / %.2f%% / %.2f%%\n",
               per_class[0], per_class[1], per_class[2]);
        printf("total time = %.1f s\n", total_s);
    }

    /* --- Cleanup --- */
    for (int t = 0; t < nthreads; t++) {
        mlpgrad_free(&tgrads[t]);
        free(ta1[t]); free(ta2[t]); free(tlogits[t]); free(tprobs[t]);
    }
    free(tgrads); free(ta1); free(ta2); free(tlogits); free(tprobs);
    free(flat_grad);
    free(indices);
    if (rank == 0) { free(y_pred_val); free(y_pred_train); free(y_pred_test); }
    mlp_free(&m);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);

    MPI_Finalize();
    return 0;
}
