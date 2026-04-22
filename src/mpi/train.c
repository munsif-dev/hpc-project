/* src/mpi/train.c — Distributed-memory MPI MLP training (data-parallel SGD).
 *
 * Parallel strategy:
 *   - All ranks load the full dataset and initialise an identical model
 *     (deterministic from --seed, so no broadcast of weights is needed).
 *   - Every rank executes the same shuffled order each epoch (same seed).
 *   - For every mini-batch, rank r processes a contiguous slice of the batch
 *     [r*chunk, (r+1)*chunk), accumulates a LOCAL gradient sum.
 *   - One MPI_Allreduce(SUM) over the flattened gradient buffer combines
 *     gradients across all ranks.
 *   - Every rank then applies the same mlp_sgd_update -> models stay in sync.
 *   - Validation/test inference and JSON logging happen only on rank 0.
 *
 * Optimisation notes:
 *   - The MLPGrad arrays are backed by ONE contiguous flat buffer so the whole
 *     gradient state can be exchanged with a single MPI_Allreduce per batch
 *     (instead of six per-array calls).
 *   - Batch loss is summed locally across the epoch and Allreduced ONCE per
 *     epoch, not per batch.
 *   - MPI_Barrier surrounds the epoch timer; per-rank epoch time is reduced
 *     with MPI_MAX so reported time reflects the slowest rank.
 *
 * Architecture: Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Usage:        mpirun -np N ./train_mpi [--data PATH] [--epochs N] [--batch N]
 *                                        [--lr F] [--seed N]
 *                                        [--hidden1 N] [--hidden2 N]
 *                                        [--out DIR] [--verbose 0|1]
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"
#include "models/mlp.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Fisher-Yates shuffle (same LCG as serial/openmp/pthreads — every rank
 * runs this independently with the same seed -> identical permutations).
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
 * Compute total flat gradient size for the model.
 * ----------------------------------------------------------------------- */
static size_t flat_grad_size(const MLP *m) {
    size_t H1 = (size_t)m->hidden1, H2 = (size_t)m->hidden2;
    size_t IN = (size_t)m->input_dim, OUT = (size_t)m->output_dim;
    return H1*IN + H1 + H2*H1 + H2 + OUT*H2 + OUT;
}

/* Lay out an MLPGrad on a caller-provided contiguous flat buffer. */
static void flat_grad_bind(MLPGrad *g, float *flat, const MLP *m) {
    size_t H1 = (size_t)m->hidden1, H2 = (size_t)m->hidden2;
    size_t IN = (size_t)m->input_dim, OUT = (size_t)m->output_dim;
    g->dW1 = flat;
    g->db1 = g->dW1 + H1*IN;
    g->dW2 = g->db1 + H1;
    g->db2 = g->dW2 + H2*H1;
    g->dW3 = g->db2 + H2;
    g->db3 = g->dW3 + OUT*H2;
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args args;
    parse_args(argc, argv, &args, "mpi");

    if (args.verbose && rank == 0) {
        printf("=== train_mpi ===\n");
        printf("  mpi_ranks   : %d\n", size);
        print_args(&args);
    }

    /* --- Load data (every rank loads the full dataset) --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val,   args.data_path, "val");
    dataset_load(&test,  args.data_path, "test");

    if (rank == 0)
        printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
               train.n, val.n, test.n, train.feature_dim);

    /* --- Run log (rank 0 only) --- */
    RunLog log;
    if (rank == 0) {
        runlog_init(&log);
        strncpy(log.variant, "mpi", sizeof(log.variant) - 1);
        snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
        log.seed          = args.seed;
        log.epochs        = args.epochs;
        log.batch_size    = args.batch_size;
        log.learning_rate = args.lr;
        log.hidden1       = args.hidden1;
        log.hidden2       = args.hidden2;
        log.threads       = 1;
        log.mpi_ranks     = size;
        runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
        runlog_set_compiler(&log, "mpicc " __VERSION__, "-O3 -march=native");
    }

    /* --- Model: identical on every rank because mlp_init is deterministic --- */
    MLP m;
    mlp_init(&m, train.feature_dim, args.hidden1, args.hidden2, 3,
             (unsigned int)args.seed);

    /* --- Gradient buffer: contiguous flat layout for single Allreduce --- */
    size_t gsize = flat_grad_size(&m);
    float *flat_grad = (float *)calloc(gsize, sizeof(float));
    if (!flat_grad) { fprintf(stderr, "[rank %d] OOM grad\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    MLPGrad g;
    flat_grad_bind(&g, flat_grad, &m);

    /* --- Per-sample scratch --- */
    float *a1     = (float *)malloc((size_t)args.hidden1 * sizeof(float));
    float *a2     = (float *)malloc((size_t)args.hidden2 * sizeof(float));
    float *logits = (float *)malloc(3 * sizeof(float));
    float *probs  = (float *)malloc(3 * sizeof(float));

    int *indices      = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_val   = (rank == 0) ? (int *)malloc((size_t)val.n   * sizeof(int)) : NULL;
    int *y_pred_train = (rank == 0) ? (int *)malloc((size_t)train.n * sizeof(int)) : NULL;
    int *y_pred_test  = (rank == 0) ? (int *)malloc((size_t)test.n  * sizeof(int)) : NULL;

    if (!a1 || !a2 || !logits || !probs || !indices) {
        fprintf(stderr, "[rank %d] OOM scratch\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < train.n; i++) indices[i] = i;

    /* --- Training loop --- */
    Timer total_timer;
    if (rank == 0) timer_start(&total_timer);

    float last_val_q3 = 0.0f;

    for (int epoch = 1; epoch <= args.epochs; epoch++) {
        /* Identical shuffle on every rank */
        shuf_seed((unsigned int)(args.seed + epoch));
        shuffle(indices, train.n);

        double local_epoch_loss = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double epoch_t0 = MPI_Wtime();

        for (int b = 0; b < train.n; b += args.batch_size) {
            int actual = args.batch_size;
            if (b + actual > train.n) actual = train.n - b;

            /* Partition the mini-batch across ranks (contiguous slice). */
            int chunk = (actual + size - 1) / size;
            int s = rank * chunk;
            int e = s + chunk;
            if (s > actual) s = actual;
            if (e > actual) e = actual;

            /* Zero local gradient buffer */
            memset(flat_grad, 0, gsize * sizeof(float));

            /* Local accumulation over rank's slice */
            for (int k = s; k < e; k++) {
                int idx = indices[b + k];
                const float *x = train.X + (size_t)idx * train.feature_dim;

                mlp_forward(&m, x, a1, a2, logits, probs);

                float p_true = probs[train.y[idx]];
                if (p_true < 1e-9f) p_true = 1e-9f;
                local_epoch_loss -= (double)logf(p_true);

                mlp_backward(&m, &g, x, a1, a2, probs, train.y[idx]);
            }

            /* Sum gradients across all ranks. Single Allreduce on the
             * flat buffer is much cheaper than six per-array calls. */
            MPI_Allreduce(MPI_IN_PLACE, flat_grad, (int)gsize,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            /* Every rank applies the same update -> models stay in sync. */
            mlp_sgd_update(&m, &g, args.lr, actual);
        }

        /* Reduce timing (max across ranks) and loss (sum) */
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

            /* Validation only on rank 0 */
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
        printf("mpi_ranks = %d\n", size);
        printf("train Q3  = %.2f%%\n", train_q3);
        printf("val   Q3  = %.2f%%\n", last_val_q3);
        printf("test  Q3  = %.2f%%\n", test_q3);
        printf("per-class (H/E/C): %.2f%% / %.2f%% / %.2f%%\n",
               per_class[0], per_class[1], per_class[2]);
        printf("total time = %.1f s\n", total_s);
    }

    /* --- Cleanup --- */
    free(flat_grad);
    free(a1); free(a2); free(logits); free(probs);
    free(indices);
    if (rank == 0) { free(y_pred_val); free(y_pred_train); free(y_pred_test); }
    mlp_free(&m);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);

    MPI_Finalize();
    return 0;
}
