/* src/serial/train.c — Serial MLP training for protein secondary structure (Q3).
 *
 * Architecture: Input(260) -> Hidden1(ReLU) -> Hidden2(ReLU) -> Output(3, Softmax)
 * Optimiser:    Mini-batch SGD
 * Usage:        ./train_serial [--data PATH] [--epochs N] [--batch N]
 *                              [--lr F] [--seed N] [--hidden1 N] [--hidden2 N]
 *                              [--out DIR] [--verbose 0|1]
 */

#include "common/cli.h"
#include "common/data_loader.h"
#include "common/logger.h"
#include "common/metrics.h"
#include "common/timer.h"
#include "models/mlp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * Fisher-Yates shuffle with an LCG (same generator as mlp.c).
 * seed is combined with epoch so each epoch gets a different permutation.
 * ----------------------------------------------------------------------- */
static unsigned int shuf_state;

static void shuf_seed(unsigned int s) { shuf_state = s; }

static unsigned int shuf_rand(void)
{
    shuf_state = shuf_state * 1664525u + 1013904223u;
    return shuf_state;
}

static void shuffle(int *arr, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = (int)(shuf_rand() % (unsigned int)(i + 1));
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/* -----------------------------------------------------------------------
 * Main
 * ----------------------------------------------------------------------- */

int main(int argc, char **argv)
{
    Args args;
    parse_args(argc, argv, &args, "serial");

    if (args.verbose)
    {
        printf("=== train_serial ===\n");
        print_args(&args);
    }

    /* --- Load data --- */
    Dataset train, val, test;
    dataset_load(&train, args.data_path, "train");
    dataset_load(&val, args.data_path, "val");
    dataset_load(&test, args.data_path, "test");

    printf("Loaded: train=%d  val=%d  test=%d  feature_dim=%d\n",
           train.n, val.n, test.n, train.feature_dim);

    /* --- Set up run log --- */
    RunLog log;
    runlog_init(&log);
    strncpy(log.variant, "serial", sizeof(log.variant) - 1);
    snprintf(log.data_path, sizeof(log.data_path), "%s", args.data_path);
    log.seed = args.seed;
    log.epochs = args.epochs;
    log.batch_size = args.batch_size;
    log.learning_rate = args.lr;
    log.hidden1 = args.hidden1;
    log.hidden2 = args.hidden2;
    log.threads = 1;
    log.mpi_ranks = 1;
    runlog_set_dataset(&log, train.n, val.n, test.n, train.feature_dim, 3);
    runlog_set_compiler(&log, "gcc " __VERSION__, "-O3 -march=native");

    /* --- Initialise MLP --- */
    MLP m;
    mlp_init(&m, train.feature_dim, args.hidden1, args.hidden2, 3,
             (unsigned int)args.seed);

    MLPGrad g;
    mlpgrad_alloc(&g, &m);

    /* --- Allocate scratch buffers (reused every sample) --- */
    float *a1 = (float *)malloc((size_t)args.hidden1 * sizeof(float));
    float *a2 = (float *)malloc((size_t)args.hidden2 * sizeof(float));
    float *logits = (float *)malloc(3 * sizeof(float));
    float *probs = (float *)malloc(3 * sizeof(float));

    /* --- Prediction buffers --- */
    int *indices = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_val = (int *)malloc((size_t)val.n * sizeof(int));
    int *y_pred_train = (int *)malloc((size_t)train.n * sizeof(int));
    int *y_pred_test = (int *)malloc((size_t)test.n * sizeof(int));

    if (!a1 || !a2 || !logits || !probs || !indices ||
        !y_pred_val || !y_pred_train || !y_pred_test)
    {
        fprintf(stderr, "[train_serial] OOM\n");
        exit(1);
    }

    for (int i = 0; i < train.n; i++)
        indices[i] = i;

    /* --- Training loop --- */
    Timer total_timer;
    timer_start(&total_timer);

    float last_val_q3 = 0.0f;

    for (int epoch = 1; epoch <= args.epochs; epoch++)
    {
        /* Shuffle training indices (different seed per epoch) */
        shuf_seed((unsigned int)(args.seed + epoch));
        shuffle(indices, train.n);

        double epoch_loss = 0.0;
        Timer epoch_timer;
        timer_start(&epoch_timer);

        /* Mini-batch SGD */
        for (int b = 0; b < train.n; b += args.batch_size)
        {
            int actual = args.batch_size;
            if (b + actual > train.n)
                actual = train.n - b;

            mlpgrad_zero(&g, &m);

            for (int k = 0; k < actual; k++)
            {
                int idx = indices[b + k];
                const float *x = train.X + (size_t)idx * train.feature_dim;

                mlp_forward(&m, x, a1, a2, logits, probs);

                /* Accumulate cross-entropy loss */
                float p_true = probs[train.y[idx]];
                if (p_true < 1e-9f)
                    p_true = 1e-9f;
                epoch_loss -= logf(p_true);

                mlp_backward(&m, &g, x, a1, a2, probs, train.y[idx]);
            }

            mlp_sgd_update(&m, &g, args.lr, actual);
        }

        double epoch_time = timer_elapsed_s(&epoch_timer);
        float avg_loss = (float)(epoch_loss / train.n);

        /* Validation Q3 */
        mlp_predict(&m, val.X, val.n, val.feature_dim, y_pred_val);
        float val_q3 = compute_q3(val.y, y_pred_val, val.n);
        last_val_q3 = val_q3;

        runlog_add_epoch(&log, epoch, avg_loss, val_q3, epoch_time);

        if (args.verbose)
        {
            printf("Epoch %3d  loss=%.4f  val_q3=%6.2f%%  lr=%.6f  %.2fs\n",
                   epoch, avg_loss, val_q3, args.lr, epoch_time);
        }

        /* LR step decay */
        if (args.lr_decay != 1.0f && epoch % args.lr_decay_every == 0)
            args.lr *= args.lr_decay;
    }

    /* --- Final evaluation --- */
    mlp_predict(&m, train.X, train.n, train.feature_dim, y_pred_train);
    mlp_predict(&m, test.X, test.n, test.feature_dim, y_pred_test);

    float train_q3 = compute_q3(train.y, y_pred_train, train.n);
    float test_q3 = compute_q3(test.y, y_pred_test, test.n);
    float per_class[3];
    int conf[3][3];
    compute_per_class_accuracy(test.y, y_pred_test, test.n, per_class);
    compute_confusion_matrix(test.y, y_pred_test, test.n, conf);

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
    free(a1);
    free(a2);
    free(logits);
    free(probs);
    free(indices);
    free(y_pred_val);
    free(y_pred_train);
    free(y_pred_test);
    mlpgrad_free(&g);
    mlp_free(&m);
    dataset_free(&train);
    dataset_free(&val);
    dataset_free(&test);

    return 0;
}
