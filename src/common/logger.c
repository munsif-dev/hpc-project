#include "common/logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

void runlog_init(RunLog *log) {
    memset(log, 0, sizeof(RunLog));
    log->mpi_ranks   = 1;
    log->num_classes = 3;
    log->finalized   = 0;
    log->epoch_count = 0;
    gethostname(log->hostname, sizeof(log->hostname) - 1);
    strncpy(log->compiler, "unknown", sizeof(log->compiler) - 1);
    strncpy(log->flags,    "-O3 -march=native", sizeof(log->flags) - 1);
}

void runlog_set_dataset(RunLog *log, int n_train, int n_val, int n_test,
                        int feature_dim, int num_classes) {
    log->n_train     = n_train;
    log->n_val       = n_val;
    log->n_test      = n_test;
    log->feature_dim = feature_dim;
    log->num_classes = num_classes;
}

void runlog_set_compiler(RunLog *log, const char *compiler, const char *flags) {
    strncpy(log->compiler, compiler, sizeof(log->compiler) - 1);
    strncpy(log->flags,    flags,    sizeof(log->flags)    - 1);
}

void runlog_add_epoch(RunLog *log, int epoch, float train_loss,
                      float val_q3, double epoch_time_s) {
    if (log->epoch_count >= LOG_MAX_EPOCHS) return;
    EpochEntry *e = &log->epoch_log[log->epoch_count++];
    e->epoch        = epoch;
    e->train_loss   = train_loss;
    e->val_q3       = val_q3;
    e->epoch_time_s = epoch_time_s;
}

void runlog_finalize(RunLog *log, float train_q3, float val_q3, float test_q3,
                     const float per_class[3], const int conf[3][3],
                     double total_time_s) {
    log->train_q3    = train_q3;
    log->val_q3      = val_q3;
    log->test_q3     = test_q3;
    log->total_time_s = total_time_s;
    for (int i = 0; i < 3; i++) {
        log->per_class_acc[i] = per_class[i];
        for (int j = 0; j < 3; j++)
            log->confusion[i][j] = conf[i][j];
    }
    log->finalized = 1;
}

/* mkdir -p equivalent (single level, ignores EEXIST). */
static void mkdir_p(const char *path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0755);
#endif
}

void runlog_write(const RunLog *log, const char *out_dir) {
    mkdir_p(out_dir);

    /* Build filename with the full execution shape to avoid fast-run collisions. */
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/%s_r%d_t%d_b%d_s%d_%s.json",
             out_dir, log->variant, log->mpi_ranks, log->threads,
             log->batch_size, log->seed, timestamp);

    FILE *f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "[logger] Cannot write log to %s\n", filepath);
        return;
    }

    /* --- Root object --- */
    fprintf(f, "{\n");
    fprintf(f, "  \"variant\": \"%s\",\n",      log->variant);
    fprintf(f, "  \"seed\": %d,\n",             log->seed);
    fprintf(f, "  \"epochs\": %d,\n",           log->epochs);
    fprintf(f, "  \"batch_size\": %d,\n",       log->batch_size);
    fprintf(f, "  \"learning_rate\": %.6f,\n",  log->learning_rate);
    fprintf(f, "  \"hidden1\": %d,\n",          log->hidden1);
    fprintf(f, "  \"hidden2\": %d,\n",          log->hidden2);
    fprintf(f, "  \"threads\": %d,\n",          log->threads);
    fprintf(f, "  \"mpi_ranks\": %d,\n",        log->mpi_ranks);
    fprintf(f, "  \"data_path\": \"%s\",\n",    log->data_path);
    fprintf(f, "  \"n_train\": %d,\n",          log->n_train);
    fprintf(f, "  \"n_val\": %d,\n",            log->n_val);
    fprintf(f, "  \"n_test\": %d,\n",           log->n_test);
    fprintf(f, "  \"feature_dim\": %d,\n",      log->feature_dim);
    fprintf(f, "  \"num_classes\": %d,\n",      log->num_classes);

    /* Hardware block */
    fprintf(f, "  \"hardware\": {\n");
    fprintf(f, "    \"hostname\": \"%s\",\n",   log->hostname);
    fprintf(f, "    \"compiler\": \"%s\",\n",   log->compiler);
    fprintf(f, "    \"flags\": \"%s\"\n",       log->flags);
    fprintf(f, "  },\n");

    /* Epoch log array */
    fprintf(f, "  \"epoch_log\": [\n");
    for (int i = 0; i < log->epoch_count; i++) {
        const EpochEntry *e = &log->epoch_log[i];
        fprintf(f, "    {\"epoch\": %d, \"train_loss\": %.6f, \"val_q3\": %.4f, \"epoch_time_s\": %.4f}%s\n",
                e->epoch, e->train_loss, e->val_q3, e->epoch_time_s,
                (i < log->epoch_count - 1) ? "," : "");
    }
    fprintf(f, "  ],\n");

    /* Final metrics */
    fprintf(f, "  \"final\": {\n");
    fprintf(f, "    \"train_q3\": %.4f,\n",  log->train_q3);
    fprintf(f, "    \"val_q3\": %.4f,\n",    log->val_q3);
    fprintf(f, "    \"test_q3\": %.4f,\n",   log->test_q3);
    fprintf(f, "    \"per_class_accuracy\": {\"H\": %.4f, \"E\": %.4f, \"C\": %.4f},\n",
            log->per_class_acc[0], log->per_class_acc[1], log->per_class_acc[2]);
    fprintf(f, "    \"confusion_matrix\": [\n");
    for (int r = 0; r < 3; r++) {
        fprintf(f, "      [%d, %d, %d]%s\n",
                log->confusion[r][0], log->confusion[r][1], log->confusion[r][2],
                (r < 2) ? "," : "");
    }
    fprintf(f, "    ],\n");
    fprintf(f, "    \"total_time_s\": %.4f\n", log->total_time_s);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("[logger] Log written to %s\n", filepath);
}
