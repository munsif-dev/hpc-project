#ifndef LOGGER_H
#define LOGGER_H

/* JSON run-log for one training experiment.
 *
 * Naming convention for output files:
 *   results/<variant>/<variant>_r<ranks>_t<threads>_b<batch>_s<seed>_<YYYYMMDD_HHMMSS>.json
 *
 * Schema (all fields written by runlog_write):
 *   variant, seed, epochs, batch_size, learning_rate, hidden1, hidden2,
 *   threads, mpi_ranks, data_path, n_train, n_val, n_test, feature_dim,
 *   num_classes, hardware{hostname, compiler, flags},
 *   epoch_log[{epoch, train_loss, val_q3, epoch_time_s}],
 *   final{train_q3, val_q3, test_q3, per_class_accuracy{H,E,C},
 *         confusion_matrix[3][3], total_time_s}
 *
 * Maximum epochs supported: LOG_MAX_EPOCHS.
 */

#define LOG_MAX_EPOCHS 2000

typedef struct {
    int    epoch;
    float  train_loss;
    float  val_q3;
    double epoch_time_s;
} EpochEntry;

typedef struct {
    /* Config */
    char  variant[32];
    int   seed, epochs, batch_size, threads, mpi_ranks;
    float learning_rate;
    int   hidden1, hidden2;
    char  data_path[256];

    /* Dataset sizes (filled by runlog_set_dataset) */
    int n_train, n_val, n_test, feature_dim, num_classes;

    /* Hardware (auto-filled by runlog_init) */
    char hostname[128];
    char compiler[128];  /* set to compile-time string via runlog_set_compiler */
    char flags[256];

    /* Per-epoch entries */
    EpochEntry epoch_log[LOG_MAX_EPOCHS];
    int        epoch_count;

    /* Final results (filled by runlog_finalize) */
    float  train_q3, val_q3, test_q3;
    float  per_class_acc[3];   /* H, E, C */
    int    confusion[3][3];    /* [true][pred] */
    double total_time_s;
    int    finalized;
} RunLog;

/* Initialise a RunLog to defaults and capture hostname. */
void runlog_init(RunLog *log);

/* Set dataset counts (call after dataset_load). */
void runlog_set_dataset(RunLog *log, int n_train, int n_val, int n_test,
                        int feature_dim, int num_classes);

/* Set compiler string and flags (use __VERSION__ and compile-time macro). */
void runlog_set_compiler(RunLog *log, const char *compiler, const char *flags);

/* Append one epoch's metrics. */
void runlog_add_epoch(RunLog *log, int epoch, float train_loss,
                      float val_q3, double epoch_time_s);

/* Fill final metrics after test evaluation. */
void runlog_finalize(RunLog *log, float train_q3, float val_q3, float test_q3,
                     const float per_class[3], const int conf[3][3],
                     double total_time_s);

/* Write JSON log to
 * out_dir/<variant>_r<ranks>_t<threads>_b<batch>_s<seed>_<timestamp>.json.
 * Creates out_dir if it does not exist. */
void runlog_write(const RunLog *log, const char *out_dir);

#endif /* LOGGER_H */
