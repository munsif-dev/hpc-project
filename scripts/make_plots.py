#!/usr/bin/env python3
"""Generate report figures and CSV summaries from JSON run logs.

Input:
  results/{serial,openmp,pthreads,mpi,hybrid,cuda}/*.json

Output:
  plots/timing_summary.csv       grouped mean/std per configuration
  plots/accuracy_table.csv       representative accuracy row per variant
  plots/time_vs_threads.png      OpenMP and pthreads time per epoch
  plots/speedup_vs_threads.png   OpenMP and pthreads speedup
  plots/time_vs_ranks_mpi.png    MPI rank scaling
  plots/hybrid_heatmap.png       hybrid rank x thread timing
  plots/cuda_time_vs_batch.png   CUDA batch-size timing

The timing plots prefer 20-epoch runs when they exist, because the project
uses 20 epochs for timing sweeps and 80 epochs for final accuracy runs.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import Counter, defaultdict
from statistics import mean, stdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = "results"
OUT = "plots"
VARIANTS = ("serial", "openmp", "pthreads", "mpi", "hybrid", "cuda")

os.makedirs(OUT, exist_ok=True)


def load_runs(variant: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(f"{RESULTS}/{variant}/*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"  [skip] {path}: {exc}")
            continue

        epochs = data.get("epoch_log") or []
        if not epochs:
            continue

        epoch_times = [float(entry["epoch_time_s"]) for entry in epochs]
        final = data.get("final", {})
        runs.append(
            {
                "file": path,
                "variant": data.get("variant", variant),
                "seed": data.get("seed"),
                "epochs": data.get("epochs"),
                "threads": data.get("threads", 1),
                "ranks": data.get("mpi_ranks", 1),
                "batch": data.get("batch_size"),
                "lr": data.get("learning_rate"),
                "hidden1": data.get("hidden1"),
                "hidden2": data.get("hidden2"),
                "avg_epoch_s": mean(epoch_times),
                "total_s": final.get("total_time_s"),
                "test_q3": final.get("test_q3"),
                "train_q3": final.get("train_q3"),
                "val_q3": final.get("val_q3"),
                "val_q3_last": epochs[-1].get("val_q3"),
            }
        )
    return runs


def config_key(run: dict) -> tuple:
    return (
        run["variant"],
        run["epochs"],
        run["threads"],
        run["ranks"],
        run["batch"],
        run["lr"],
        run["hidden1"],
        run["hidden2"],
        run["seed"],
    )


def summarize_runs(runs_by_variant: dict[str, list[dict]]) -> list[dict]:
    grouped = defaultdict(list)
    for runs in runs_by_variant.values():
        for run in runs:
            grouped[config_key(run)].append(run)

    rows = []
    for key, runs in sorted(grouped.items()):
        (
            variant,
            epochs,
            threads,
            ranks,
            batch,
            lr,
            hidden1,
            hidden2,
            seed,
        ) = key
        epoch_values = [run["avg_epoch_s"] for run in runs]
        total_values = [run["total_s"] for run in runs if run["total_s"] is not None]
        test_values = [run["test_q3"] for run in runs if run["test_q3"] is not None]
        train_values = [run["train_q3"] for run in runs if run["train_q3"] is not None]
        val_values = [run["val_q3"] for run in runs if run["val_q3"] is not None]

        rows.append(
            {
                "variant": variant,
                "epochs": epochs,
                "threads": threads,
                "ranks": ranks,
                "batch": batch,
                "lr": lr,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "seed": seed,
                "repeat_count": len(runs),
                "mean_epoch_s": mean(epoch_values),
                "std_epoch_s": stdev(epoch_values) if len(epoch_values) > 1 else 0.0,
                "min_epoch_s": min(epoch_values),
                "max_epoch_s": max(epoch_values),
                "mean_total_s": mean(total_values) if total_values else None,
                "mean_train_q3": mean(train_values) if train_values else None,
                "mean_val_q3": mean(val_values) if val_values else None,
                "mean_test_q3": mean(test_values) if test_values else None,
                "files": ";".join(run["file"] for run in runs),
            }
        )
    return rows


def write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    print(f"  wrote {path}")


def placeholder_plot(filename: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    out = f"{OUT}/{filename}"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out} (placeholder)")


def timing_epoch_for(rows: list[dict], variant: str) -> int | None:
    epochs = [row["epochs"] for row in rows if row["variant"] == variant and row["epochs"]]
    timing_epochs = [epoch for epoch in epochs if epoch < 80]
    if not timing_epochs:
        return max(epochs) if epochs else None
    counts = Counter(timing_epochs)
    if counts.get(20):
        return 20
    return sorted(counts.items(), key=lambda item: (item[1], item[0]))[-1][0]


def select_timing(rows: list[dict], variant: str) -> list[dict]:
    epoch = timing_epoch_for(rows, variant)
    if epoch is None:
        return []
    return [row for row in rows if row["variant"] == variant and row["epochs"] == epoch]


def best_by(rows: list[dict], field: str) -> dict:
    best = {}
    for row in rows:
        key = row[field]
        if key not in best or row["mean_epoch_s"] < best[key]["mean_epoch_s"]:
            best[key] = row
    return best


def plot_time_vs_threads(summary: list[dict]) -> None:
    omp = best_by(select_timing(summary, "openmp"), "threads")
    pthreads = best_by(select_timing(summary, "pthreads"), "threads")
    if not omp and not pthreads:
        print("  [skip] shared-memory plots - no OpenMP/Pthreads data")
        placeholder_plot("time_vs_threads.png", "Shared-memory timing", "No OpenMP/Pthreads timing data found.")
        placeholder_plot("speedup_vs_threads.png", "Shared-memory speedup", "No OpenMP/Pthreads timing data found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, data, marker in (
        ("OpenMP", omp, "o-"),
        ("Pthreads", pthreads, "s-"),
    ):
        if not data:
            continue
        keys = sorted(data)
        ax.errorbar(
            keys,
            [data[key]["mean_epoch_s"] for key in keys],
            yerr=[data[key]["std_epoch_s"] for key in keys],
            fmt=marker,
            capsize=4,
            label=label,
        )

    ax.set_xlabel("threads")
    ax.set_ylabel("mean time per epoch (s)")
    ax.set_xscale("log", base=2)
    ax.set_title("Training time per epoch vs threads")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = f"{OUT}/time_vs_threads.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_speedup(summary: list[dict]) -> None:
    omp = best_by(select_timing(summary, "openmp"), "threads")
    pthreads = best_by(select_timing(summary, "pthreads"), "threads")
    if not omp and not pthreads:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    all_threads = sorted(set(omp) | set(pthreads))
    for label, data, marker in (
        ("OpenMP", omp, "o-"),
        ("Pthreads", pthreads, "s-"),
    ):
        if not data or 1 not in data:
            continue
        base = data[1]["mean_epoch_s"]
        keys = sorted(data)
        ax.plot(keys, [base / data[key]["mean_epoch_s"] for key in keys], marker, label=label)
        plotted = True

    if plotted:
        ax.plot(all_threads, all_threads, "k--", alpha=0.45, label="ideal")
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs 1 thread")
        ax.set_xscale("log", base=2)
        ax.set_title("Shared-memory speedup")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = f"{OUT}/speedup_vs_threads.png"
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")
    plt.close(fig)


def plot_time_vs_ranks(summary: list[dict]) -> None:
    mpi = best_by(select_timing(summary, "mpi"), "ranks")
    if not mpi:
        print("  [skip] MPI plot - no MPI data")
        placeholder_plot("time_vs_ranks_mpi.png", "MPI rank scaling", "No MPI timing data found yet.")
        return

    keys = sorted(mpi)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(key) for key in keys], [mpi[key]["mean_epoch_s"] for key in keys], color="#4472C4")
    ax.errorbar(
        [str(key) for key in keys],
        [mpi[key]["mean_epoch_s"] for key in keys],
        yerr=[mpi[key]["std_epoch_s"] for key in keys],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("mean time per epoch (s)")
    ax.set_title("MPI training time per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/time_vs_ranks_mpi.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_hybrid(summary: list[dict]) -> None:
    rows = select_timing(summary, "hybrid")
    if not rows:
        print("  [skip] hybrid plot - no hybrid data")
        placeholder_plot("hybrid_heatmap.png", "Hybrid MPI+OpenMP timing", "No hybrid timing data found yet.")
        return

    best = {}
    for row in rows:
        key = (row["ranks"], row["threads"])
        if key not in best or row["mean_epoch_s"] < best[key]["mean_epoch_s"]:
            best[key] = row

    keys = sorted(best)
    labels = [f"{ranks}x{threads}" for ranks, threads in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [best[key]["mean_epoch_s"] for key in keys], color="#70AD47")
    ax.errorbar(
        labels,
        [best[key]["mean_epoch_s"] for key in keys],
        yerr=[best[key]["std_epoch_s"] for key in keys],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_xlabel("ranks x threads per rank")
    ax.set_ylabel("mean time per epoch (s)")
    ax.set_title("Hybrid MPI+OpenMP time per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/hybrid_heatmap.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_cuda(summary: list[dict]) -> None:
    cuda = best_by(select_timing(summary, "cuda"), "batch")
    if not cuda:
        print("  [skip] CUDA plot - no CUDA data")
        placeholder_plot("cuda_time_vs_batch.png", "CUDA batch-size timing", "No CUDA timing data found yet.")
        return

    keys = sorted(cuda)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        keys,
        [cuda[key]["mean_epoch_s"] for key in keys],
        yerr=[cuda[key]["std_epoch_s"] for key in keys],
        fmt="d-",
        capsize=4,
        color="#C00000",
    )
    ax.set_xlabel("batch size")
    ax.set_ylabel("mean time per epoch (s)")
    ax.set_xscale("log", base=2)
    ax.set_title("CUDA training time per epoch vs batch size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/cuda_time_vs_batch.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def write_accuracy_table(summary: list[dict]) -> list[dict]:
    by_variant = defaultdict(list)
    for row in summary:
        if row["mean_test_q3"] is not None:
            by_variant[row["variant"]].append(row)

    selected = {}
    for variant, rows in by_variant.items():
        max_epochs = max(row["epochs"] or 0 for row in rows)
        candidates = [row for row in rows if row["epochs"] == max_epochs]
        selected[variant] = min(candidates, key=lambda row: row["mean_epoch_s"])

    serial_q3 = selected.get("serial", {}).get("mean_test_q3")
    rows = []
    for variant in VARIANTS:
        row = selected.get(variant)
        if not row:
            continue
        delta = None if serial_q3 is None else row["mean_test_q3"] - serial_q3
        tolerance = 1.0 if variant == "cuda" else 0.5
        if variant == "serial" or delta is None:
            status = "reference"
        else:
            status = "pass" if abs(delta) <= tolerance else "check"
        rows.append(
            {
                "variant": variant,
                "epochs": row["epochs"],
                "threads": row["threads"],
                "ranks": row["ranks"],
                "batch": row["batch"],
                "repeat_count": row["repeat_count"],
                "test_q3": f"{row['mean_test_q3']:.4f}",
                "delta_vs_serial": "" if delta is None else f"{delta:.4f}",
                "accuracy_status": status,
                "mean_epoch_s": f"{row['mean_epoch_s']:.4f}",
                "std_epoch_s": f"{row['std_epoch_s']:.4f}",
                "mean_total_s": "" if row["mean_total_s"] is None else f"{row['mean_total_s']:.4f}",
            }
        )

    fields = [
        "variant",
        "epochs",
        "threads",
        "ranks",
        "batch",
        "repeat_count",
        "test_q3",
        "delta_vs_serial",
        "accuracy_status",
        "mean_epoch_s",
        "std_epoch_s",
        "mean_total_s",
    ]
    write_csv(f"{OUT}/accuracy_table.csv", rows, fields)
    return rows


def main() -> None:
    runs = {variant: load_runs(variant) for variant in VARIANTS}
    for variant, variant_runs in runs.items():
        print(f"{variant:10s}: {len(variant_runs)} run(s)")

    summary = summarize_runs(runs)
    summary_fields = [
        "variant",
        "epochs",
        "threads",
        "ranks",
        "batch",
        "lr",
        "hidden1",
        "hidden2",
        "seed",
        "repeat_count",
        "mean_epoch_s",
        "std_epoch_s",
        "min_epoch_s",
        "max_epoch_s",
        "mean_total_s",
        "mean_train_q3",
        "mean_val_q3",
        "mean_test_q3",
        "files",
    ]

    print("\n--- summaries ---")
    write_csv(f"{OUT}/timing_summary.csv", summary, summary_fields)

    print("\n--- plots ---")
    plot_time_vs_threads(summary)
    plot_speedup(summary)
    plot_time_vs_ranks(summary)
    plot_hybrid(summary)
    plot_cuda(summary)

    print("\n--- accuracy table ---")
    accuracy_rows = write_accuracy_table(summary)
    for row in accuracy_rows:
        print(
            f"  {row['variant']:10s} ep={row['epochs']:>3} "
            f"t={row['threads']} r={row['ranks']} b={row['batch']} "
            f"test_q3={row['test_q3']} status={row['accuracy_status']}"
        )


if __name__ == "__main__":
    main()
