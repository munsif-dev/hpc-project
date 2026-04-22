#!/usr/bin/env python3
"""scripts/make_plots.py — generate the figures needed for the report.

Reads all JSON run logs under results/{serial,openmp,pthreads,mpi,hybrid,cuda}/
and produces:
  - plots/time_vs_threads.png       : time/epoch for OpenMP and pthreads
  - plots/speedup_vs_threads.png    : speedup relative to 1-thread run of the
                                       same variant
  - plots/time_vs_ranks_mpi.png     : time/epoch for MPI ranks
  - plots/hybrid_heatmap.png        : time/epoch for hybrid MPI x OMP matrix
  - plots/cuda_time_vs_batch.png    : time/epoch for CUDA batch sizes
  - plots/accuracy_table.csv        : final test_q3 for every variant

Only plots where data exists are generated; missing variants are skipped with
a printed warning. Usage:   python3 scripts/make_plots.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = "results"
OUT = "plots"
os.makedirs(OUT, exist_ok=True)


def load_runs(variant: str):
    out = []
    for f in sorted(glob.glob(f"{RESULTS}/{variant}/*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [skip] {f}: {e}")
            continue
        epochs = d.get("epoch_log") or []
        if not epochs:
            continue
        avg_ep = sum(e["epoch_time_s"] for e in epochs) / len(epochs)
        out.append({
            "file": f,
            "variant": d.get("variant", variant),
            "threads": d.get("threads", 1),
            "ranks": d.get("mpi_ranks", 1),
            "batch": d.get("batch_size"),
            "epochs": d.get("epochs"),
            "avg_ep_s": avg_ep,
            "total_s": d.get("final", {}).get("total_time_s"),
            "test_q3": d.get("final", {}).get("test_q3"),
            "train_q3": d.get("final", {}).get("train_q3"),
            "val_q3_last": epochs[-1].get("val_q3"),
        })
    return out


def best_per_threads(runs, key):
    """Return dict[threads] -> run, keeping the best (lowest avg_ep_s) match."""
    best = {}
    for r in runs:
        k = r[key]
        if k not in best or r["avg_ep_s"] < best[k]["avg_ep_s"]:
            best[k] = r
    return best


def plot_time_vs_threads(omp, pth):
    fig, ax = plt.subplots(figsize=(7, 5))
    if omp:
        ks = sorted(omp.keys())
        ax.plot(ks, [omp[k]["avg_ep_s"] for k in ks], "o-", label="OpenMP")
    if pth:
        ks = sorted(pth.keys())
        ax.plot(ks, [pth[k]["avg_ep_s"] for k in ks], "s-", label="Pthreads")
    ax.set_xlabel("threads")
    ax.set_ylabel("time per epoch (s)")
    ax.set_xscale("log", base=2)
    ax.set_title("Training time per epoch vs threads (CB513 MLP 256/128)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = f"{OUT}/time_vs_threads.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_speedup(omp, pth):
    fig, ax = plt.subplots(figsize=(7, 5))
    any_plotted = False
    for label, data, marker in (("OpenMP", omp, "o-"), ("Pthreads", pth, "s-")):
        if not data or 1 not in data:
            continue
        base = data[1]["avg_ep_s"]
        ks = sorted(data.keys())
        sp = [base / data[k]["avg_ep_s"] for k in ks]
        ax.plot(ks, sp, marker, label=label)
        any_plotted = True
    if any_plotted:
        ks_all = sorted(set(list(omp.keys()) + list(pth.keys())))
        if ks_all:
            ax.plot(ks_all, ks_all, "k--", label="ideal (linear)", alpha=0.5)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs 1 thread")
        ax.set_xscale("log", base=2)
        ax.set_title("Speedup vs threads")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = f"{OUT}/speedup_vs_threads.png"
        fig.savefig(out, dpi=130)
        print(f"  wrote {out}")
    plt.close(fig)


def plot_time_vs_ranks(mpi):
    if not mpi:
        print("  [skip] MPI — no data")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(mpi.keys())
    ax.bar([str(k) for k in ks], [mpi[k]["avg_ep_s"] for k in ks], color="#4472C4")
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("time per epoch (s)")
    ax.set_title("MPI training time per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/time_vs_ranks_mpi.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_hybrid(runs):
    if not runs:
        print("  [skip] hybrid — no data")
        return
    by_rt = {}
    for r in runs:
        by_rt[(r["ranks"], r["threads"])] = r["avg_ep_s"]
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{r}x{t}" for (r, t) in sorted(by_rt.keys())]
    vals = [by_rt[k] for k in sorted(by_rt.keys())]
    ax.bar(labels, vals, color="#70AD47")
    ax.set_xlabel("ranks × threads-per-rank")
    ax.set_ylabel("time per epoch (s)")
    ax.set_title("Hybrid MPI+OpenMP — time per epoch")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/hybrid_heatmap.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_cuda(runs):
    if not runs:
        print("  [skip] CUDA — no data")
        return
    by_b = {}
    for r in runs:
        by_b[r["batch"]] = r["avg_ep_s"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = sorted(by_b.keys())
    ax.plot(ks, [by_b[k] for k in ks], "d-", color="#C00000")
    ax.set_xlabel("batch size")
    ax.set_ylabel("time per epoch (s)")
    ax.set_xscale("log", base=2)
    ax.set_title("CUDA training time per epoch vs batch size")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = f"{OUT}/cuda_time_vs_batch.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out}")


def accuracy_table(all_runs):
    rows = []
    for variant, runs in all_runs.items():
        if not runs:
            continue
        # Prefer 80-epoch run; fall back to longest run we have
        best = max(runs, key=lambda r: r["epochs"] or 0)
        rows.append({
            "variant": variant,
            "epochs": best["epochs"],
            "threads": best["threads"],
            "ranks": best["ranks"],
            "batch": best["batch"],
            "test_q3": best["test_q3"],
            "train_q3": best["train_q3"],
            "val_q3_last": best["val_q3_last"],
            "avg_ep_s": round(best["avg_ep_s"], 3) if best["avg_ep_s"] else None,
            "total_s": round(best["total_s"], 1) if best["total_s"] else None,
        })
    out = f"{OUT}/accuracy_table.csv"
    with open(out, "w") as f:
        if rows:
            keys = rows[0].keys()
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) if r[k] is not None else "" for k in keys) + "\n")
    print(f"  wrote {out}")
    return rows


def main():
    runs = {
        v: load_runs(v)
        for v in ("serial", "openmp", "pthreads", "mpi", "hybrid", "cuda")
    }
    for v, lst in runs.items():
        print(f"{v:10s}: {len(lst)} run(s)")

    omp = best_per_threads(runs["openmp"], "threads")
    pth = best_per_threads(runs["pthreads"], "threads")
    mpi = best_per_threads(runs["mpi"], "ranks")

    print("\n--- plots ---")
    plot_time_vs_threads(omp, pth)
    plot_speedup(omp, pth)
    plot_time_vs_ranks(mpi)
    plot_hybrid(runs["hybrid"])
    plot_cuda(runs["cuda"])

    print("\n--- accuracy table ---")
    table = accuracy_table(runs)
    for r in table:
        print(f"  {r['variant']:10s} ep={r['epochs']:>3} t={r['threads']} r={r['ranks']} "
              f"test_q3={r['test_q3']} avg_ep={r['avg_ep_s']}s")


if __name__ == "__main__":
    main()
