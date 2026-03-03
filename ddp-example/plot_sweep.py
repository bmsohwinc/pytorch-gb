#!/usr/bin/env python3
"""
Plot results from a gradient_as_bucket_view parameter sweep.

Usage:
    python3 plot_sweep.py <path_to_sweep_results.csv>

Produces 4 subplots:
  1. Backward pass time: gb=0 vs gb=1 across parameter sizes
  2. Backward improvement % vs parameter size
  3. C++ copy time: gb=0 vs gb=1 across parameter sizes
  4. C++ copy improvement % vs parameter size
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def format_params(val):
    """Format parameter count for tick labels: 262K, 1M, 16M, etc."""
    if val >= 1e6:
        return f"{val/1e6:.0f}M"
    elif val >= 1e3:
        return f"{val/1e3:.0f}K"
    return str(int(val))


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_sweep.py <sweep_results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    # Output dir = same directory as the CSV
    out_dir = os.path.dirname(os.path.abspath(csv_path))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Gradient-as-Bucket-View Performance vs Parameter Size",
                 fontsize=16, fontweight="bold", y=0.98)

    x = df["num_params"]
    x_labels = [format_params(v) for v in x]

    # ── Plot 1: Backward pass time comparison ────────────────────────
    ax = axes[0, 0]
    ax.errorbar(range(len(x)), df["bwd_gb0_mean"] * 1000, yerr=df["bwd_gb0_std"] * 1000,
                fmt="o-", label="Standard DDP (gb=0)", capsize=4, color="#e74c3c")
    ax.errorbar(range(len(x)), df["bwd_gb1_mean"] * 1000, yerr=df["bwd_gb1_std"] * 1000,
                fmt="s--", label="Bucket View (gb=1)", capsize=4, color="#2ecc71")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Backward Pass Time (ms)")
    ax.set_title("Backward Pass Time: gb=0 vs gb=1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Backward improvement % ───────────────────────────────
    ax = axes[0, 1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["bwd_improve_pct"]]
    bars = ax.bar(range(len(x)), df["bwd_improve_pct"], color=colors,
                  edgecolor="black", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Backward Pass Improvement (gb=1 vs gb=0)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, df["bwd_improve_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=9)

    # ── Plot 3: C++ copy time comparison ─────────────────────────────
    ax = axes[1, 0]
    ax.plot(range(len(x)), df["copy_gb0_mean"], "o-",
            label="Standard DDP (gb=0)", color="#e74c3c", markersize=8)
    ax.plot(range(len(x)), df["copy_gb1_mean"], "s--",
            label="Bucket View (gb=1)", color="#2ecc71", markersize=8)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Copy Time (μs)")
    ax.set_title("C++ Reducer Copy Time: gb=0 vs gb=1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Copy improvement % ───────────────────────────────────
    ax = axes[1, 1]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in df["copy_improve_pct"]]
    bars = ax.bar(range(len(x)), df["copy_improve_pct"], color=colors,
                  edgecolor="black", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("C++ Copy Time Improvement (gb=1 vs gb=0)")
    ax.grid(axis="y", alpha=0.3)
    # Annotate bars
    for bar, val in zip(bars, df["copy_improve_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(out_dir, "sweep_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {out_path}")

    # Also save a text summary table
    summary_path = os.path.join(out_dir, "sweep_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Gradient-as-Bucket-View Parameter Sweep Results\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Params':>12} {'Bwd gb=0 (ms)':>15} {'Bwd gb=1 (ms)':>15} {'Bwd Δ%':>10} "
                f"{'Copy gb=0 (μs)':>16} {'Copy gb=1 (μs)':>16} {'Copy Δ%':>10}\n")
        f.write("-" * 90 + "\n")
        for _, row in df.iterrows():
            f.write(f"{format_params(row['num_params']):>12} "
                    f"{row['bwd_gb0_mean']*1000:>15.3f} "
                    f"{row['bwd_gb1_mean']*1000:>15.3f} "
                    f"{row['bwd_improve_pct']:>+10.1f} "
                    f"{row['copy_gb0_mean']:>16.0f} "
                    f"{row['copy_gb1_mean']:>16.0f} "
                    f"{row['copy_improve_pct']:>+10.1f}\n")
        f.write("-" * 90 + "\n")
    print(f"Saved summary to: {summary_path}")

    plt.show()


if __name__ == "__main__":
    main()
