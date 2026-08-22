#!/usr/bin/env python3
"""Create benchmark charts from benchmark_results.csv.

Examples:
    python plot_benchmarks.py benchmark_results.csv
    python plot_benchmarks.py results.csv --output-dir benchmark_charts
    python plot_benchmarks.py results.csv --metric median_ms --dpi 200

Produces PNG images for:
  - latency by sequence length, split by batch and hidden dimension
  - speedup heatmaps for CUDA Graph replay versus torch.compile
  - throughput by sequence length
  - latency distributions by implementation
  - numerical accuracy by implementation
  - a summary dashboard

Median latency is the default because it is less sensitive to the timing
outliers visible in short GPU workloads. Use --metric mean_ms if desired.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMPLEMENTATION_LABELS = {
    "cuda_graph": "CUDA Graph",

    "torch_compile": "torch.compile",
}
COLORS = {
    "cuda_graph": "#ED7D31",

    "torch_compile": "#70AD47",
}
MARKERS = {
    "cuda_graph": "s",

    "torch_compile": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot convolution-gate benchmark CSV")
    parser.add_argument("csv", type=Path, help="Input benchmark CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_charts"),
        help="Directory for generated PNG files",
    )
    parser.add_argument(
        "--metric",
        choices=("median_ms", "mean_ms", "p95_ms"),
        default="median_ms",
        help="Latency statistic used for comparisons and speedups",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help="Plot only this convolution kernel size; default plots every size",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "batch", "seq", "dim", "kernel", "implementation",
        "mean_ms", "median_ms", "p95_ms", "tokens_per_second",
        "max_abs_error", "mean_abs_error", "passed", "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    numeric = [
        "batch", "seq", "dim", "kernel", "mean_ms", "median_ms",
        "p05_ms", "p95_ms", "std_ms", "iterations_per_second",
        "tokens_per_second", "speedup_vs_compile", "max_abs_error",
        "mean_abs_error", "max_rel_error",
    ]
    for column in numeric:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    valid = df["status"].fillna("").eq("ok")
    passed = df["passed"].astype(str).str.lower().eq("true")
    df = df[valid & passed].copy()
    if df.empty:
        raise ValueError("CSV contains no successful benchmark rows")
    return df


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values) & (values > 0))


def add_speedup(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute torch.compile latency divided by each implementation latency."""
    keys = ["batch", "seq", "dim", "kernel", "dtype"]
    compiled = (
        df[df["implementation"] == "torch_compile"]
        .drop_duplicates(keys)
        .set_index(keys)[metric]
        .rename("compile_latency")
    )
    if compiled.empty:
        raise ValueError(
            "CSV has no torch_compile rows; rerun rangebench without --skip-compile"
        )
    result = df.join(compiled, on=keys)
    result["speedup_selected"] = result["compile_latency"] / result[metric]
    return result


def plot_latency_facets(df: pd.DataFrame, metric: str, output: Path, dpi: int) -> None:
    for batch in sorted(df["batch"].dropna().unique()):
        subset = df[df["batch"] == batch]
        dims = sorted(subset["dim"].dropna().unique())
        cols = min(2, len(dims))
        rows = math.ceil(len(dims) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
        for ax, dim in zip(axes.flat, dims):
            panel = subset[subset["dim"] == dim]
            for impl in IMPLEMENTATION_LABELS:
                values = panel[panel["implementation"] == impl].sort_values("seq")
                if values.empty:
                    continue
                ax.plot(
                    values["seq"], finite_positive(values[metric]),
                    label=IMPLEMENTATION_LABELS[impl], color=COLORS[impl],
                    marker=MARKERS[impl], linewidth=2, markersize=5,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"Batch {int(batch)}, hidden dimension {int(dim)}")
            ax.set_xlabel("Sequence length")
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
        for ax in axes.flat[len(dims):]:
            ax.remove()
        fig.suptitle(f"Convolution-gate latency ({metric})", fontsize=15)
        save_figure(fig, output / f"latency_batch_{int(batch)}.png", dpi)


def plot_speedup_heatmaps(df: pd.DataFrame, metric: str, output: Path, dpi: int) -> None:
    graph = df[df["implementation"] == "cuda_graph"].copy()
    for batch in sorted(graph["batch"].dropna().unique()):
        panel = graph[graph["batch"] == batch]
        pivot = panel.pivot_table(
            index="dim", columns="seq", values="speedup_selected", aggfunc="median"
        ).sort_index().sort_index(axis=1)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 1.25 * len(pivot.columns)),
                                        max(4.5, 0.75 * len(pivot.index))))
        data = pivot.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        bound = max(1.0, np.nanmax(np.abs(finite - 1.0)) + 1.0) if finite.size else 2.0
        low, high = max(0.0, 2.0 - bound), bound
        image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=low, vmax=high)
        ax.set_xticks(np.arange(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)), [str(int(x)) for x in pivot.index])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Hidden dimension")
        ax.set_title(f"CUDA Graph speedup vs torch.compile, batch {int(batch)} ({metric})")
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if np.isfinite(data[row, col]):
                    ax.text(col, row, f"{data[row, col]:.2f}x", ha="center", va="center",
                            color="black", fontsize=9)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Speedup; greater than 1 is faster")
        save_figure(fig, output / f"speedup_heatmap_batch_{int(batch)}.png", dpi)


def plot_throughput(df: pd.DataFrame, output: Path, dpi: int) -> None:
    for dim in sorted(df["dim"].dropna().unique()):
        subset = df[df["dim"] == dim]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for (batch, impl), values in subset.groupby(["batch", "implementation"]):
            if impl not in IMPLEMENTATION_LABELS:
                continue
            values = values.sort_values("seq")
            ax.plot(
                values["seq"], finite_positive(values["tokens_per_second"]),
                color=COLORS[impl], marker=MARKERS[impl],
                linestyle="-" if int(batch) == 1 else "--",
                linewidth=1.7, label=f"{IMPLEMENTATION_LABELS[impl]}, B={int(batch)}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Tokens per second")
        ax.set_title(f"Throughput, hidden dimension {int(dim)}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        save_figure(fig, output / f"throughput_dim_{int(dim)}.png", dpi)


def plot_latency_distribution(df: pd.DataFrame, metric: str, output: Path, dpi: int) -> None:
    implementations = [x for x in IMPLEMENTATION_LABELS if x in set(df["implementation"])]
    values = [finite_positive(df.loc[df["implementation"] == impl, metric]).dropna() for impl in implementations]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    box = ax.boxplot(values, label=[IMPLEMENTATION_LABELS[x] for x in implementations],
                     patch_artist=True, showfliers=True)
    for patch, impl in zip(box["boxes"], implementations):
        patch.set_facecolor(COLORS[impl])
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Latency distribution across benchmark shapes ({metric})")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    save_figure(fig, output / "latency_distribution.png", dpi)


def plot_accuracy(df: pd.DataFrame, output: Path, dpi: int) -> None:
    implementations = [x for x in IMPLEMENTATION_LABELS if x in set(df["implementation"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in (
        (axes[0], "max_abs_error", "Maximum absolute error"),
        (axes[1], "mean_abs_error", "Mean absolute error"),
    ):
        values = [finite_positive(df.loc[df["implementation"] == impl, metric]).dropna()
                  for impl in implementations]
        box = ax.boxplot(values, label=[IMPLEMENTATION_LABELS[x] for x in implementations],
                         patch_artist=True, showfliers=True)
        for patch, impl in zip(box["boxes"], implementations):
            patch.set_facecolor(COLORS[impl])
            patch.set_alpha(0.75)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("Absolute error vs high-precision reference")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.suptitle("Numerical accuracy across benchmark shapes", fontsize=15)
    save_figure(fig, output / "numerical_accuracy.png", dpi)


def plot_dashboard(df: pd.DataFrame, metric: str, output: Path, dpi: int) -> None:
    graph = df[df["implementation"] == "cuda_graph"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Speedup against flattened token count.
    ax = axes[0, 0]
    for dim, values in graph.groupby("dim"):
        ax.scatter(values["batch"] * values["seq"], values["speedup_selected"],
                   label=f"D={int(dim)}", s=42, alpha=0.8)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Flattened tokens (batch x sequence)")
    ax.set_ylabel("CUDA Graph speedup vs torch.compile")
    ax.set_title("Speedup by workload size")
    ax.grid(True, alpha=0.25)
    ax.legend()

    # CUDA Graph latency scaling.
    ax = axes[0, 1]
    for dim, values in graph.groupby("dim"):
        values = values.assign(tokens=values["batch"] * values["seq"]).sort_values("tokens")
        grouped = values.groupby("tokens", as_index=False)[metric].median()
        ax.plot(grouped["tokens"], grouped[metric], marker="o", label=f"D={int(dim)}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Flattened tokens")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("CUDA Graph latency scaling")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    # Speedup distribution.
    ax = axes[1, 0]
    speedups = graph["speedup_selected"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(speedups, bins=min(15, max(5, len(speedups) // 3)), color=COLORS["cuda_graph"],
            edgecolor="black", alpha=0.8)
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
    if not speedups.empty:
        ax.axvline(speedups.median(), color="#7030A0", linewidth=2,
                   label=f"Median {speedups.median():.2f}x")
        ax.legend()
    ax.set_xlabel("CUDA Graph speedup vs torch.compile")
    ax.set_ylabel("Number of shapes")
    ax.set_title("Speedup distribution")
    ax.grid(True, axis="y", alpha=0.25)

    # Error vs latency.
    ax = axes[1, 1]
    for impl, values in df.groupby("implementation"):
        if impl not in IMPLEMENTATION_LABELS:
            continue
        ax.scatter(finite_positive(values[metric]), finite_positive(values["max_abs_error"]),
                   label=IMPLEMENTATION_LABELS[impl], color=COLORS[impl],
                   marker=MARKERS[impl], alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("Maximum absolute error")
    ax.set_title("Accuracy versus latency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    fig.suptitle("CUDA Graph vs torch.compile benchmark summary", fontsize=17)
    save_figure(fig, output / "benchmark_dashboard.png", dpi)


def main() -> None:
    args = parse_args()
    df = load_results(args.csv)
    if args.kernel_size is not None:
        df = df[df["kernel"] == args.kernel_size].copy()
        if df.empty:
            raise ValueError(f"No successful rows found for kernel size {args.kernel_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = add_speedup(df, args.metric)

    for kernel, kernel_df in df.groupby("kernel"):
        output = args.output_dir / f"kernel_{int(kernel)}"
        output.mkdir(parents=True, exist_ok=True)
        plot_latency_facets(kernel_df, args.metric, output, args.dpi)
        plot_speedup_heatmaps(kernel_df, args.metric, output, args.dpi)
        plot_throughput(kernel_df, output, args.dpi)
        plot_latency_distribution(kernel_df, args.metric, output, args.dpi)
        plot_accuracy(kernel_df, output, args.dpi)
        plot_dashboard(kernel_df, args.metric, output, args.dpi)

    print(f"\nAll charts written under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
