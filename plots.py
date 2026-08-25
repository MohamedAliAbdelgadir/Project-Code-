"""Plotting functions for factor sweeps and simulation checks."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import policy_label

FIGURE_SIZE = (8.5, 5.2)
DPI = 250


def _safe_name(text):
    return (str(text).lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("%", "pct").replace("/", "_").replace(".", "p"))


def sweep_plot(summary, metric, output_path, xlabel=None, logx=False,
               baseline_level=None, title=None):
    
    data = summary[summary["Metric"] == metric]
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for policy in sorted(data["Policy"].unique()):
        series = data[data["Policy"] == policy].sort_values("Level")
        levels = series["Level"].to_numpy(dtype=float)
        if logx:
            
            finite = levels[(levels > 0) & (levels < float("inf"))]
            low = finite.min() / 4 if len(finite) else 0.01
            high = finite.max() * 4 if len(finite) else 100.0
            levels = [low if x == 0 else (high if x == float("inf") else x) for x in levels]
        ax.errorbar(levels, series["Mean"], yerr=series["CI Half Width"],
                    marker="o", capsize=3, label=policy_label(policy))

    if baseline_level is not None:
        ax.axvline(baseline_level, linestyle="--", color="grey", linewidth=1,
                   label="Baseline")
    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel or data["Factor"].iloc[0])
    ax.set_ylabel(metric)
    ax.set_title(title or f"{data['Factor'].iloc[0]}: {metric}")
    ax.grid(alpha=0.25)
    if data["Policy"].nunique() > 1 or baseline_level is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def sweep_plot_set(summary, metrics, output_dir, prefix, xlabel=None, logx=False,
                   baseline_level=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in metrics:
        path = sweep_plot(summary, metric,
                          output_dir / f"{prefix}_{_safe_name(metric)}.png",
                          xlabel=xlabel, logx=logx, baseline_level=baseline_level)
        if path:
            paths.append(path)
    return paths


def plot_welch(welch_means, output_path, metric="System Size"):
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(welch_means["Time"], welch_means[metric], linewidth=0.8,
            label="Across-replication mean")
    ax.plot(welch_means["Time"], welch_means[f"{metric} Welch MA"], linewidth=1.8,
            label="Centred moving average")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel(metric)
    ax.set_title(f"Welch transient analysis: {metric}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_run_length(stability, output_dir, metrics):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in metrics:
        subset = stability[stability["Metric"] == metric].sort_values("Observation Horizon")
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        ax.errorbar(subset["Observation Horizon"], subset["Mean"],
                    yerr=subset["CI Half Width"], marker="o", capsize=4)
        ax.set_xlabel("Observation horizon")
        ax.set_ylabel(metric)
        ax.set_title(f"Run-length stability: {metric}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"run_length_{_safe_name(metric)}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_replication_precision(decisions, output_path, target):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.plot(decisions["Replications"], decisions["Worst Relative Half Width (%)"],
            marker="o")
    ax.axhline(target, linestyle="--", color="grey", label=f"Target {target:g}%")
    ax.set_xlabel("Number of replications")
    ax.set_ylabel("Worst relative half-width (%)")
    ax.set_title("Replication precision")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path
