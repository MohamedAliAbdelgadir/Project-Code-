"""
Selection of the warm-up period, run length and number of replications 
using Welch transient analysis, run-length stability, and replication precision.
"""

import heapq

import numpy as np
import pandas as pd

from simulation import Simulation
from experiments import run_replications
from analysis import summarise_replications
from parameters import BASELINE_MODEL, UNIFORM, KEY_METRICS


def _run_with_time_series(model, policy_A, seed, total_time, sample_interval):
    """One replication, sampling queue lengths on a fixed time grid."""
    sim = Simulation(model, policy_A, warm_up=0.0, run_length=total_time, seed=seed)
    sample_times = np.arange(0.0, total_time + sample_interval / 2.0, sample_interval)
    index = 0
    rows = []

    def record_until(limit):
        nonlocal index
        while index < len(sample_times) and sample_times[index] <= limit:
            rows.append({
                "Time": float(sample_times[index]),
                "Q1": float(len(sim.queue_A1)),
                "Q2": float(len(sim.queue_A2)),
                "Qe": float(len(sim.queue_E)),
                "System Size": float(len(sim.queue_A1) + len(sim.queue_A2)
                                     + len(sim.queue_E)),
            })
            index += 1

    record_until(0.0)
    while sim.event_calendar:
        event_time, _, event_type, participant_id = heapq.heappop(sim.event_calendar)
        record_until(min(event_time, sim.termination_time))
        sim._cross_warm_up(sim.clock, event_time)
        sim._accumulate_queue_area(sim.clock, event_time)
        sim.clock = event_time

        if event_type == "termination":
            break
        elif event_type == "arrival_A1":
            sim.arrival_applicant("A1")
        elif event_type == "arrival_A2":
            sim.arrival_applicant("A2")
        elif event_type == "arrival_E":
            sim.arrival_employer()
        elif event_type == "abandonment":
            sim.abandon(participant_id)
        sim._update_maximum_queue_lengths()

    record_until(total_time)
    return pd.DataFrame(rows)


def run_welch_analysis(model=BASELINE_MODEL, policy_A=UNIFORM, replications=50,
                       total_time=3_000.0, sample_interval=5.0,
                       moving_average_window=21, seed=30_000):
    
    series = []
    for replication in range(replications):
        one = _run_with_time_series(model, policy_A, seed + replication,
                                    total_time, sample_interval)
        one["Replication"] = replication + 1
        series.append(one)

    raw = pd.concat(series, ignore_index=True)
    means = (raw.groupby("Time", as_index=False)[["Q1", "Q2", "Qe", "System Size"]]
             .mean().sort_values("Time"))
    for metric in ("Q1", "Q2", "Qe", "System Size"):
        means[f"{metric} Welch MA"] = (means[metric]
                                       .rolling(moving_average_window, center=True,
                                                min_periods=1).mean())
    return raw, means


def suggest_warm_up(welch_means, metric="System Size Welch MA", tail_fraction=0.25,
                    tolerance_fraction=0.02, consecutive_points=20):
    
    values = welch_means[metric].to_numpy(dtype=float)
    times = welch_means["Time"].to_numpy(dtype=float)
    tail_start = max(0, int(np.floor(len(values) * (1.0 - tail_fraction))))
    tail_mean = float(np.mean(values[tail_start:]))
    tolerance = tolerance_fraction * max(abs(tail_mean), 1e-12)
    inside = np.abs(values - tail_mean) <= tolerance

    candidate = None
    for start in range(len(values) - consecutive_points + 1):
        if inside[start:start + consecutive_points].all():
            candidate = start
            break

    return {
        "Suggested Warm-up": float(times[candidate]) if candidate is not None else float("nan"),
        "Tail Mean": tail_mean,
        "Tolerance": tolerance,
        "Consecutive Points": consecutive_points,
        "Found": candidate is not None,
    }



COUNT_METRICS = ("Matches", "A1 Matches", "A2 Matches", "Scans", "Total Arrivals",
                 "A1 Arrivals", "A2 Arrivals", "Employer Arrivals", "Abandonments",
                 "A1 Abandonments", "A2 Abandonments", "Employer Abandonments",
                 "Rejections")


def analyse_run_lengths(model=BASELINE_MODEL, policy_A=UNIFORM,
                        horizons=(5_000, 10_000, 20_000, 40_000, 80_000),
                        replications=20, warm_up=550.0, seed=40_000,
                        key_metrics=KEY_METRICS):
    summaries = []
    for horizon in horizons:
        raw = run_replications(model, policy_A, warm_up, float(horizon), replications, seed)
        summary = summarise_replications(raw)
        summary["Observation Horizon"] = horizon
        summaries.append(summary)

    full = pd.concat(summaries, ignore_index=True)
    selected = full[full["Metric"].isin(key_metrics)].copy()
    selected.sort_values(["Metric", "Observation Horizon"], inplace=True)
    selected["Previous Mean"] = selected.groupby("Metric")["Mean"].shift(1)
    selected["Relative Mean Change (%)"] = np.where(
        selected["Previous Mean"].notna() & (selected["Previous Mean"] != 0),
        100.0 * np.abs(selected["Mean"] - selected["Previous Mean"])
        / np.abs(selected["Previous Mean"]),
        np.nan,
    )
    return full, selected


def analyse_replication_precision(model=BASELINE_MODEL, policy_A=UNIFORM,
                                  candidates=(5, 10, 20, 30, 40, 50, 60),
                                  warm_up=550.0, observation_horizon=80_000.0,
                                  seed=50_000, target=1.0, key_metrics=KEY_METRICS):
    everything = run_replications(model, policy_A, warm_up, observation_horizon,
                                  max(candidates), seed)
    summaries, decisions = [], []
    for n in candidates:
        summary = summarise_replications(everything.iloc[:n].copy())
        summary["Replications"] = n
        summaries.append(summary)

        key = summary[summary["Metric"].isin(key_metrics)]
        finite = key["Relative Half Width (%)"].replace([np.inf, -np.inf], np.nan).dropna()
        worst = float(finite.max()) if not finite.empty else float("nan")
        decisions.append({
            "Replications": n,
            "Worst Relative Half Width (%)": worst,
            "Meets Target": bool(np.isfinite(worst) and worst <= target),
        })

    return everything, pd.concat(summaries, ignore_index=True), pd.DataFrame(decisions)


def write_calibration_report(output_path, warm_up_result, stability, decisions):
    from pathlib import Path
    output_path = Path(output_path)

    last = float(stability["Observation Horizon"].max())
    latest = stability[(stability["Observation Horizon"] == last)
                       & (~stability["Metric"].isin(COUNT_METRICS))]
    worst_change = float(latest["Relative Mean Change (%)"].dropna().max())

    passing = decisions[decisions["Meets Target"]]
    selected = int(passing.iloc[0]["Replications"]) if not passing.empty else None

    lines = [
        "# Calibration Report", "",
        "## Welch warm-up analysis",
        f"- Suggested warm-up: {warm_up_result['Suggested Warm-up']}",
        f"- Tail mean: {warm_up_result['Tail Mean']:.6f}",
        f"- Stable window found: {warm_up_result['Found']}", "",
        "## Run-length analysis",
        f"- Largest horizon tested: {last:g}",
        f"- Worst successive relative change at that horizon (rate and average "
        f"metrics only): {worst_change:.4f}%", "",
        "## Replication analysis",
        f"- First replication count meeting the target: {selected}",
        f"- Smallest worst relative half-width tested: "
        f"{decisions['Worst Relative Half Width (%)'].min():.4f}%", "",
        "These diagnostics support, but do not replace, inspection of the plots.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
