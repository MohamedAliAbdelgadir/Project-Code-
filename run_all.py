"""
Runs selected stages of the analysis, with an optional --smoke mode for a quick check with shorter runs.

    python run_all.py --stage verification
    python run_all.py --stage welch
    python run_all.py --stage run_length
    python run_all.py --stage replication_precision
    python run_all.py --stage calibration          # all three of the above
    python run_all.py --stage baseline
    python run_all.py --stage policy_sweep
    python run_all.py --stage factor --factor lambda_1

"""

import argparse
from pathlib import Path

import pandas as pd

from parameters import (
    BASELINE_MODEL, WARM_UP, OBSERVATION_HORIZON, REPLICATIONS, BASE_SEED,
    FACTORS, KEY_METRICS, A_SWEEP, MAIN_POLICIES,
)
from validation import assert_verification_suite
from experiments import run_factor_sweep, run_policy_sweep, run_baseline, export
from calibration import (
    run_welch_analysis, suggest_warm_up, analyse_run_lengths,
    analyse_replication_precision, write_calibration_report,
)
from plots import (
    sweep_plot_set, plot_welch, plot_run_length, plot_replication_precision,
)


def settings_for(smoke):
    if smoke:
        return 25.0, 400.0, 3
    return WARM_UP, OBSERVATION_HORIZON, REPLICATIONS


def stage_verification(output, smoke):
    print("Stage: verification")
    results = assert_verification_suite()
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "verification_results.csv", index=False)
    print(results.to_string(index=False))
    print(f"\nAll {len(results)} verification checks passed.\n")


def stage_welch(output, smoke):
    print("Stage: Welch warm-up analysis")
    directory = output / "calibration"
    directory.mkdir(parents=True, exist_ok=True)

    replications = 5 if smoke else 50
    total_time = 300.0 if smoke else 3_000.0
    raw, means = run_welch_analysis(replications=replications, total_time=total_time)
    suggestion = suggest_warm_up(means)

    raw.to_csv(directory / "welch_raw.csv", index=False)
    means.to_csv(directory / "welch_means.csv", index=False)
    pd.DataFrame([suggestion]).to_csv(directory / "welch_suggestion.csv", index=False)
    plot_welch(means, directory / "welch_system_size.png")

    print(f"Suggested warm-up (compare against the plot): {suggestion['Suggested Warm-up']}")
    print(f"Saved to {directory.resolve()}\n")
    return suggestion


def stage_run_length(output, smoke):
    print("Stage: run-length stability")
    directory = output / "calibration"
    directory.mkdir(parents=True, exist_ok=True)

    horizons = (400.0, 800.0) if smoke else (5_000, 10_000, 20_000, 40_000, 80_000)
    replications = 3 if smoke else 20
    full, stability = analyse_run_lengths(horizons=horizons, replications=replications,
                                          warm_up=WARM_UP)
    full.to_csv(directory / "run_length_summary.csv", index=False)
    stability.to_csv(directory / "run_length_stability.csv", index=False)
    plot_run_length(stability, directory,
                    ("Throughput", "Matching Split p1", "Abandonment Rate"))
    print(f"Saved to {directory.resolve()}\n")
    return stability


def stage_replication_precision(output, smoke):
    print("Stage: replication precision")
    directory = output / "calibration"
    directory.mkdir(parents=True, exist_ok=True)

    candidates = (3, 5) if smoke else (5, 10, 20, 30, 40, 50, 60)
    horizon = 400.0 if smoke else OBSERVATION_HORIZON
    everything, summary, decisions = analyse_replication_precision(
        candidates=candidates, warm_up=WARM_UP, observation_horizon=horizon)
    everything.to_csv(directory / "precision_replications.csv", index=False)
    summary.to_csv(directory / "precision_summary.csv", index=False)
    decisions.to_csv(directory / "precision_decisions.csv", index=False)
    plot_replication_precision(decisions, directory / "replication_precision.png", target=1.0)
    print(decisions.to_string(index=False))
    print(f"Saved to {directory.resolve()}\n")
    return decisions


def stage_calibration(output, smoke):
    suggestion = stage_welch(output, smoke)
    stability = stage_run_length(output, smoke)
    decisions = stage_replication_precision(output, smoke)
    report = write_calibration_report(output / "calibration" / "calibration_report.md",
                                      suggestion, stability, decisions)
    print(f"Calibration report: {report.resolve()}\n")


def stage_baseline(output, smoke):
    warm_up, horizon, replications = settings_for(smoke)
    print("Stage: baseline")
    raw, summary = run_baseline(warm_up, horizon, replications, BASE_SEED)
    directory = export({"baseline_replications": raw, "baseline_summary": summary},
                       output / "baseline", settings={"warm_up": warm_up,
                                                      "horizon": horizon,
                                                      "replications": replications})
    print(summary[summary["Metric"].isin(KEY_METRICS)]
          .pivot_table(index="Metric", columns="PolicyLabel", values="Mean")
          .to_string())
    print(f"\nSaved to {directory.resolve()}\n")


def stage_policy_sweep(output, smoke):
    warm_up, horizon, replications = settings_for(smoke)
    print(f"Stage: policy sweep over A ({len(A_SWEEP)} values)")
    raw, summary = run_policy_sweep(warm_up, horizon, replications, BASE_SEED)
    directory = export({"policy_sweep_replications": raw, "policy_sweep_summary": summary},
                       output / "policy_sweep", settings={"A_values": list(A_SWEEP)})
    sweep_plot_set(summary, KEY_METRICS, directory / "figures", prefix="policy_sweep",
                   xlabel="Policy weight A", logx=True)
    print(f"Saved to {directory.resolve()}\n")


def stage_factor(output, smoke, factor_id):
    warm_up, horizon, replications = settings_for(smoke)
    factor_ids = [factor_id] if factor_id else list(FACTORS)
    for identifier in factor_ids:
        factor = FACTORS[identifier]
        print(f"Stage: factor sweep, {factor['name']} "
              f"({len(factor['levels'])} levels x {len(MAIN_POLICIES)} policies)")
        raw, summary = run_factor_sweep(identifier, warm_up, horizon, replications, BASE_SEED)
        directory = export({f"{identifier}_replications": raw,
                            f"{identifier}_summary": summary},
                           output / "factors" / identifier)
        baseline_level = getattr(BASELINE_MODEL, identifier, None)
        if identifier == "theta_applicants":
            baseline_level = BASELINE_MODEL.theta_1
        sweep_plot_set(summary, KEY_METRICS, directory / "figures", prefix=identifier,
                       xlabel=factor["name"], baseline_level=baseline_level)
        print(f"Saved to {directory.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the matching-system study.")
    parser.add_argument("--output", default="results")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage", default="verification", choices=[
        "verification", "welch", "run_length", "replication_precision",
        "calibration", "baseline", "policy_sweep", "factor",
    ])
    parser.add_argument("--factor", default=None, choices=list(FACTORS),
                        help="Used with --stage factor. Omit to run all factors.")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.stage == "verification":
        stage_verification(output, args.smoke)
    elif args.stage == "welch":
        stage_welch(output, args.smoke)
    elif args.stage == "run_length":
        stage_run_length(output, args.smoke)
    elif args.stage == "replication_precision":
        stage_replication_precision(output, args.smoke)
    elif args.stage == "calibration":
        stage_calibration(output, args.smoke)
    elif args.stage == "baseline":
        stage_baseline(output, args.smoke)
    elif args.stage == "policy_sweep":
        stage_policy_sweep(output, args.smoke)
    elif args.stage == "factor":
        stage_factor(output, args.smoke, args.factor)


if __name__ == "__main__":
    main()
