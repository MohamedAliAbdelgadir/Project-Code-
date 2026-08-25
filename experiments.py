"""
Runs experiments by varying one parameter or policy weight across levels, 
policies and replications, then summarises the results with means and confidence intervals.
"""

from pathlib import Path
from dataclasses import asdict
import json

import pandas as pd

from simulation import Simulation
from analysis import summarise_replications
from parameters import (
    BASELINE_MODEL, FACTORS, A_SWEEP, MAIN_POLICIES,
    apply_changes, policy_label,
)


def run_replications(model, policy_A, warm_up, run_length, replications, base_seed):

    rows = []
    for replication in range(replications):
        result = Simulation(model, policy_A, warm_up, run_length,
                            seed=base_seed + replication).run()
        result["Replication"] = replication + 1
        rows.append(result)
    return pd.DataFrame(rows)


def _summarise(frame):

    outcomes = frame.drop(columns=["Policy A", "Replication"], errors="ignore")
    return summarise_replications(outcomes).reset_index(drop=True)


def _label(frame, **columns):
    for name, value in columns.items():
        frame[name] = value
    return frame


def run_factor_sweep(factor_id, warm_up, run_length, replications, base_seed,
                     model=BASELINE_MODEL, policies=MAIN_POLICIES):
    """Vary one model parameter across its levels, under each policy."""
    factor = FACTORS[factor_id]

    varied_key = factor.get("varies", factor_id)
    fixed = factor.get("fixed", {})
    replication_frames, summary_frames = [], []

    for level_index, level in enumerate(factor["levels"]):
        varied_model = apply_changes(model, {**fixed, varied_key: level})

        seed = base_seed + level_index * 1_000
        for policy_A in policies:
            raw = run_replications(varied_model, policy_A, warm_up, run_length,
                                   replications, seed)
            summary = _summarise(raw)
            columns = dict(Factor=factor["name"], FactorId=factor_id,
                           Level=level, Policy=policy_A,
                           PolicyLabel=policy_label(policy_A))
            replication_frames.append(_label(raw, **columns))
            summary_frames.append(_label(summary, **columns))

    return (pd.concat(replication_frames, ignore_index=True),
            pd.concat(summary_frames, ignore_index=True))


def run_policy_sweep(warm_up, run_length, replications, base_seed,
                     model=BASELINE_MODEL, policy_values=A_SWEEP):
    
    replication_frames, summary_frames = [], []

    
    for policy_A in policy_values:
        raw = run_replications(model, policy_A, warm_up, run_length,
                               replications, base_seed)
        summary = _summarise(raw)
        columns = dict(Factor="Policy weight", FactorId="policy_A",
                       Level=policy_A, Policy=policy_A,
                       PolicyLabel=policy_label(policy_A))
        replication_frames.append(_label(raw, **columns))
        summary_frames.append(_label(summary, **columns))

    return (pd.concat(replication_frames, ignore_index=True),
            pd.concat(summary_frames, ignore_index=True))


def run_baseline(warm_up, run_length, replications, base_seed,
                 model=BASELINE_MODEL, policies=MAIN_POLICIES):
    
    replication_frames, summary_frames = [], []
    for policy_A in policies:
        raw = run_replications(model, policy_A, warm_up, run_length,
                               replications, base_seed)
        summary = _summarise(raw)
        columns = dict(Policy=policy_A, PolicyLabel=policy_label(policy_A))
        replication_frames.append(_label(raw, **columns))
        summary_frames.append(_label(summary, **columns))
    return (pd.concat(replication_frames, ignore_index=True),
            pd.concat(summary_frames, ignore_index=True))


def pivot_metric(summary, metric, value="Mean"):

    subset = summary[summary["Metric"] == metric]
    if subset.empty:
        return pd.DataFrame()
    if (subset["FactorId"] == "policy_A").all():
        return (subset.set_index("Level")[["Mean", "CI Half Width"]]
                .sort_index().rename(columns={"Mean": metric}))
    return subset.pivot_table(index="Level", columns="PolicyLabel", values=value)


def export(frames, output_dir, model=BASELINE_MODEL, settings=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    record = {"baseline_model": asdict(model)}
    if settings:
        record.update(settings)
    (output_dir / "settings.json").write_text(json.dumps(record, indent=2, default=str),
                                              encoding="utf-8")
    return output_dir
