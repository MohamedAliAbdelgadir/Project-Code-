"""
Tests the simulation under conditions with predictable outcomes. 
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from parameters import (
    ModelParameters, BASELINE_MODEL, apply_changes,
    PRIORITY_1, PRIORITY_2, UNIFORM, A_SWEEP,
)
from simulation import Simulation

SHORT_WARM_UP = 50.0
SHORT_HORIZON = 400.0


@dataclass
class VerificationResult:
    test: str
    passed: bool
    evidence: str


def _result(test, condition, evidence):
    return VerificationResult(test=test, passed=bool(condition), evidence=evidence)


def _run(model, policy_A, seed, warm_up=SHORT_WARM_UP, horizon=SHORT_HORIZON):
    return Simulation(model, policy_A, warm_up, horizon, seed=seed).run()


def _run_many(model, policy_A, replications, seed=2026, warm_up=SHORT_WARM_UP,
              horizon=SHORT_HORIZON):
    return pd.DataFrame([_run(model, policy_A, seed + i, warm_up, horizon)
                         for i in range(replications)])


def verify_flow_conservation():
    results = _run_many(BASELINE_MODEL, UNIFORM, 20)
    worst = int(results["Flow Balance Error"].abs().max())
    return _result("Flow conservation", worst == 0,
                   f"Maximum absolute flow-balance error over 20 replications: {worst}.")


def verify_zero_matching_probability():
    model = apply_changes(BASELINE_MODEL, {"q_1": 0.0, "q_2": 0.0})
    results = _run_many(model, UNIFORM, 10)
    worst = int(results["Matches"].max())
    return _result("Zero matching probability", worst == 0,
                   f"Maximum matches with q1 = q2 = 0 over 10 replications: {worst}.")


def verify_certain_matching():
    """When q=1, any scan of a non-empty queue succeeds, 
    preventing applicants and employers from waiting simultaneously."""
    model = apply_changes(BASELINE_MODEL, {"q_1": 1.0, "q_2": 1.0})

    class Instrumented(Simulation):
        
        def __init__(self, *args, **kwargs):
            self.worst_simultaneous = 0
            super().__init__(*args, **kwargs)

        def _update_maximum_queue_lengths(self):
            super()._update_maximum_queue_lengths()
            waiting_applicants = len(self.queue_A1) + len(self.queue_A2)
            simultaneous = min(waiting_applicants, len(self.queue_E))
            if simultaneous > self.worst_simultaneous:
                self.worst_simultaneous = simultaneous

    sim = Instrumented(model, UNIFORM, SHORT_WARM_UP, SHORT_HORIZON, seed=2026)
    result = sim.run()
    condition = result["Successful Scan Rate"] == 1.0 and sim.worst_simultaneous == 0
    return _result("Certain matching (q = 1)", condition,
                   f"Scan success rate {result['Successful Scan Rate']:.3f}; largest "
                   f"simultaneous applicant and employer occupancy {sim.worst_simultaneous}.")


def verify_no_employer_arrivals():
    model = apply_changes(BASELINE_MODEL, {"lambda_e": 0.0})
    results = _run_many(model, UNIFORM, 5)
    condition = (results["Employer Arrivals"] == 0).all() and (results["Matches"] == 0).all()
    return _result("No employer arrivals", condition,
                   "Employer arrivals and matches were zero in every replication.")


def verify_no_applicant_arrivals():
    model = apply_changes(BASELINE_MODEL, {"lambda_1": 0.0, "lambda_2": 0.0})
    results = _run_many(model, UNIFORM, 5)
    condition = ((results["A1 Arrivals"] == 0).all()
                 and (results["A2 Arrivals"] == 0).all()
                 and (results["Matches"] == 0).all())
    return _result("No applicant arrivals", condition,
                   "Applicant arrivals and matches were zero in every replication.")


def verify_single_class_reduction():
    """With no Type 2 arrivals, all matches must be Type 1 regardless of policy."""
    model = apply_changes(BASELINE_MODEL, {"lambda_2": 0.0})
    splits = [_run(model, A, seed=7)["Matching Split p1"]
              for A in (PRIORITY_2, UNIFORM, PRIORITY_1)]
    return _result("Single-class reduction", all(s == 1.0 for s in splits),
                   "With no Type 2 arrivals every match came from Type 1 under all policies.")


def verify_priority_dominance():
    """Priority to Type 1 should increase its share of matches 
    and reduce its abandonment relative to priority to Type 2."""
    p1 = _run_many(BASELINE_MODEL, PRIORITY_1, 10)
    p2 = _run_many(BASELINE_MODEL, PRIORITY_2, 10)
    split_ordered = p1["Matching Split p1"].mean() > p2["Matching Split p1"].mean()
    abandon_ordered = p1["A1 Abandonment Rate"].mean() < p2["A1 Abandonment Rate"].mean()
    return _result("Priority ordering", split_ordered and abandon_ordered,
                   f"Type 1 share {p2['Matching Split p1'].mean():.3f} under priority-2 "
                   f"rises to {p1['Matching Split p1'].mean():.3f} under priority-1; "
                   f"Type 1 abandonment falls from {p2['A1 Abandonment Rate'].mean():.3f} "
                   f"to {p1['A1 Abandonment Rate'].mean():.3f}.")


def verify_symmetric_classes():
    """If both classes have identical parameters, the uniform policy should give similar outcomes for both classes."""
    model = apply_changes(BASELINE_MODEL,
                          {"lambda_1": 0.5, "lambda_2": 0.5, "q_1": 0.2, "q_2": 0.2})
    results = _run_many(model, UNIFORM, 30, horizon=2_000.0)
    split = results["Matching Split p1"].mean()
    gap = abs(results["A1 Abandonment Rate"].mean() - results["A2 Abandonment Rate"].mean())
    return _result("Symmetric classes", abs(split - 0.5) < 0.02 and gap < 0.02,
                   f"Matching split {split:.4f} (expected 0.5); "
                   f"abandonment-rate gap {gap:.4f}.")


def verify_policy_weight_monotonicity():
    """Increasing A should increase the share of matches going to Type 1."""
    splits = []
    for A in (0.0, 0.25, 1.0, 4.0, PRIORITY_1):
        results = _run_many(BASELINE_MODEL, A, 20, horizon=2_000.0)
        splits.append(results["Matching Split p1"].mean())
    increasing = all(b >= a - 1e-9 for a, b in zip(splits, splits[1:]))
    formatted = ", ".join(f"{s:.4f}" for s in splits)
    return _result("Policy weight monotonicity", increasing,
                   f"Type 1 match share across A = 0, 0.25, 1, 4, infinity: {formatted}.")


def verify_capacity_binding():
    model = apply_changes(BASELINE_MODEL, {"k_1": 2, "k_2": 3, "k_e": 1})
    sim = Simulation(model, UNIFORM, SHORT_WARM_UP, SHORT_HORIZON, seed=2026)
    result = sim.run()
    within = (result["Max Q1"] <= 2 and result["Max Q2"] <= 3 and result["Max Qe"] <= 1)
    return _result("Capacity invariants", within and result["Rejections"] > 0,
                   f"Observed maxima Q1={result['Max Q1']}, Q2={result['Max Q2']}, "
                   f"Qe={result['Max Qe']} within capacities; "
                   f"{result['Rejections']} rejections recorded.")


def verify_infinite_capacity_no_rejections():
    results = _run_many(BASELINE_MODEL, UNIFORM, 10)
    return _result("No rejections at default capacity", (results["Rejections"] == 0).all(),
                   "No rejections occurred under the default effectively infinite capacities.")


def verify_stale_abandonment_ignored():
    model = apply_changes(BASELINE_MODEL,
                          {"lambda_1": 0.0, "lambda_2": 0.0, "lambda_e": 0.0})
    sim = Simulation(model, UNIFORM, 0.0, 10.0, seed=11)
    participant = sim.new_participant("A1")
    sim.attempt_admission(participant, "A1")
    sim.queue_A1.remove(participant)
    sim.status[participant] = "matched"
    sim.abandon(participant)
    return _result("Stale abandonment event",
                   sim.abandonments["A1"] == 0 and sim.status[participant] == "matched",
                   "An abandonment event for an already matched applicant changed nothing.")


def verify_uniform_selection_within_class():
    """Each waiting member should have the same chance of selection under the uniform rule."""
    model = apply_changes(BASELINE_MODEL,
                          {"lambda_1": 0.0, "lambda_2": 0.0, "lambda_e": 0.0})
    sim = Simulation(model, UNIFORM, 0.0, 10.0, seed=5)
    counts = {i: 0 for i in range(5)}
    trials = 20_000
    for _ in range(trials):
        queue = __import__("collections").deque(range(5))
        counts[sim.remove_random(queue)] += 1
    expected = trials / 5
    worst = max(abs(c - expected) / expected for c in counts.values())
    return _result("Uniform selection within class", worst < 0.05,
                   f"Largest deviation from equal selection frequency: {100 * worst:.2f}%.")


def verify_seed_reproducibility():
    first = _run(BASELINE_MODEL, UNIFORM, seed=31415)
    second = _run(BASELINE_MODEL, UNIFORM, seed=31415)
    third = _run(BASELINE_MODEL, UNIFORM, seed=27182)
    return _result("Seed reproducibility", first == second and first != third,
                   "Identical seeds reproduced identical results; a different seed did not.")


def verify_littles_law():
    """Each queue should satisfy L = lambda_eff * W over a sufficiently long run."""
    results = _run_many(BASELINE_MODEL, UNIFORM, 10, horizon=20_000.0, warm_up=550.0)
    worst = max(abs(results[f"Little Error {label} (%)"].mean())
                for label in ("A1", "A2", "Employer"))
    return _result("Little's Law", worst < 2.0,
                   f"Largest mean discrepancy between L and lambda x W across the "
                   f"three queues: {worst:.3f}%.")


def run_verification_suite():
    tests = (
        verify_flow_conservation,
        verify_zero_matching_probability,
        verify_certain_matching,
        verify_no_employer_arrivals,
        verify_no_applicant_arrivals,
        verify_single_class_reduction,
        verify_priority_dominance,
        verify_symmetric_classes,
        verify_policy_weight_monotonicity,
        verify_capacity_binding,
        verify_infinite_capacity_no_rejections,
        verify_stale_abandonment_ignored,
        verify_uniform_selection_within_class,
        verify_seed_reproducibility,
        verify_littles_law,
    )
    results = [test() for test in tests]
    return pd.DataFrame({
        "Test": [r.test for r in results],
        "Passed": [r.passed for r in results],
        "Evidence": [r.evidence for r in results],
    })


def assert_verification_suite():
    results = run_verification_suite()
    failed = results.loc[~results["Passed"]]
    if not failed.empty:
        messages = "; ".join(f"{row.Test}: {row.Evidence}" for row in failed.itertuples(index=False))
        raise AssertionError(f"Verification suite failed: {messages}")
    return results
