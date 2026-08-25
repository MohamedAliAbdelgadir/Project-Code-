"""
Defines model parameters, policies and experiment designs, with all rates expressed relative to lamda_e =1.
"""

from dataclasses import dataclass, replace

INFINITE_CAPACITY = 10 ** 9


@dataclass
class ModelParameters:
    lambda_1: float   # Type 1 applicant arrival rate
    lambda_2: float   # Type 2 applicant arrival rate
    lambda_e: float   # employer arrival rate, fixed at 1 by normalisation
    q_1: float        # per-pair matching probability, Type 1 with employer
    q_2: float        # per-pair matching probability, Type 2 with employer
    theta_1: float    # Type 1 abandonment rate
    theta_2: float    # Type 2 abandonment rate
    theta_e: float    # employer abandonment rate
    k_1: int = INFINITE_CAPACITY
    k_2: int = INFINITE_CAPACITY
    k_e: int = INFINITE_CAPACITY


# ---------------------------------------------------------------------------
# Baseline 
# ---------------------------------------------------------------------------
BASELINE_MODEL = ModelParameters(
    lambda_1=0.6,
    lambda_2=0.4,
    lambda_e=1.0,
    q_1=0.25,
    q_2=0.10,
    theta_1=0.05,
    theta_2=0.05,
    theta_e=0.05,
)

WARM_UP = 550.0
OBSERVATION_HORIZON = 80_000.0
REPLICATIONS = 50
BASE_SEED = 17_000

# ---------------------------------------------------------------------------
# Policy family: A controls the relative priority given to class 1 when both classes have acceptable candidates.
#     P(class 1) = A * X1 / (A * X1 + X2)
# ---------------------------------------------------------------------------
PRIORITY_1 = float("inf")
UNIFORM = 1.0
PRIORITY_2 = 0.0

NAMED_POLICIES = {
    "priority_1": PRIORITY_1,
    "uniform": UNIFORM,
    "priority_2": PRIORITY_2,
}

POLICY_LABELS = {
    PRIORITY_1: "Priority to Type 1",
    UNIFORM: "Uniform",
    PRIORITY_2: "Priority to Type 2",
}


A_SWEEP = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0, PRIORITY_1)


MAIN_POLICIES = (PRIORITY_2, UNIFORM, PRIORITY_1)


def policy_label(policy_A):
    if policy_A in POLICY_LABELS:
        return POLICY_LABELS[policy_A]
    return f"A = {policy_A:g}"


KEY_METRICS = (
    "Throughput",
    "Matching Split p1",
    "A1 Throughput",
    "A2 Throughput",
    "Successful Scan Rate",
    "Abandonment Rate",
    "A1 Abandonment Rate",
    "A2 Abandonment Rate",
    "Employer Abandonment Rate",
    "Average Wait A1",
    "Average Wait A2",
    "Average Wait Employer",
    "Average Q1",
    "Average Q2",
    "Average Qe",
)


def apply_changes(model: ModelParameters, changes: dict) -> ModelParameters:

    changes = dict(changes)
    if "theta_applicants" in changes:
        value = changes.pop("theta_applicants")
        changes["theta_1"] = value
        changes["theta_2"] = value
    if "k_applicants" in changes:
        value = int(changes.pop("k_applicants"))
        changes["k_1"] = value
        changes["k_2"] = value
    if "composition" in changes:
 
        p1 = changes.pop("composition")
        total = model.lambda_1 + model.lambda_2
        changes["lambda_1"] = p1 * total
        changes["lambda_2"] = (1.0 - p1) * total
    return replace(model, **changes)



# ---------------------------------------------------------------------------
FACTORS = {
    "lambda_1": {
        "name": "Type 1 arrival rate",
        "symbol": "lambda_1",
        "levels": (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
    },
    "lambda_2": {
        "name": "Type 2 arrival rate",
        "symbol": "lambda_2",
        "levels": (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
    },
    "q_1": {
        "name": "Type 1 matching probability",
        "symbol": "q_1",
        "levels": (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    },
    "q_2": {
        "name": "Type 2 matching probability",
        "symbol": "q_2",
        "levels": (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    },
    "theta_applicants": {
        "name": "Applicant abandonment rate",
        "symbol": "theta_1 = theta_2",
        "levels": (0.02, 0.05, 0.1, 0.2, 0.5),
    },

    "theta_1": {
        "name": "Class 1 abandonment rate",
        "symbol": "theta_1",
        "levels": (0.02, 0.05, 0.1, 0.2, 0.5),
    },
    "theta_2": {
        "name": "Class 2 abandonment rate",
        "symbol": "theta_2",
        "levels": (0.02, 0.05, 0.1, 0.2, 0.5),
    },
    "theta_e": {
        "name": "Employer abandonment rate",
        "symbol": "theta_e",
        "levels": (0.02, 0.05, 0.1, 0.2, 0.5),
    },

    "composition": {
        "name": "Class 1 share of applicants",
        "symbol": "p_1",
        "levels": (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
    },

    "composition_equal_q": {
        "name": "Class 1 share of applicants, equal compatibility",
        "symbol": "p_1",
        "levels": (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
        "varies": "composition",
        "fixed": {"q_1": 0.25, "q_2": 0.25},
    },
}
