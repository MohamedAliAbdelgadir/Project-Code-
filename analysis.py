"""
Produces summary statistics and confidence intervals across replications.
"""

import numpy as np
import pandas as pd
from scipy.stats import t


def summarise_replications(replications: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """One row per numeric metric: mean, std dev, 95% CI, relative half-width."""
    numeric = replications.select_dtypes(include=np.number)
    n = len(numeric)
    critical_value = t.ppf(1 - (1 - confidence) / 2, df=n - 1) if n > 1 else np.nan

    summary = pd.DataFrame({
        "Metric": numeric.columns,
        "Mean": numeric.mean().values,
        "Std Dev": numeric.std(ddof=1).values,
        "Minimum": numeric.min().values,
        "Maximum": numeric.max().values,
    })
    summary["Std Error"] = summary["Std Dev"] / np.sqrt(n)
    summary["CI Half Width"] = critical_value * summary["Std Error"]
    summary["CI Lower"] = summary["Mean"] - summary["CI Half Width"]
    summary["CI Upper"] = summary["Mean"] + summary["CI Half Width"]
    summary["Relative Half Width (%)"] = np.where(
        summary["Mean"] != 0,
        100 * summary["CI Half Width"] / np.abs(summary["Mean"]),
        np.nan,
    )
    return summary
