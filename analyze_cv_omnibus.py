#!/usr/bin/env python3
"""
Omnibus comparison across >=3 CV experiment configurations, respecting the
paired (repeated-measures) structure of the cross-validation splits.

Motivation
----------
The configurations are evaluated on the SAME cross-validation splits (one row
per "Shuffle number" per config). A plain one-way (between-groups) ANOVA --
e.g. the spreadsheet computation in lizard-paper-scratch/anova/ -- ignores that
pairing, so its within-group sum of squares is inflated by the fact that some
splits are simply harder than others, and it is badly under-powered. The right
omnibus test treats the split as a repeated-measures block.

This script reports, per metric, across all configured experiments:
  1) Per-config descriptives (mean +/- std) on the common paired splits
  2) Friedman test (non-parametric repeated-measures omnibus) -- preferred when
     split-level errors have heavy tails / outliers
  3) Repeated-measures ANOVA (parametric; removes between-split variance)
  4) Naive one-way ANOVA (between-groups) -- reported ONLY to show how much the
     pairing matters; do not cite this one.

Pairing: inner join on PAIR_KEY_COLUMN across every config, then drop any split
with a missing value in any config, so all configs are compared on an identical
set of splits.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:  # pragma: no cover
    raise ImportError("This script requires scipy. Install via: pip install scipy") from e


# =========================
# Configuration (edit here)
# =========================

# Ordered {display label: experiment_id}. The experiment_id maps to
# cv_results_{experiment_id}.csv. Add/remove configs as needed.
CONFIGS = {
    "DLC": "control",
    "LLL": "ll_0d05",
    "THT": "tht_gt_1d05",
    "LLL+THT": "tht_gt_1d05__ll_0d05",
}

PAIR_KEY_COLUMN = "Shuffle number"

# Metrics to run the omnibus on (column names use the "{set}__{metric}" convention).
METRICS = ["all__test rmse", "all__test mAP"]

DECIMALS = 4


def load_aligned(metric: str) -> pd.DataFrame:
    """Return an (n_splits x n_configs) DataFrame of `metric`, inner-joined on
    PAIR_KEY_COLUMN across all CONFIGS and with any incomplete split dropped."""
    series = {}
    for label, eid in CONFIGS.items():
        path = f"cv_results_{eid}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results file for '{label}': {path}")
        df = pd.read_csv(path)
        if PAIR_KEY_COLUMN not in df.columns:
            raise KeyError(f"'{PAIR_KEY_COLUMN}' not in {path}")
        if metric not in df.columns:
            raise KeyError(f"'{metric}' not in {path}")
        if df[PAIR_KEY_COLUMN].duplicated().any():
            raise ValueError(f"Duplicate '{PAIR_KEY_COLUMN}' in {path}")
        series[label] = df.set_index(PAIR_KEY_COLUMN)[metric]
    aligned = pd.DataFrame(series).dropna()
    return aligned


def repeated_measures_anova(X: np.ndarray) -> dict:
    """One-way repeated-measures ANOVA on an (n_subjects x k_conditions) matrix.
    Subjects (= CV splits) are the repeated-measures blocks."""
    n, k = X.shape
    grand = X.mean()
    cond_means = X.mean(axis=0)
    subj_means = X.mean(axis=1)

    ss_conditions = n * np.sum((cond_means - grand) ** 2)
    ss_subjects = k * np.sum((subj_means - grand) ** 2)
    ss_total = np.sum((X - grand) ** 2)
    ss_error = ss_total - ss_conditions - ss_subjects

    df_conditions = k - 1
    df_error = (k - 1) * (n - 1)
    ms_conditions = ss_conditions / df_conditions
    ms_error = ss_error / df_error
    F = ms_conditions / ms_error
    p = float(stats.f.sf(F, df_conditions, df_error))
    return {"F": float(F), "df1": df_conditions, "df2": df_error, "p": p}


def main():
    print("=" * 78)
    print("Omnibus comparison across configurations (paired CV splits)")
    print("=" * 78)
    print("Configs:", ", ".join(f"{k} -> cv_results_{v}.csv" for k, v in CONFIGS.items()))
    print(f"Pairing key: {PAIR_KEY_COLUMN}")

    for metric in METRICS:
        M = load_aligned(metric)
        n, k = M.shape
        cols = list(CONFIGS.keys())
        X = M[cols].to_numpy(dtype=float)

        print("\n" + "=" * 78)
        print(f"METRIC: {metric}   (n={n} common paired splits, k={k} configs)")
        print("=" * 78)
        print("Per-config mean +/- std on the common splits:")
        for c in cols:
            print(f"  {c:10s}  {M[c].mean():.{DECIMALS}f} +/- {M[c].std(ddof=1):.{DECIMALS}f}")

        # Non-parametric repeated-measures omnibus
        fried = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
        # Parametric repeated-measures ANOVA
        rm = repeated_measures_anova(X)
        # Naive one-way ANOVA (ignores pairing) -- for contrast only
        ow_F, ow_p = stats.f_oneway(*[X[:, j] for j in range(k)])

        print("\nOmnibus tests across the {} configurations:".format(k))
        print(f"  Friedman (non-param, paired)     : chi2={fried.statistic:.4f}, "
              f"df={k-1}, p={fried.pvalue:.4f}")
        print(f"  Repeated-measures ANOVA (paired) : F({rm['df1']},{rm['df2']})="
              f"{rm['F']:.4f}, p={rm['p']:.4f}")
        print(f"  One-way ANOVA (IGNORES pairing)  : F({k-1},{n*k-k})="
              f"{ow_F:.4f}, p={ow_p:.4f}   <-- under-powered; do not cite")

    print("\nDone.")


if __name__ == "__main__":
    main()
