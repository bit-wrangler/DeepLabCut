#!/usr/bin/env python3
"""
Paired CV experiment comparison (Control vs Method) using Shuffle number for pairing.

- Reads:
    cv_results_{EXPERIMENT_ID_CONTROL}.csv
    cv_results_{EXPERIMENT_ID_METHOD}.csv
- Pairs rows by: PAIR_KEY_COLUMN (default: "Shuffle number")
  (expects one row per Shuffle number per file; duplicates are rejected)
- Assumes metric columns follow your convention:
    "{landmark_set}__{metric}"
  Tolerates extra columns; only analyzes requested landmark sets + metrics.

Outputs:
1) Control and Method mean ± std (descriptives)
2) Paired deltas (Method - Control): mean, sd, SE, 95% CI
3) Paired t-test p-value
4) Wilcoxon signed-rank p-value (rank-based; robust to outlier splits)
5) Sign-flip permutation p-value (optional)
6) Estimated required n (paired) for Δ=0.25 and Δ=0.5 at POWER_TARGET, alpha=ALPHA
   based on observed sd of paired differences.

Note: n here is number of paired Shuffle numbers.
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as e:
    raise ImportError("This script requires scipy. Install via: pip install scipy") from e


# =========================
# Configuration (edit here)
# =========================

EXPERIMENT_ID_CONTROL = "control"
EXPERIMENT_ID_METHOD = "ll_0d05"

PAIR_KEY_COLUMN = "Shuffle number"

LANDMARK_SET_NAMES = ["all", "truncated", "non_truncated"]
METRIC_NAMES = ["test rmse", "test rmse_pcutoff", "test mAP", "test mAR"]

ALPHA = 0.05            # two-sided
POWER_TARGET = 0.80     # used for sample size estimates for Δ targets below

DELTA_TARGETS = [0.25, 0.50]  # target mean differences to "clear" (Method - Control)

# Permutation test (sign-flip). Set N_PERMUTATIONS=0 to disable.
N_PERMUTATIONS = 20000
PERMUTATION_SEED = 0

DECIMALS = 4


def _z(p: float) -> float:
    return float(stats.norm.ppf(p))


def paired_required_n(sd_diff: float, delta: float, alpha: float, power: float) -> int:
    """
    Approximate required number of paired samples (n) for two-sided paired mean test:
        n ≈ ((z_{1-α/2} + z_{power}) * sd_diff / delta)^2
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if sd_diff <= 0 or not np.isfinite(sd_diff):
        return 0
    z_alpha = _z(1 - alpha / 2)
    z_pow = _z(power)
    n = ((z_alpha + z_pow) * sd_diff / delta) ** 2
    return int(math.ceil(n))


def sign_flip_permutation_pvalue(diffs: np.ndarray, n_perm: int, seed: int) -> float:
    """
    Two-sided sign-flip permutation test for mean(diffs)=0.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    obs = float(np.mean(diffs))

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n), replace=True)
    perm_means = (signs * diffs).mean(axis=1)

    p = (np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def validate_unique_pair_key(df: pd.DataFrame, key: str, label: str) -> None:
    if key not in df.columns:
        raise KeyError(f"Missing required pairing column '{key}' in {label} CSV.")
    dup = df[key].duplicated(keep=False)
    if dup.any():
        bad = df.loc[dup, key].tolist()
        bad_unique = sorted(set(bad))
        preview = bad_unique[:25]
        raise ValueError(
            f"Non-unique pairing key '{key}' in {label} CSV. "
            f"Found duplicates for values: {preview}{' ...' if len(bad_unique) > 25 else ''}"
        )


def build_metric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for ls in LANDMARK_SET_NAMES:
        for m in METRIC_NAMES:
            c = f"{ls}__{m}"
            if c in df.columns:
                cols.append(c)
    return cols


def format_mean_std(mean: float, std: float) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "N/A"
    return f"{mean:.{DECIMALS}f} ± {std:.{DECIMALS}f}"


def main():
    control_file = f"cv_results_{EXPERIMENT_ID_CONTROL}.csv"
    method_file = f"cv_results_{EXPERIMENT_ID_METHOD}.csv"

    if not os.path.exists(control_file):
        raise FileNotFoundError(
            f"Control results file not found: {control_file}\n"
            f"Please check EXPERIMENT_ID_CONTROL='{EXPERIMENT_ID_CONTROL}'."
        )
    if not os.path.exists(method_file):
        raise FileNotFoundError(
            f"Method results file not found: {method_file}\n"
            f"Please check EXPERIMENT_ID_METHOD='{EXPERIMENT_ID_METHOD}'."
        )

    print(f"Loading control results from: {control_file}")
    control_df = pd.read_csv(control_file)
    print(f"Loaded {len(control_df)} rows\n")

    print(f"Loading method results from: {method_file}")
    method_df = pd.read_csv(method_file)
    print(f"Loaded {len(method_df)} rows\n")

    validate_unique_pair_key(control_df, PAIR_KEY_COLUMN, "control")
    validate_unique_pair_key(method_df, PAIR_KEY_COLUMN, "method")

    control_cols = set(build_metric_columns(control_df))
    method_cols = set(build_metric_columns(method_df))
    common_cols = sorted(control_cols.intersection(method_cols))

    if not common_cols:
        raise ValueError(
            "No common metric columns found to analyze.\n"
            "Expected columns like '{landmark_set}__{metric}'.\n"
            f"LANDMARK_SET_NAMES={LANDMARK_SET_NAMES}\n"
            f"METRIC_NAMES={METRIC_NAMES}"
        )

    # Inner join on pairing key (use only paired runs)
    c = control_df[[PAIR_KEY_COLUMN] + common_cols].copy()
    m = method_df[[PAIR_KEY_COLUMN] + common_cols].copy()
    merged = c.merge(m, on=PAIR_KEY_COLUMN, suffixes=("_control", "_method"), how="inner")

    n_pairs_total = len(merged)
    if n_pairs_total < 2:
        raise ValueError(f"Not enough paired runs after merge on '{PAIR_KEY_COLUMN}'. n_pairs={n_pairs_total}")

    control_only = set(control_df[PAIR_KEY_COLUMN]) - set(method_df[PAIR_KEY_COLUMN])
    method_only = set(method_df[PAIR_KEY_COLUMN]) - set(control_df[PAIR_KEY_COLUMN])
    if control_only or method_only:
        print(
            f"Warning: pairing mismatch on '{PAIR_KEY_COLUMN}': "
            f"{len(control_only)} only-in-control, {len(method_only)} only-in-method. "
            f"Using {n_pairs_total} paired rows (inner join)."
        )

    print(f"{'='*90}")
    print(f"Paired Comparison Summary")
    print(f"{'='*90}")
    print(f"Control: {EXPERIMENT_ID_CONTROL}")
    print(f"Method : {EXPERIMENT_ID_METHOD}")
    print(f"Pairing key: {PAIR_KEY_COLUMN}")
    print(f"Paired runs (rows after merge): {n_pairs_total}")
    print(f"alpha={ALPHA} (two-sided), power target={POWER_TARGET}")
    print(f"Δ targets: {DELTA_TARGETS}")
    print(f"{'='*90}\n")

    rows = []
    for col in common_cols:
        x = merged[f"{col}_control"].to_numpy(dtype=float)
        y = merged[f"{col}_method"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        diffs = y - x
        n = diffs.size
        if n < 2:
            continue

        mean_c, std_c = float(np.mean(x)), float(np.std(x, ddof=1))
        mean_m, std_m = float(np.mean(y)), float(np.std(y, ddof=1))

        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        se_d = sd_d / math.sqrt(n)

        t_stat, p_t = stats.ttest_rel(y, x, nan_policy="omit")

        # Wilcoxon signed-rank: rank-based paired test, robust to a few
        # high-magnitude (outlier) split differences. Raises when all paired
        # differences are zero; report NaN in that degenerate case.
        try:
            _, p_w = stats.wilcoxon(y, x)
            p_w = float(p_w)
        except ValueError:
            p_w = float("nan")

        t_crit = float(stats.t.ppf(1 - ALPHA / 2, df=n - 1))
        ci_lo = mean_d - t_crit * se_d
        ci_hi = mean_d + t_crit * se_d

        rho = float(np.corrcoef(x, y)[0, 1]) if n >= 2 else float("nan")

        p_perm = (
            sign_flip_permutation_pvalue(diffs, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED)
            if N_PERMUTATIONS and N_PERMUTATIONS > 0
            else float("nan")
        )

        n_reqs = {d: paired_required_n(sd_d, d, alpha=ALPHA, power=POWER_TARGET) for d in DELTA_TARGETS}

        ls, metric = col.split("__", 1)
        row = {
            "landmark_set": ls,
            "metric": metric,
            "n_pairs": n,
            "control_mean±std": format_mean_std(mean_c, std_c),
            "method_mean±std": format_mean_std(mean_m, std_m),
            "delta_mean": round(mean_d, DECIMALS),
            "delta_sd": round(sd_d, DECIMALS),
            "delta_95%CI": f"[{ci_lo:.{DECIMALS}f}, {ci_hi:.{DECIMALS}f}]",
            "paired_t_p": float(p_t),
            "wilcoxon_p": p_w,
            "perm_p": float(p_perm),
            "corr_rho": round(rho, 4) if np.isfinite(rho) else float("nan"),
        }
        for d in DELTA_TARGETS:
            row[f"n_req_Δ{d:g}@{int(POWER_TARGET*100)}%"] = n_reqs[d]
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["landmark_set", "metric"]).reset_index(drop=True)

    # Pretty print grouped by landmark set
    for ls, g in summary_df.groupby("landmark_set", sort=False):
        print(f"{ls.upper()} Landmarks")
        print("-" * 90)
        for _, r in g.iterrows():
            print(
                f"{r['metric']:25s} | n={int(r['n_pairs']):3d} | "
                f"ctrl {r['control_mean±std']:>18s} | "
                f"meth {r['method_mean±std']:>18s} | "
                f"Δ {r['delta_mean']:>8} (sd {r['delta_sd']:>8}) | "
                f"CI {r['delta_95%CI']:>22s} | "
                f"p_t={r['paired_t_p']:.3g} | "
                f"p_wilcox={r['wilcoxon_p']:.3g} | "
                f"p_perm={r['perm_p']:.3g} | ρ={r['corr_rho']}"
            )
        print("")

    print(f"{'='*90}")
    print("Complete Results Table (paired deltas are Method - Control)")
    print(f"{'='*90}\n")

    # Compact columns for table output
    base_cols = [
        "landmark_set",
        "metric",
        "n_pairs",
        "control_mean±std",
        "method_mean±std",
        "delta_mean",
        "delta_sd",
        "delta_95%CI",
        "paired_t_p",
        "wilcoxon_p",
        "perm_p",
        "corr_rho",
    ]
    req_cols = [f"n_req_Δ{d:g}@{int(POWER_TARGET*100)}%" for d in DELTA_TARGETS]
    cols = base_cols + req_cols

    print(summary_df[cols].to_string(index=False))

    print(f"\n{'='*90}")
    print("Additional Statistics")
    print(f"{'='*90}")
    print(f"Control rows: {len(control_df)}")
    print(f"Method rows : {len(method_df)}")
    print(f"Paired rows : {n_pairs_total}")
    if control_only:
        print(f"Only-in-control {PAIR_KEY_COLUMN}: {sorted(list(control_only))[:25]}{' ...' if len(control_only)>25 else ''}")
    if method_only:
        print(f"Only-in-method  {PAIR_KEY_COLUMN}: {sorted(list(method_only))[:25]}{' ...' if len(method_only)>25 else ''}")

    # If fold/seed columns exist, print counts (optional)
    for label, df in [("Control", control_df), ("Method", method_df)]:
        has_fold = "fold" in df.columns
        has_seed = "seed" in df.columns
        if has_fold or has_seed:
            print(f"\n{label} split metadata:")
            if has_fold:
                print(f"  Number of folds: {df['fold'].nunique()} | values: {sorted(df['fold'].unique())}")
            if has_seed:
                print(f"  Number of seeds: {df['seed'].nunique()} | values: {sorted(df['seed'].unique())}")

    print("\nDone.")


if __name__ == "__main__":
    main()
