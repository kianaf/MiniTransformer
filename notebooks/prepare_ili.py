"""Prepare the ILI (Influenza-Like Illness) benchmark for use with
MiniTransformer's existing pipeline.

Source: weekly CDC ILI counts 2002-2021, packaged by Nixtla's
``datasetsforecast.long_horizon.LongHorizon`` (group='ILI'). 7 variables x
966 weeks. The dataset is the medical entry on the reviewer's §3.2 list of
"standard short/medium-horizon time series benchmarks" (Zhou et al., 2021,
Informer; the same suite that introduced the ETT family).

Adaptation to a cohort-style pipeline
-------------------------------------
ILI is a single long multivariate series, not a cohort of short sequences.
To fit MiniTransformer's pipeline (many short sequences, one-step-ahead
prediction at the last visit), we slice the series into rolling windows of
length ``WINDOW`` with stride ``STRIDE`` and treat each window as one
"patient sequence". We use ``WINDOW=10`` (matching the model's precomputed
positional matrices) and ``STRIDE=5`` (50% overlap, giving ~190 windows --
similar in cohort size to PBC2). This is an honest reshape that we
acknowledge in the manuscript; the alternative is fully non-overlapping
windows (~96 windows, much smaller cohort).

The values are already z-scored in the Nixtla packaging. We binarise each
variable at its own median so the existing permutation test (which uses
unit-vector contexts and binary visit patterns) applies unchanged.

Output (CSV):
    id, t, %weighted_ILI, %unweighted_ILI, AGE_0_4, AGE_5_24,
        ILITOTAL, NUM_PROVIDERS, OT
where ``OT`` (overall ILI target) is the last column to match the
``predindex = p - 1`` convention used for LORA and PBC2.

Usage:  python notebooks/prepare_ili.py
"""
from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd

# ----------------------------- configuration -------------------------------- #
OUT_DIR = "data/ili"
OUT_PATH = os.path.join(OUT_DIR, "ili_binarised.csv")

WINDOW = 10
STRIDE = 5

# Order of variables in the output CSV (matches Informer benchmark with OT last).
VAR_ORDER_LONG = [
    "% WEIGHTED ILI",
    "%UNWEIGHTED ILI",
    "AGE 0-4",
    "AGE 5-24",
    "ILITOTAL",
    "NUM. OF PROVIDERS",
    "OT",  # target: overall ILI rate
]
VAR_ORDER_SHORT = [
    "weighted_ILI", "unweighted_ILI",
    "AGE_0_4", "AGE_5_24",
    "ILITOTAL", "NUM_PROVIDERS",
    "OT",
]


# ---------------------- step 1: pull raw data ------------------------------- #
def load_raw_ili():
    """Returns a wide DataFrame indexed by date, with one column per variable
    (in the canonical Informer order)."""
    from datasetsforecast.long_horizon import LongHorizon
    cache_dir = os.path.expanduser("~/.cache/datasetsforecast")
    long_df, _, _ = LongHorizon.load(directory=cache_dir, group="ILI")
    # Pivot to wide: rows = date, cols = variable, value = y.
    wide = long_df.pivot(index="ds", columns="unique_id", values="y")
    wide = wide[VAR_ORDER_LONG]                # canonical column order
    wide.columns = VAR_ORDER_SHORT             # rename for readability
    wide = wide.sort_index()
    return wide


# ---------------------- step 2: window + binarise --------------------------- #
def make_windows(wide_df, window=WINDOW, stride=STRIDE):
    """Slice the long series into rolling windows. Returns a list of DataFrames
    (one per window) each with ``window`` rows and the original variable
    columns."""
    arr = wide_df.values  # (T, p) float
    T, p = arr.shape
    windows = []
    for start in range(0, T - window + 1, stride):
        windows.append(arr[start:start + window])
    return np.stack(windows), wide_df.columns.tolist()


def binarise_per_variable(arr, columns):
    """Median-split each column. ``arr`` is (n_windows, window, p); we use the
    global median across all observations of each variable so the threshold
    is consistent across windows. Returns a (n_windows, window, p) int array
    with values in {0, 1}."""
    n_windows, window, p = arr.shape
    flat = arr.reshape(-1, p)
    medians = np.median(flat, axis=0)
    bin_arr = (arr > medians[None, None, :]).astype(int)
    return bin_arr, medians


# ----------------------------- main ----------------------------------------- #
def main():
    print("=== Preparing ILI for MiniTransformer ===")
    wide = load_raw_ili()
    print(f"Raw ILI: {len(wide)} weeks x {wide.shape[1]} variables")
    print(f"Variables: {list(wide.columns)}")

    print(f"Slicing into rolling windows (window={WINDOW}, stride={STRIDE}) ...")
    windows, columns = make_windows(wide)
    n_windows = windows.shape[0]
    print(f"  -> {n_windows} windows of shape ({WINDOW}, {windows.shape[2]})")

    print("Binarising at the per-variable median ...")
    bin_arr, medians = binarise_per_variable(windows, columns)
    print("  thresholds (medians):")
    for col, m in zip(columns, medians):
        frac_pos = bin_arr.reshape(-1, len(columns)).mean(axis=0)
    for col, m, f in zip(columns, medians, bin_arr.reshape(-1, len(columns)).mean(axis=0)):
        print(f"    {col:<18s} median={m:+.3f}  fraction_positive={f:.3f}")

    # Long-format DataFrame: id, t, var1..varN
    rows = []
    for i in range(n_windows):
        for t_idx in range(WINDOW):
            row = {"id": i, "t": t_idx + 1}
            for k, col in enumerate(columns):
                row[col] = int(bin_arr[i, t_idx, k])
            rows.append(row)
    out = pd.DataFrame(rows)

    seq_lengths = out.groupby("id").size()
    print(f"\nWindow-length distribution: min={seq_lengths.min()}  "
          f"max={seq_lengths.max()}  mean={seq_lengths.mean():.2f}")
    print(f"Total windows: {n_windows}, total rows: {len(out)}")
    print(f"\nMarginal frequencies (proportion = 1):")
    for col in columns:
        print(f"  {col:<18s} {out[col].mean():.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
