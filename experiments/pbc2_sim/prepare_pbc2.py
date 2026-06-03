"""Prepare the PBC2 dataset (Mayo Clinic primary biliary cirrhosis longitudinal
data) for use with MiniTransformer's existing pipeline.

The raw data lives in R's `survival` package as `pbcseq` (1945 observations on
312 patients, the canonical "PBC2" dataset of the joint-modelling literature).
We extract it via a one-line Rscript invocation, then do all preprocessing in
Python: binarisation at clinical thresholds, sequence filtering, truncation to
maxlen=10 visits, and column ordering to match `src/data_preparation.py::load_real_data`.

Output (CSV in `data/pbc2/pbc2_binarised.csv`):
    id, t, ascites, hepatomegaly, spiders, edema_present,
        albumin_low, alkphos_high, ast_high, platelet_low, protime_high,
        bili_high
where `bili_high` (target) is in the last column to match the predindex=9
convention used for the LORA datasets.

Usage:  python notebooks/prepare_pbc2.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)

import pandas as pd
import numpy as np


# ----------------------------- configuration -------------------------------- #
OUT_DIR = "data/pbc2"
OUT_PATH = os.path.join(OUT_DIR, "pbc2_binarised.csv")

# Clinical thresholds for binarising continuous markers
ALBUMIN_THRESHOLD = 3.5    # < 3.5 g/dl is low
ALKPHOS_THRESHOLD = 1500   # > 1500 U/l is elevated
AST_THRESHOLD = 100        # > 100 U/l is elevated (a.k.a. SGOT)
PLATELET_THRESHOLD = 150   # < 150 (10^9/l) is low
PROTIME_THRESHOLD = 11     # > 11 s is prolonged
BILI_THRESHOLD = 2         # > 2 mg/dl is elevated (target)

# Inclusion criteria
MIN_VISITS = 3             # MiniTransformer requires sequences of length >= 3
MAX_VISITS = 10            # model uses precomputed positional matrices of size 10


# ---------------------- step 1: pull raw data from R ------------------------ #
def extract_pbcseq_via_rscript(out_csv):
    """Invoke Rscript to dump survival::pbcseq to a CSV.

    R is used purely as a one-off data source; all subsequent processing is in
    Python. This keeps the workflow Python-driven while sourcing the canonical
    PBC2 data straight from `survival`.
    """
    rscript_path = shutil.which("Rscript")
    if rscript_path is None:
        sys.exit(
            "Rscript not found on PATH. Install R + the 'survival' package, "
            "or replace this step with another source of the pbcseq data."
        )
    r_code = (
        "suppressMessages(library(survival));"
        "data(pbcseq, package='survival');"
        f"write.csv(pbcseq, '{out_csv}', row.names=FALSE, na='');"
    )
    subprocess.run([rscript_path, "-e", r_code], check=True, capture_output=True)


def load_raw_pbcseq():
    """Returns the raw pbcseq DataFrame (1945 rows). Caches the extraction in a
    temp file so repeated runs in the same session don't re-shell to R."""
    cache = os.path.join(tempfile.gettempdir(), "pbcseq_raw.csv")
    if not os.path.exists(cache):
        print(f"Extracting survival::pbcseq via Rscript -> {cache}")
        extract_pbcseq_via_rscript(cache)
    return pd.read_csv(cache)


# ---------------------- step 2: binarise + filter --------------------------- #
def binarise(raw):
    """Apply clinical thresholds and return a 12-column dataframe in the
    canonical (id, t, 9 features, target) order."""
    df = pd.DataFrame({
        "id":           raw["id"].astype(int),
        "day":          raw["day"].astype(int),
        # Already binary in pbcseq:
        "ascites":      raw["ascites"].astype("Int64"),
        "hepatomegaly": raw["hepato"].astype("Int64"),
        "spiders":      raw["spiders"].astype("Int64"),
        # Edema is in {0, 0.5, 1}; treat any positive value as present.
        "edema_present": (raw["edema"] > 0).astype("Int64"),
        # Continuous markers binarised at clinical thresholds.
        "albumin_low":  (raw["albumin"] < ALBUMIN_THRESHOLD).astype("Int64"),
        "alkphos_high": (raw["alk.phos"] > ALKPHOS_THRESHOLD).astype("Int64"),
        "ast_high":     (raw["ast"] > AST_THRESHOLD).astype("Int64"),
        "platelet_low": (raw["platelet"] < PLATELET_THRESHOLD).astype("Int64"),
        "protime_high": (raw["protime"] > PROTIME_THRESHOLD).astype("Int64"),
        "bili_high":    (raw["bili"] > BILI_THRESHOLD).astype("Int64"),
    })

    n_before = len(df)
    df = df.dropna()
    df = df.astype({c: int for c in df.columns if c not in {"id", "day"}})
    print(f"  Dropped {n_before - len(df)} rows with missing values; "
          f"kept {len(df)}.")
    return df


def filter_and_index(df):
    """Sort by (id, day), assign per-patient visit index t = 1..k, drop
    patients with fewer than MIN_VISITS visits, and truncate to the last
    MAX_VISITS visits per patient."""
    df = df.sort_values(["id", "day"]).reset_index(drop=True)
    df["t"] = df.groupby("id").cumcount() + 1

    visits_per_patient = df.groupby("id").size()
    keep = visits_per_patient[visits_per_patient >= MIN_VISITS].index
    df = df[df["id"].isin(keep)]
    print(f"  Kept {df['id'].nunique()} patients with >= {MIN_VISITS} visits "
          f"({len(df)} rows).")

    # Truncate to last MAX_VISITS per patient and re-index t
    def _last_k(g):
        if len(g) > MAX_VISITS:
            g = g.iloc[-MAX_VISITS:]
        return g
    df = df.groupby("id", group_keys=False).apply(_last_k)
    df["t"] = df.groupby("id").cumcount() + 1
    print(f"  After truncation to last {MAX_VISITS} visits: "
          f"{df['id'].nunique()} patients, {len(df)} rows.")
    return df


# ----------------------------- main ----------------------------------------- #
def main():
    print("=== Preparing PBC2 (survival::pbcseq) for MiniTransformer ===")
    raw = load_raw_pbcseq()
    print(f"Raw pbcseq: {raw.shape[0]} rows, {raw.shape[1]} columns, "
          f"{raw['id'].nunique()} patients")

    print("Binarising at clinical thresholds...")
    df = binarise(raw)

    print(f"Filtering and indexing (>= {MIN_VISITS} visits, <= {MAX_VISITS} kept)...")
    df = filter_and_index(df)

    # Final canonical column order: id, t, 9 features, target last.
    # `ascites` is placed last (predindex=9) because it is the most clinically
    # severe endpoint among the binarised PBC2 markers (decompensated cirrhosis
    # event), and beating regression on a rare binary endpoint is a strict gate.
    feature_cols = [
        "hepatomegaly", "spiders", "edema_present",
        "albumin_low", "alkphos_high", "ast_high", "platelet_low", "protime_high",
        "bili_high",
        "ascites",   # target -> predindex=9
    ]
    out = df[["id", "t", *feature_cols]]

    # Sanity-check: all features must be in {0, 1}
    for col in feature_cols:
        assert set(out[col].unique()).issubset({0, 1}), (
            f"Column {col} has non-binary values: {sorted(out[col].unique())}"
        )

    print("\nMarginal frequencies (proportion = 1):")
    for col in feature_cols:
        print(f"  {col:<15s} {out[col].mean():.3f}")

    seq_lengths = out.groupby("id").size()
    print(f"\nSequence length distribution (visits per patient):")
    print(f"  min={seq_lengths.min()}  median={int(seq_lengths.median())}  "
          f"max={seq_lengths.max()}  mean={seq_lengths.mean():.2f}")
    print(f"\nTotal patients: {out['id'].nunique()}, total rows: {len(out)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
