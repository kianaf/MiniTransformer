"""Combined V-sweep figure for Appendix S2: synthetic simulation (top row) and
PBC2-substrate simulation (bottom row), each with mean p-value (left) and
empirical rejection rate at alpha=0.05 (right).

Reads the per-variable summaries already produced by
v_monotonicity_check.py and v_monotonicity_check_pbc2_sim.py (no retraining):
    notebooks/results/v_monotonicity_check/v_sweep_summary.csv
    notebooks/results/v_monotonicity_check_pbc2_sim/v_sweep_summary.csv

Output:
    notebooks/results/v_monotonicity_check/v_sweep_combined.png
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLOR_SIGNAL = "#AE232F"
COLOR_NULL = "#1D4A91"

SIM_CSV = "notebooks/results/v_monotonicity_check/v_sweep_summary.csv"
PBC2_CSV = "notebooks/results/v_monotonicity_check_pbc2_sim/v_sweep_summary.csv"
OUT_PATH = "notebooks/results/v_monotonicity_check/v_sweep_combined.png"


def _panel(ax, df, col, signal_col, label_signal, label_null, ylabel, V_list):
    """Plot one metric column vs V, one line per variable, coloured by signal."""
    for var, sub in df.groupby("var"):
        sub = sub.sort_values("V")
        is_sig = bool(sub[signal_col].iloc[0])
        ax.plot(sub["V"], sub[col], marker="o", markersize=4,
                color=COLOR_SIGNAL if is_sig else COLOR_NULL, alpha=0.85)
    ax.set_xlabel("V (visit-sample size)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(V_list)
    ax.grid(True, alpha=0.3)


def main():
    sim = pd.read_csv(SIM_CSV)
    pbc2 = pd.read_csv(PBC2_CSV)
    V_sim = sorted(sim["V"].unique())
    V_pbc2 = sorted(pbc2["V"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Top row: synthetic simulation (signal flag = is_signal).
    _panel(axes[0, 0], sim, "mean_p", "is_signal", None, None,
           "Mean p-value", V_sim)
    _panel(axes[0, 1], sim, "rej_05", "is_signal", None, None,
           r"Rejection rate at $\alpha=0.05$", V_sim)
    axes[0, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Synthetic simulation", fontsize=12)
    axes[0, 1].set_title("Synthetic simulation", fontsize=12)

    # Bottom row: PBC2-substrate simulation (signal flag = is_trigger).
    _panel(axes[1, 0], pbc2, "mean_p", "is_trigger", None, None,
           "Mean p-value", V_pbc2)
    _panel(axes[1, 1], pbc2, "rej_05", "is_trigger", None, None,
           r"Rejection rate at $\alpha=0.05$", V_pbc2)
    axes[1, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[1, 0].set_title("PBC2-substrate simulation", fontsize=12)
    axes[1, 1].set_title("PBC2-substrate simulation", fontsize=12)

    # Shared legend (colour key only; individual variables are not labelled to
    # keep the figure readable).
    sig_handle = plt.Line2D([], [], color=COLOR_SIGNAL, marker="o",
                            linestyle="-", label="signal / trigger variables")
    null_handle = plt.Line2D([], [], color=COLOR_NULL, marker="o",
                             linestyle="-", label="null variables")
    fig.legend(handles=[sig_handle, null_handle], loc="lower center",
               ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        r"Sensitivity of the permutation test to the visit-sample size $V$ "
        r"($\mathrm{nrepp}=500$)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined V-sweep figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
