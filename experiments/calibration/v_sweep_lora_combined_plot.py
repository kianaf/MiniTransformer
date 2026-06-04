"""Combined V-sweep figure for Appendix S3: LORA D1 (top row) and LORA D2
(bottom row), each with mean p-value (left) and empirical rejection rate at
alpha=0.05 (right).

Reads the per-variable summaries already produced by v_monotonicity_check_lora.py
(no retraining):
    notebooks/results/v_monotonicity_check_lora/ghq_b_sum/v_sweep_summary.csv
    notebooks/results/v_monotonicity_check_lora/ghq_sum/v_sweep_summary.csv

The target variable (predindex=9, the GHQ target used as its own context) is
drawn in red; all candidate context variables are blue.

Output:
    notebooks/results/v_monotonicity_check_lora/v_sweep_lora_combined.png
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import matplotlib.pyplot as plt

COLOR_TARGET = "#AE232F"
COLOR_OTHER = "#1D4A91"
PREDINDEX = 9  # the GHQ target used as its own context

D1_CSV = "notebooks/results/v_monotonicity_check_lora/ghq_b_sum/v_sweep_summary.csv"
D2_CSV = "notebooks/results/v_monotonicity_check_lora/ghq_sum/v_sweep_summary.csv"
OUT_PATH = "notebooks/results/v_monotonicity_check_lora/v_sweep_lora_combined.png"


def _panel(ax, df, col, ylabel, V_list):
    for var, sub in df.groupby("var"):
        sub = sub.sort_values("V")
        color = COLOR_TARGET if var == PREDINDEX else COLOR_OTHER
        ax.plot(sub["V"], sub[col], marker="o", markersize=4, color=color,
                alpha=0.85)
    ax.set_xlabel("V (visit-sample size)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(V_list)
    ax.grid(True, alpha=0.3)


def main():
    d1 = pd.read_csv(D1_CSV)
    d2 = pd.read_csv(D2_CSV)
    V1 = sorted(d1["V"].unique())
    V2 = sorted(d2["V"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5))

    _panel(axes[0, 0], d1, "mean_p", "Mean p-value", V1)
    _panel(axes[0, 1], d1, "rej_05", r"Rejection rate at $\alpha=0.05$", V1)
    axes[0, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[0, 0].set_title("LORA D1 (ghq_b_sum)", fontsize=10)
    axes[0, 1].set_title("LORA D1 (ghq_b_sum)", fontsize=10)

    _panel(axes[1, 0], d2, "mean_p", "Mean p-value", V2)
    _panel(axes[1, 1], d2, "rej_05", r"Rejection rate at $\alpha=0.05$", V2)
    axes[1, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[1, 0].set_title("LORA D2 (ghq_sum)", fontsize=10)
    axes[1, 1].set_title("LORA D2 (ghq_sum)", fontsize=10)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)

    target_handle = plt.Line2D([], [], color=COLOR_TARGET, marker="o",
                               linestyle="-", label="target as its own context")
    other_handle = plt.Line2D([], [], color=COLOR_OTHER, marker="o",
                              linestyle="-", label="candidate context variables")
    fig.legend(handles=[target_handle, other_handle], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined LORA V-sweep figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
