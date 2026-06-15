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


def _var_label(name):
    """Short code as used in the manuscript tables (e.g. 'dh_38'), dropping the
    parenthetical description for a compact legend entry while keeping the
    target's readable name."""
    return name.split(" ")[0]


def _panel(ax, df, col, ylabel, V_list, colors):
    """Plot one line per variable, each in its own colour. The target variable
    is drawn thicker and in black as a reference. Returns ordered (handle, label)
    pairs for the legend (only needed once per row)."""
    handles, labels = [], []
    for var, sub in df.groupby("var"):
        sub = sub.sort_values("V")
        name = sub["name"].iloc[0]
        is_target = var == PREDINDEX
        color = "black" if is_target else colors[var]
        lw = 2.2 if is_target else 1.3
        (line,) = ax.plot(sub["V"], sub[col], marker="o", markersize=4,
                          color=color, linewidth=lw, alpha=0.9,
                          zorder=3 if is_target else 2)
        label = _var_label(name) + (" (target)" if is_target else "")
        handles.append(line)
        labels.append(label)
    ax.set_xlabel("V (visit-sample size)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(V_list)
    ax.grid(True, alpha=0.3)
    return handles, labels


def main():
    d1 = pd.read_csv(D1_CSV)
    d2 = pd.read_csv(D2_CSV)
    V1 = sorted(d1["V"].unique())
    V2 = sorted(d2["V"].unique())

    # One distinct colour per (non-target) variable, shared within a dataset.
    cmap = plt.cm.tab10
    n_vars = int(max(d1["var"].max(), d2["var"].max())) + 1
    colors = {j: cmap(j % 10) for j in range(n_vars)}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))

    h1, l1 = _panel(axes[0, 0], d1, "mean_p", "Mean p-value", V1, colors)
    _panel(axes[0, 1], d1, "rej_05", r"Rejection rate at $\alpha=0.05$", V1, colors)
    axes[0, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[0, 0].set_title("LORA D1 (ghq_b_sum)", fontsize=10)
    axes[0, 1].set_title("LORA D1 (ghq_b_sum)", fontsize=10)

    h2, l2 = _panel(axes[1, 0], d2, "mean_p", "Mean p-value", V2, colors)
    _panel(axes[1, 1], d2, "rej_05", r"Rejection rate at $\alpha=0.05$", V2, colors)
    axes[1, 1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    axes[1, 0].set_title("LORA D2 (ghq_sum)", fontsize=10)
    axes[1, 1].set_title("LORA D2 (ghq_sum)", fontsize=10)

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)

    # Per-row legend to the right of each row, listing every variable by its
    # table code (D1 and D2 have different variables, so two legends).
    axes[0, 1].legend(h1, l1, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=7.5, title="D1 variables", title_fontsize=8,
                      frameon=False)
    axes[1, 1].legend(h2, l2, loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=7.5, title="D2 variables", title_fontsize=8,
                      frameon=False)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined LORA V-sweep figure to {OUT_PATH}")


if __name__ == "__main__":
    main()
