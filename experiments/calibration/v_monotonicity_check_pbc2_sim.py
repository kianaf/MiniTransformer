"""V monotonicity check on the PBC2-substrate controlled simulation
(response to reviewer §3.6(a), companion to the synthetic-simulation check in
v_monotonicity_check.py and the LORA check in v_monotonicity_check_lora.py).

The §3.1.2 controlled simulation runs the §2.3 permutation test on real
binarised PBC2 predictors with a synthetic j1 -> j2 -> j3 target overwriting the
ascites column (j1 = bili_high, j2 = albumin_low). This script asks the §3.6(a)
question on that substrate directly: as the visit-sample size V grows, does the
significance ranking stay a property of the data, or does it drift?

It mirrors v_monotonicity_check.py:
  - trains ONE MiniTransformer on the full PBC2-sim substrate (real-data config,
    NHEADS=8, NCUM=8, 150 epochs), reused across all V (single train);
  - runs the permutation test at predindex = COL_ASCITES for V in {5, 6, 7}
    with nrepp = 500;
  - reports per-variable mean p-value, std, and rejection rates.

Unlike the LORA cohorts (dense real signal, the regime the reviewer worries
about), the PBC2-sim non-trigger predictors are null *by construction* for the
synthetic target, so this substrate sits in the same clean-null regime as the
synthetic simulation while carrying real PBC2 predictor correlations. The two
triggers (bili_high, albumin_low) are the only generative-signal variables.

Outputs (under notebooks/results/v_monotonicity_check_pbc2_sim/):
    v_sweep_summary.csv        per-variable p-value summary across V
    v_sweep_pvals.pkl          raw (V -> (p, nrepp)) p-value matrices
    v_sweep_plots.png          mean p-value, rejection rate vs V
    summary.txt                human-readable text dump
"""
import os
import sys
import time
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data_preparation import load_real_data, collate_function
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
    count_parameters,
)
from src.statistical_testing import statistical_testing
from src.pbc2_substrate import (
    inject_synthetic_target,
    COL_BILI_HIGH,
    COL_ALBUMIN_LOW,
    COL_ASCITES,
)


# ----------------------------- configuration -------------------------------- #
device = torch.device("cpu")

# Model setup (PBC2-sim real-data defaults, matches pbc2_controlled_simulation.py)
maxlen = 10
nheads = 8
ncum = 8
dk = 1
dv = 1

batch_size = 1
learning_rate = 1e-3
lambda_l2 = 1e-3
EPOCHS = int(os.environ.get("VSGP_EPOCHS", 150))

# The synthetic target is injected once, deterministically, with seed=42 (matches
# pbc2_controlled_simulation.py so the substrate is identical).
INJECT_SEED = 42

predindex = COL_ASCITES
true_pattern_idx = (COL_BILI_HIGH, COL_ALBUMIN_LOW)  # the two generative triggers

V_LIST = [int(v) for v in os.environ.get("VSGP_V_LIST", "5,6,7").split(",")]
nrepp = int(os.environ.get("VSGP_NREPP", 500))
seed_train = 42
seed_test = 123456789  # matches pbc2_controlled_simulation.py TEST_SEED

VAR_NAMES = [
    "hepatomegaly", "spiders", "edema_present", "albumin_low",
    "alkphos_high", "ast_high", "platelet_low", "protime_high",
    "bili_high", "y_synthetic",
]

OUT_DIR = "notebooks/results/v_monotonicity_check_pbc2_sim"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------- main ----------------------------------------- #
def main():
    torch.set_printoptions(sci_mode=False, precision=6)

    print("=== V-monotonicity check on the PBC2-substrate simulation ===")
    raw_tensors, max_T = load_real_data("pbc2")
    n_total = len(raw_tensors)
    p = raw_tensors[0].shape[1]
    print(f"Cohort: n={n_total} patients, p={p}, max_T={max_T}")
    print(f"Triggers: j1=bili_high (col {COL_BILI_HIGH}), "
          f"j2=albumin_low (col {COL_ALBUMIN_LOW})")
    print(f"Target overwritten: ascites (col {COL_ASCITES})")
    print(f"V_LIST={V_LIST}  nrepp={nrepp}  predindex={predindex}  EPOCHS={EPOCHS}\n")
    assert p == 10
    assert COL_ASCITES == p - 1

    # 1. Inject the synthetic target once (deterministic, seed=42), matching the
    #    §3.1.2 controlled simulation substrate.
    seqs, z_list, y_list = inject_synthetic_target(raw_tensors, seed=INJECT_SEED)
    flat_y = np.concatenate(y_list)
    print(f"synthetic target marginal: P(y=1) = {flat_y.mean():.3f}\n")

    # 2. Build + train ONE MiniTransformer on the full substrate (reused across V).
    #    Single training run; seed set once so the run is reproducible.
    torch.manual_seed(seed_train)
    np.random.seed(seed_train)

    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

    model = MiniTransformer(
        p, nheads, dk, dv, ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix,
        device,
    ).to(device)
    print(f"Model parameter count: {count_parameters(model)}")

    dataloader = DataLoader(
        seqs, batch_size=batch_size, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    eval_loader = DataLoader(
        seqs, batch_size=len(seqs), shuffle=False,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    t0 = time.time()
    train_mini_transformer(model, dataloader, eval_loader, optimizer, lambda_l2, EPOCHS, device)
    print(f"\nTraining done in {time.time() - t0:.1f}s.\n")

    # 3. V-sweep on the same trained model.
    pvals_by_V = {}
    for V in V_LIST:
        print(f"--- V = {V} (nrepp={nrepp}) ---")
        t0 = time.time()
        avepval, stdpval, ctx, tgts, pval_mat = statistical_testing(
            model, seqs, p, predindex, nrepp, V,
            return_pval_mat=True, seed=seed_test,
        )
        print(f"V={V} done in {time.time() - t0:.1f}s")
        pvals_by_V[V] = pval_mat.detach().cpu().numpy()

    # 4. Aggregate across V.
    summary_rows = []
    for V, mat in pvals_by_V.items():
        for j in range(p):
            if j == predindex:
                continue  # the target itself is not a candidate context
            pvals_j = mat[j]
            summary_rows.append({
                "V": V, "var": j, "name": VAR_NAMES[j],
                "is_trigger": j in true_pattern_idx,
                "mean_p": float(pvals_j.mean()),
                "std_p": float(pvals_j.std()),
                "rej_05": float((pvals_j < 0.05).mean()),
                "rej_01": float((pvals_j < 0.01).mean()),
            })
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUT_DIR, "v_sweep_summary.csv")
    sum_df.to_csv(sum_csv, index=False, float_format="%.5f")

    # 5. Persist raw matrices.
    pkl_path = os.path.join(OUT_DIR, "v_sweep_pvals.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "pvals_by_V": pvals_by_V,
            "config": {
                "n_total": n_total, "p": p, "maxlen": maxlen,
                "true_pattern_idx": true_pattern_idx, "predindex": predindex,
                "nheads": nheads, "ncum": ncum, "EPOCHS": EPOCHS,
                "nrepp": nrepp, "V_LIST": V_LIST, "inject_seed": INJECT_SEED,
                "seed_train": seed_train, "seed_test": seed_test,
            },
        }, f)
    print(f"\nSaved raw V-sweep matrices to {pkl_path}")

    # 6. Plots: mean p-value and rejection rate vs V (triggers red, nulls blue).
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    color_signal = "#AE232F"
    color_null = "#1D4A91"
    metrics = [
        ("mean_p", "Mean p-value", axes[0]),
        ("rej_05", r"Empirical rejection rate at $\alpha=0.05$", axes[1]),
    ]
    for col, label, ax in metrics:
        for j in range(p):
            if j == predindex:
                continue
            sub = sum_df[sum_df["var"] == j].sort_values("V")
            color = color_signal if j in true_pattern_idx else color_null
            ax.plot(sub["V"], sub[col], marker="o", color=color, alpha=0.85,
                    label=f"{VAR_NAMES[j]}"
                          + (" (trigger)" if j in true_pattern_idx else ""))
        ax.set_xlabel("V (visit-sample size)")
        ax.set_ylabel(label)
        ax.set_xticks(V_LIST)
        ax.grid(True, alpha=0.3)
    axes[1].axhline(0.05, color="grey", linestyle="--", linewidth=1,
                    label=r"$\alpha=0.05$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"V-sweep on PBC2-substrate simulation (predindex={predindex}, "
        f"nrepp={nrepp})  red = triggers (bili_high, albumin_low)   "
        f"blue = null-by-construction",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    plot_path = os.path.join(OUT_DIR, "v_sweep_plots.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V-sweep plots to {plot_path}")

    # 7. Human-readable summary.
    lines = []
    lines.append("=== V-monotonicity check on the PBC2-substrate simulation ===")
    lines.append(f"Cohort: real binarised PBC2, {n_total} patients, "
                 f"synthetic j1->j2->j3 target on ascites.")
    lines.append(f"Triggers (generative signal): bili_high (col {COL_BILI_HIGH}), "
                 f"albumin_low (col {COL_ALBUMIN_LOW}).")
    lines.append(f"P(y=1) = {flat_y.mean():.3f}")
    lines.append(f"V-sweep at predindex={predindex}, nrepp={nrepp}, EPOCHS={EPOCHS}")
    lines.append("")
    pretty = sum_df[["V", "var", "name", "is_trigger", "mean_p", "std_p",
                     "rej_05", "rej_01"]].copy()
    lines.append(pretty.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
