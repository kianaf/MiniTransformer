"""Null-calibration check for the MiniTransformer permutation test.

Trains the MiniTransformer on the standard simulation (p=10, n=200, only
variables 0,1,2 carry the true context pattern) with the signal target
predindex=2 and then runs ``statistical_testing`` with a large number of
repetitions to obtain a per-variable distribution of p-values. Variables
3..9 are null with respect to this target. The empirical permutation null
is contaminated by signal rows (variables 0,1,2 with non-zero Delta), so
the paper (Section 2.3) predicts that p-values for null variables are
shifted toward 1 -- i.e.\ conservative rather than uniform. We empirically
verify Type-I rate <= alpha.

The script only supports the signal-target setting; the test is interpretable
only on targets the model can predict well (the Section 2.3 guideline), so
calibration on a generatively-null target is not part of the calibration
analysis reported in Appendix S1.

Outputs (under notebooks/results/null_calibration/signal_target_predindex=2):
    - histograms.png         per-variable p-value histogram + uniform reference
    - qq_null_variables.png  Q-Q vs Uniform[0,1] for all variables
    - summary.txt            mean p, rejection rates @ 0.05 / 0.01
    - pval_mat.pkl           raw (p, nrepp) p-value matrix + run config
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
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data_preparation import SimulatedDataset, collate_function
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
    count_parameters,
)
from src.statistical_testing import statistical_testing


# ----------------------------- configuration -------------------------------- #
device = torch.device("cpu")

# Simulation / model hyperparameters (match the paper's simulation setup)
n_train = int(os.environ.get("CALIB_N", 200))
p = 10
maxlen = 10
true_pattern_idx = (0, 1, 2)  # variables involved in the true context pattern
# We calibrate only on the signal target (predindex = j_3 = 2). The
# generatively-null-target mode used in earlier drafts has been removed:
# the Section 2.3 guideline ties test interpretation to targets the model
# can predict, so calibration on a target the model cannot predict is not
# part of the analysis reported in Appendix S1.
predindex = 2

# Architecture
nheads = 12
ncum = 2
dk = 1
dv = 1

# Training
batch_size = 1
learning_rate = 1e-3
lambda_l2 = 1e-3
EPOCHS = int(os.environ.get("CALIB_EPOCHS", 100))
seed_train = 42

# Calibration test
# Note: the inner permutation enumerates ncont^V = p^V combinations per repetition.
# With p=10 and V=8 (paper setup) this is 10^8 = 100M, prohibitive at nrepp=500.
# We default to V=5 for the calibration (10^5 = 100K per rep), which does not affect
# the null distribution shape — only test power on signal variables.
nrepp = int(os.environ.get("CALIB_NREPP", 500))
target_sample_size = int(os.environ.get("CALIB_V", 5))
seed_test = 20260505

_mode = "signal_target"  # only signal-target mode is supported
OUT_DIR = f"notebooks/results/null_calibration/{_mode}_predindex={predindex}"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    torch.manual_seed(seed_train)

    print(f"=== MiniTransformer null-calibration (nrepp={nrepp}) ===\n")

    # 1. Generate simulation data, in the SAME RNG-consumption order as
    #    simulation_experiments.ipynb (which produced Table 1) and as
    #    v_monotonicity_check.py: seed once at top, then train, then test
    #    (n_test=1000 to match the paper), then model. The test set is
    #    consumed by the RNG even though this script does not use it for
    #    evaluation -- this keeps the trained model identical to the one
    #    used in v_monotonicity_check.py for cross-script comparability.
    train_dataset = SimulatedDataset(n_train, p, maxlen=maxlen, device=device).data
    _ = SimulatedDataset(1000, p, maxlen=maxlen, device=device).data  # consume RNG
    print(f"Generated {len(train_dataset)} train sequences with p={p}.\n")

    # 2. Build model
    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

    model = MiniTransformer(
        p, nheads, dk, dv, ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix,
        device,
    ).to(device)
    print(f"Model parameter count: {count_parameters(model)}\n")

    # 3. Train
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    t0 = time.time()
    train_mini_transformer(model, dataloader, None, optimizer, lambda_l2, EPOCHS, device)
    print(f"\nTraining done in {time.time() - t0:.1f}s.\n")

    # 4. Run calibration: large nrepp, return full p-value matrix
    print(f"Running statistical_testing with nrepp={nrepp}, V={target_sample_size}...")
    t0 = time.time()
    avepval, stdpval, context, targetall, pval_mat = statistical_testing(
        model, train_dataset, p, predindex, nrepp, target_sample_size,
        return_pval_mat=True, seed=seed_test,
    )
    print(f"Statistical testing done in {time.time() - t0:.1f}s.\n")

    pval_mat_np = pval_mat.detach().cpu().numpy()  # shape (p, nrepp)
    assert pval_mat_np.shape == (p, nrepp), (
        f"pval_mat shape {pval_mat_np.shape} != expected ({p}, {nrepp})"
    )

    # 5. Persist raw p-value matrix
    raw_path = os.path.join(OUT_DIR, "pval_mat.pkl")
    with open(raw_path, "wb") as f:
        pickle.dump({
            "pval_mat": pval_mat_np,
            "config": {
                "n_train": n_train, "p": p, "maxlen": maxlen,
                "true_pattern_idx": true_pattern_idx, "predindex": predindex,
                "nheads": nheads, "ncum": ncum, "EPOCHS": EPOCHS,
                "nrepp": nrepp, "target_sample_size": target_sample_size,
                "seed_train": seed_train, "seed_test": seed_test,
            },
        }, f)
    print(f"Saved raw p-value matrix to {raw_path}\n")

    # 6. Per-variable summary + rejection rates. With the signal target
    # (predindex=2), "null" means "not in true_pattern_idx", i.e. variables
    # 3..9. The permutation null is contaminated by signal rows from
    # variables 0,1,2, so p-values for null variables are shifted toward 1
    # (conservative behaviour), as predicted in Section 2.3.
    null_set = set(range(p)) - set(true_pattern_idx)

    summary_lines = []
    summary_lines.append(
        f"Mode: {_mode}   predindex={predindex}   nrepp={nrepp}   V={target_sample_size}"
    )
    summary_lines.append("")
    header = (
        f"Var | mean p |  std  | rej@0.05 | rej@0.01 | true"
    )
    summary_lines.append(header)
    summary_lines.append("-" * len(header))
    rej05_null, rej05_signal = [], []
    rej01_null, rej01_signal = [], []
    for j in range(p):
        pvals_j = pval_mat_np[j]
        rej05 = float((pvals_j < 0.05).mean())
        rej01 = float((pvals_j < 0.01).mean())
        cls = "NULL  " if j in null_set else "SIGNAL"
        summary_lines.append(
            f"{j:>2}  | {pvals_j.mean():.3f}  | {pvals_j.std():.3f} "
            f"|  {rej05:.3f}   |  {rej01:.3f}   | {cls}"
        )
        if j in null_set:
            rej05_null.append(rej05)
            rej01_null.append(rej01)
        else:
            rej05_signal.append(rej05)
            rej01_signal.append(rej01)

    summary_lines.append("")
    summary_lines.append("Empirical rejection rates (averaged across variables):")
    if rej05_null:
        summary_lines.append(
            f"  null variables    : {np.mean(rej05_null):.3f} @ alpha=0.05 "
            f"|   {np.mean(rej01_null):.3f} @ alpha=0.01     "
            f"(target: <= alpha; lower = conservative)"
        )
    if rej05_signal:
        summary_lines.append(
            f"  signal variables  : {np.mean(rej05_signal):.3f} @ alpha=0.05 "
            f"|   {np.mean(rej01_signal):.3f} @ alpha=0.01     "
            f"(target: high = good power)"
        )

    summary_txt = "\n".join(summary_lines)
    print(summary_txt)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    # 7. Per-variable histograms
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True, sharey=True)
    for j, ax in enumerate(axes.flat):
        pvals_j = pval_mat_np[j]
        is_null = j in null_set
        color = "#1D4A91" if is_null else "#AE232F"
        ax.hist(pvals_j, bins=20, range=(0, 1), color=color, alpha=0.85,
                edgecolor="white")
        ax.axhline(nrepp / 20, color="grey", linestyle="--", linewidth=1,
                   label="Uniform expectation")
        title = f"Var {j} ({'null' if is_null else 'signal'})"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("p-value")
        if j % 5 == 0:
            ax.set_ylabel("Frequency")
    fig.suptitle(
        f"Per-variable p-value distributions  ({_mode}, predindex={predindex}, "
        f"n_train={n_train}, p={p}, nrepp={nrepp}, V={target_sample_size})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    hist_path = os.path.join(OUT_DIR, "histograms.png")
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved histograms to {hist_path}")

    # 8. Q-Q plots vs Uniform[0,1] for ALL variables, color-coded by signal/null.
    # Layout matches the histogram figure (2 x 5) so the two figures can be read
    # side by side.
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    for ax, j in zip(axes, range(p)):
        pvals_j = np.sort(pval_mat_np[j])
        theoretical = np.linspace(0, 1, len(pvals_j) + 2)[1:-1]
        is_null = j in null_set
        color = "#1D4A91" if is_null else "#AE232F"
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.plot(theoretical, pvals_j, marker="o", linestyle="none",
                markersize=2, color=color)
        title = f"Var {j} ({'null' if is_null else 'signal'})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Uniform[0,1] quantiles")
        if j % 5 == 0:
            ax.set_ylabel("Empirical p-value quantiles")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(
        f"Q-Q vs Uniform[0,1]  ({_mode}, predindex={predindex}, nrepp={nrepp}, "
        f"V={target_sample_size})  red = signal, blue = null",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Keep the legacy filename for backward compatibility with the manuscript.
    qq_path = os.path.join(OUT_DIR, "qq_null_variables.png")
    fig.savefig(qq_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Q-Q plot to {qq_path}\n")

    print("=== Done. ===")


if __name__ == "__main__":
    main()
