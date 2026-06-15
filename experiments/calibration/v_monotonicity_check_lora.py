"""V monotonicity check on LORA cohort data.

Mirrors notebooks/v_monotonicity_check.py (simulation version) but applied to one
fold of the LORA dataset. Trains the MiniTransformer with the same
hyperparameters used in the paper for real data (H=8 heads, C=8 cumulants,
batch size 2, 150 epochs, learning rate 1e-3, L2 1e-3) on fold 0 of a
10-fold KFold split with random_state=42 (matching the existing real-data
notebooks). Then:

(A) Computes per-target evaluation-fold MSE for the model and three baselines
    (marginal-mean "average", per-target Gaussian regression, and "carry
    forward" / repeat-last-value); flags which targets pass the predictability
    gate from Appendix S1 of the manuscript.

(B) On the same fitted model, runs the permutation test for V in {5, 6, 7}
    with nrepp=500 each, predicting the GHQ target (predindex=9). Reports
    per-variable mean p-value, std, and rejection rates at
    alpha in {0.05, 0.01}. Directly addresses the reviewer's §3.6(a) concern:
    "what happens on the LORA datasets as V is increased toward its maximum?".

Outputs (under notebooks/results/v_monotonicity_check_lora/<data_str>/):
    per_target_mse.csv       per-target MSE table + gate column
    v_sweep_summary.csv      per-variable p-value summary across V
    v_sweep_pvals.pkl        raw (V -> (p, nrepp)) p-value matrices
    v_sweep_plots.png        mean p-value, rejection rate vs V
    summary.txt              human-readable text dump
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
import statsmodels.api as sm
from sklearn.model_selection import KFold

from src.data_preparation import collate_function, load_real_data
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
    count_parameters,
)
from src.evaluation import calculate_regression_loss
from src.statistical_testing import statistical_testing


# ----------------------------- configuration -------------------------------- #
device = torch.device("cpu")

# Which LORA dataset (matches existing real-data notebooks)
DATA_STR = os.environ.get("VSGL_DATA", "ghq_b_sum")  # or "ghq_sum"

# Architecture (matches paper §3.2: H=8, C=8 for real data)
nheads = int(os.environ.get("VSGL_NHEADS", 8))
ncum = int(os.environ.get("VSGL_NCUM", 8))
dk = 1
dv = 1

# Training
batch_size = int(os.environ.get("VSGL_BATCH", 2))
learning_rate = 1e-3
lambda_l2 = 1e-3
EPOCHS = int(os.environ.get("VSGL_EPOCHS", 150))

# CV split (matches existing real-data notebooks: KFold n=10, random_state=42)
N_SPLITS = 10
CV_RANDOM_STATE = 42
FOLD_INDEX = int(os.environ.get("VSGL_FOLD", 0))  # fold 0 of 10

# Test
predindex = int(os.environ.get("VSGL_PREDINDEX", 9))  # GHQ is the last column
V_LIST = [int(v) for v in os.environ.get("VSGL_V_LIST", "5,6,7").split(",")]
nrepp = int(os.environ.get("VSGL_NREPP", 500))
seed_test = 20260505

OUT_DIR = f"notebooks/results/v_monotonicity_check_lora/{DATA_STR}"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------- per-target MSE ------------------------------- #
def per_target_mse_average(train_data, test_data):
    """Predict each target by the training mean. Returns (p,)."""
    train_concat = torch.cat(train_data, dim=0)
    means = train_concat.mean(dim=0).numpy()
    test_last = torch.stack([seq[-1] for seq in test_data]).numpy()
    return ((test_last - means[None, :]) ** 2).mean(axis=0)


def per_target_mse_repeat(test_data):
    """Carry-forward baseline: predict last timestep = previous timestep."""
    p_local = test_data[0].shape[1]
    sq_err_sum = np.zeros(p_local)
    n_eval = 0
    for seq in test_data:
        if seq.shape[0] < 2:
            continue
        sq_err_sum += (seq[-1].numpy() - seq[-2].numpy()) ** 2
        n_eval += 1
    return sq_err_sum / max(n_eval, 1)


def per_target_mse_regression(train_data, test_data):
    """Per-target Gaussian GLM using all features at t-1. Returns (p,) MSEs.

    Thin wrapper around `calculate_regression_loss` from src.evaluation.py -- the
    same regression baseline used to produce the paper's Tables 1 and 3. We
    request the full per-target MSE vector via `return_per_target=True`.
    """
    _, _, per_target = calculate_regression_loss(
        train_data, test_data, predindex=0, return_per_target=True,
    )
    return per_target


def per_target_mse_minitransformer(model, test_data):
    """MiniTransformer per-target MSE on the last-timestep prediction."""
    model.eval()
    p_local = test_data[0].shape[1]
    sq_err_sum = np.zeros(p_local)
    n_eval = 0
    with torch.no_grad():
        for seq in test_data:
            if seq.shape[0] < 3:
                continue
            x = seq.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            sq_err_sum += (pred[0, -1].numpy() - seq[-1].numpy()) ** 2
            n_eval += 1
    return sq_err_sum / max(n_eval, 1)


# ----------------------------- main ----------------------------------------- #
def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    print(f"=== V-sweep + gate analysis on LORA ({DATA_STR}) ===")
    print(f"Architecture: H={nheads}, C={ncum}, batch={batch_size}, "
          f"epochs={EPOCHS}, predindex={predindex}")
    print(f"CV: KFold(n_splits={N_SPLITS}, random_state={CV_RANDOM_STATE}), "
          f"using fold {FOLD_INDEX}")
    print(f"V_LIST={V_LIST}, nrepp={nrepp}\n")

    # 1. Load LORA data
    data, maxlen_data = load_real_data(DATA_STR)
    p = data[0].shape[1]
    print(f"Loaded {len(data)} sequences from LORA/{DATA_STR}, "
          f"p={p}, max len={maxlen_data}")
    if predindex >= p:
        raise ValueError(f"predindex={predindex} but p={p}")

    # 2. CV split, take the requested fold (matches existing real-data notebooks)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    folds = list(kf.split(data))
    train_idx, test_idx = folds[FOLD_INDEX]
    train_data = [data[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]
    print(f"Fold {FOLD_INDEX}: {len(train_data)} train + {len(test_data)} test\n")

    # 3. Build + train model. The maxlen for masks is the cap used at training
    #    time (10 in the paper), to stay within the precomputed mask shape.
    maxlen = 10
    torch.manual_seed(CV_RANDOM_STATE)
    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)
    model = MiniTransformer(
        p, nheads, dk, dv, ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix,
        device,
    ).to(device)
    print(f"Model parameter count: {count_parameters(model)}")

    # Filter sequences to fit within maxlen (the model's positional encodings
    # are precomputed up to maxlen=10; longer sequences are truncated to the
    # last 10 time points, mirroring the convention used elsewhere).
    def truncate(seq):
        return seq if seq.shape[0] <= maxlen else seq[-maxlen:]
    train_data = [truncate(s) for s in train_data]
    test_data = [truncate(s) for s in test_data]

    dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    eval_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    t0 = time.time()
    train_mini_transformer(model, dataloader, eval_loader, optimizer, lambda_l2, EPOCHS, device)
    print(f"\nTraining done in {time.time() - t0:.1f}s.\n")

    # 4. Per-target MSE on test fold + baselines
    print("Computing per-target test-fold MSE ...")
    mse_mt = per_target_mse_minitransformer(model, test_data)
    mse_avg = per_target_mse_average(train_data, test_data)
    mse_reg = per_target_mse_regression(train_data, test_data)
    mse_rep = per_target_mse_repeat(test_data)

    # Variable names from the paper (ordered as columns in the LORA CSV)
    if DATA_STR == "ghq_b_sum":
        var_names = ["dh_10 (Nightmares)", "dh_35 (Sleep problems)",
                     "dh_37 (Paperwork)", "dh_38 (Housekeeping)",
                     "dh_45 (Noise)", "dh_53 (Long work hours)",
                     "le_8 (Financial problems)", "le_17 (Arguments w/ partner)",
                     "le_22 (Serious illness)", "ghq_b_sum (Anxiety/Sleep)"]
    elif DATA_STR == "ghq_sum":
        var_names = ["dh_11 (Commute)", "dh_31 (Unwanted visit)",
                     "dh_37 (Paperwork)", "dh_38 (Housekeeping)",
                     "dh_42 (Bad weather)", "dh_46 (Traffic)",
                     "le_1 (Lost job)", "le_16 (Breakup)",
                     "le_17 (Arguments w/ partner)", "ghq_sum (Distress)"]
    else:
        var_names = [f"var{i}" for i in range(p)]

    df = pd.DataFrame({
        "target_var": np.arange(p),
        "name": var_names,
        "MSE_MT": mse_mt,
        "MSE_avg": mse_avg,
        "MSE_reg": mse_reg,
        "MSE_repeat": mse_rep,
    })
    df["beats_avg"] = df["MSE_MT"] < df["MSE_avg"]
    df["beats_reg"] = df["MSE_MT"] < df["MSE_reg"]
    df["beats_repeat"] = df["MSE_MT"] < df["MSE_repeat"]
    # Gate uses only the marginal-mean comparison; the regression and repeat
    # columns are reported alongside for transparency but are not part of the
    # gate criterion (cf. Appendix S1 of the manuscript).
    df["gate_passes"] = df["beats_avg"]

    csv_path = os.path.join(OUT_DIR, "per_target_mse.csv")
    df.to_csv(csv_path, index=False, float_format="%.5f")
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n")
    print(f"Saved per-target MSE to {csv_path}")

    gated_targets = df.loc[df["gate_passes"], "target_var"].tolist()
    print(f"Gate-passing targets: {gated_targets}")
    print(f"  (paper's chosen target for {DATA_STR} is index {predindex} = "
          f"{var_names[predindex]})\n")

    # 5. V-sweep on the same trained model
    pvals_by_V = {}
    for V in V_LIST:
        print(f"--- V = {V} (10^{V} = {10**V} permutations / rep, nrepp={nrepp}) ---")
        t0 = time.time()
        avepval, stdpval, ctx, tgts, pval_mat = statistical_testing(
            model, train_data, p, predindex, nrepp, V,
            return_pval_mat=True, seed=seed_test,
        )
        print(f"V={V} done in {time.time() - t0:.1f}s")
        pvals_by_V[V] = pval_mat.detach().cpu().numpy()

    # 6. Aggregate
    summary_rows = []
    for V, mat in pvals_by_V.items():
        for j in range(p):
            pvals_j = mat[j]
            summary_rows.append({
                "V": V, "var": j, "name": var_names[j],
                "mean_p": float(pvals_j.mean()),
                "std_p": float(pvals_j.std()),
                "rej_05": float((pvals_j < 0.05).mean()),
                "rej_01": float((pvals_j < 0.01).mean()),
            })
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUT_DIR, "v_sweep_summary.csv")
    sum_df.to_csv(sum_csv, index=False, float_format="%.5f")

    pkl_path = os.path.join(OUT_DIR, "v_sweep_pvals.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "pvals_by_V": pvals_by_V,
            "config": {
                "DATA_STR": DATA_STR, "p": p, "maxlen": maxlen,
                "predindex": predindex, "nheads": nheads, "ncum": ncum,
                "EPOCHS": EPOCHS, "batch_size": batch_size,
                "FOLD_INDEX": FOLD_INDEX, "N_SPLITS": N_SPLITS,
                "CV_RANDOM_STATE": CV_RANDOM_STATE,
                "nrepp": nrepp, "V_LIST": V_LIST, "seed_test": seed_test,
            },
            "per_target_mse": df,
            "var_names": var_names,
        }, f)
    print(f"\nSaved raw V-sweep matrices to {pkl_path}")

    # 7. Plot: 2 panels (mean p, rej rate at 0.05) x V
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    target_color = "#AE232F"  # the chosen target gets red, all others blue
    other_color = "#1D4A91"
    metrics = [
        ("mean_p", "Mean p-value", axes[0]),
        ("rej_05", r"Empirical rejection rate at $\alpha=0.05$", axes[1]),
    ]
    for col, label, ax in metrics:
        for j in range(p):
            sub = sum_df[sum_df["var"] == j].sort_values("V")
            color = target_color if j == predindex else other_color
            short_name = var_names[j].split(" ")[0]
            ax.plot(sub["V"], sub[col], marker="o", color=color, alpha=0.85,
                    label=short_name + (" (target)" if j == predindex else ""))
        ax.set_xlabel("V (visit-sample size)")
        ax.set_ylabel(label)
        ax.set_xticks(V_LIST)
        ax.grid(True, alpha=0.3)
    axes[1].axhline(0.05, color="grey", linestyle="--", linewidth=1,
                    label=r"$\alpha=0.05$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"V-sweep on LORA / {DATA_STR}  (target={var_names[predindex].split(' ')[0]}, "
        f"nrepp={nrepp})  red = target itself  blue = candidate contexts",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plot_path = os.path.join(OUT_DIR, "v_sweep_plots.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved V-sweep plots to {plot_path}")

    # 8. Text summary
    lines = []
    lines.append(f"=== V-sweep + gate on LORA / {DATA_STR} ===")
    lines.append("")
    lines.append("(A) Predictability gate on test fold")
    lines.append(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append(f"  Gate-passing targets: {gated_targets}")
    lines.append(f"  Paper's target for {DATA_STR}: predindex={predindex} "
                 f"({var_names[predindex]})")
    lines.append("")
    lines.append("(B) V-sweep at predindex=" + str(predindex)
                 + ", nrepp=" + str(nrepp))
    pretty = sum_df[["V", "var", "name", "mean_p", "std_p",
                     "rej_05", "rej_01"]].copy()
    lines.append(pretty.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
