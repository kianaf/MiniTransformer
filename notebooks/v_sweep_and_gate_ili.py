"""V-sweep + predictability-gate analysis on the ILI benchmark.

Companion to notebooks/v_sweep_and_gate_pbc2.py. Trains MiniTransformer on
the binarised ILI cohort (Influenza-Like Illness, prepared via
notebooks/prepare_ili.py: 192 length-10 windows of the CDC weekly counts
2002-2021, 7 variables median-binarised, OT as the last column = target).
Reports the predictability gate via 10-fold CV and the V-sweep on a single
model trained on the full cohort, paper-style:

(A) per-target evaluation-fold MSE for MiniTransformer + the marginal mean,
    per-target Gaussian regression and carry-forward baselines, mean +/- std
    across the 10 folds (matching real_data_experiments_*.ipynb);
(B) on a single model trained on the full ILI cohort, runs the permutation
    test of Section 2.3 for V in {5, 6, 7} with nrepp=500, predicting OT
    (predindex = p - 1 = 6).

Outputs land under notebooks/results/v_sweep_and_gate_ili/.
"""
import os
import sys
import time
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
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
from scipy import stats
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
from src.statistical_testing import statistical_testing


device = torch.device("cpu")

# Architecture (LORA-style: H=8, C=8); ILI has p=7 instead of 10
nheads = int(os.environ.get("VSGI_NHEADS", 8))
ncum   = int(os.environ.get("VSGI_NCUM", 8))
dk     = 1
dv     = 1
batch_size    = int(os.environ.get("VSGI_BATCH", 2))
learning_rate = 1e-3
lambda_l2     = 1e-3
EPOCHS        = int(os.environ.get("VSGI_EPOCHS", 150))

N_SPLITS        = 10
CV_RANDOM_STATE = 42

V_LIST = [int(v) for v in os.environ.get("VSGI_V_LIST", "5,6,7").split(",")]
nrepp  = int(os.environ.get("VSGI_NREPP", 500))
seed_test = 20260505

OUT_DIR = "notebooks/results/v_sweep_and_gate_ili"
os.makedirs(OUT_DIR, exist_ok=True)


def per_target_mse_average(train_data, test_data):
    train_concat = torch.cat(train_data, dim=0)
    means = train_concat.mean(dim=0).numpy()
    test_last = torch.stack([seq[-1] for seq in test_data]).numpy()
    return ((test_last - means[None, :]) ** 2).mean(axis=0)


def per_target_mse_repeat(test_data):
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
    p_local = train_data[0].shape[1]
    out = np.zeros(p_local)
    X_train, Y_train = [], []
    for seq in train_data:
        T = seq.shape[0]
        for t in range(2, T - 1):
            X_train.append(seq[t - 1].numpy())
            Y_train.append(seq[t].numpy())
    X_train = sm.add_constant(np.stack(X_train), has_constant="add")
    Y_train = np.stack(Y_train)
    X_test, Y_test = [], []
    for seq in test_data:
        if seq.shape[0] < 3:
            continue
        X_test.append(seq[-2].numpy())
        Y_test.append(seq[-1].numpy())
    X_test = sm.add_constant(np.stack(X_test), has_constant="add")
    Y_test = np.stack(Y_test)
    for r in range(p_local):
        m = sm.GLM(Y_train[:, r], X_train, family=sm.families.Gaussian()).fit()
        out[r] = float(np.mean((m.predict(X_test) - Y_test[:, r]) ** 2))
    return out


def per_target_mse_minitransformer(model, test_data):
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


def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    print("=== V-sweep + gate analysis on ILI ===")
    print(f"H={nheads}, C={ncum}, batch={batch_size}, epochs={EPOCHS}")
    print(f"CV: KFold(n_splits={N_SPLITS}, random_state={CV_RANDOM_STATE})")
    print(f"V_LIST={V_LIST}, nrepp={nrepp}\n")

    data, _ = load_real_data("ili")
    p = data[0].shape[1]
    target_idx = p - 1
    print(f"Loaded {len(data)} windows, p={p}, target column index={target_idx}\n")

    var_names = [
        "weighted_ILI", "unweighted_ILI",
        "AGE_0_4", "AGE_5_24",
        "ILITOTAL", "NUM_PROVIDERS",
        "OT (target)",
    ]

    # Masks (precomputed positional matrices use maxlen=10)
    maxlen = 10
    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

    # ---- (A) 10-fold gate on per-target MSE ---------------------------------
    print("(A) Predictability gate via 10-fold CV ...")
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                       random_state=CV_RANDOM_STATE).split(data))
    SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]

    fold_mt, fold_avg, fold_reg, fold_rep = [], [], [], []
    n_params_mt = None
    for f_idx, ((tr_idx, te_idx), seed) in enumerate(zip(folds, SEEDS)):
        train_data = [data[i] for i in tr_idx]
        test_data  = [data[i] for i in te_idx]
        torch.manual_seed(seed); np.random.seed(seed)
        model = MiniTransformer(p, nheads, dk, dv, ncum, mask_pairwise,
                                pairwise_distance_matrix, distance_to_end_matrix,
                                device).to(device)
        if n_params_mt is None:
            n_params_mt = count_parameters(model)
            print(f"  MiniTransformer params: {n_params_mt}")
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_function, num_workers=0)
        opt = optim.Adam(model.parameters(), lr=learning_rate)
        train_mini_transformer(model, loader, None, opt, lambda_l2, EPOCHS, device)
        fold_mt.append(per_target_mse_minitransformer(model, test_data))
        fold_avg.append(per_target_mse_average(train_data, test_data))
        fold_reg.append(per_target_mse_regression(train_data, test_data))
        fold_rep.append(per_target_mse_repeat(test_data))
        print(f"  fold {f_idx+1}/{N_SPLITS}, seed={seed}: "
              f"MSE_target_MT={fold_mt[-1][target_idx]:.4f}")

    mse_mt  = np.stack(fold_mt)   # (n_folds, p)
    mse_avg = np.stack(fold_avg)
    mse_reg = np.stack(fold_reg)
    mse_rep = np.stack(fold_rep)

    gate = pd.DataFrame({
        "name": var_names,
        "MSE_MT_mean":     mse_mt.mean(axis=0),  "MSE_MT_std":     mse_mt.std(axis=0),
        "MSE_avg_mean":    mse_avg.mean(axis=0),
        "MSE_reg_mean":    mse_reg.mean(axis=0),
        "MSE_repeat_mean": mse_rep.mean(axis=0),
    })
    gate["beats_avg_mean"] = gate["MSE_MT_mean"] < gate["MSE_avg_mean"]
    gate["beats_reg_mean"] = gate["MSE_MT_mean"] < gate["MSE_reg_mean"]
    gate["gate_mean"] = gate["beats_avg_mean"] & gate["beats_reg_mean"]
    gate["folds_pass_gate"] = ((mse_mt < mse_avg) & (mse_mt < mse_reg)).sum(axis=0)
    gate.to_csv(os.path.join(OUT_DIR, "per_target_mse.csv"),
                index=False, float_format="%.5f")
    print("\nGate (CV mean):")
    print(gate.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  Gate passes for: {gate.loc[gate['gate_mean'], 'name'].tolist()}")

    # ---- (B) Full-data permutation test ------------------------------------
    print("\n(B) Permutation test on a single full-data model ...")
    torch.manual_seed(42); np.random.seed(42)
    model = MiniTransformer(p, nheads, dk, dv, ncum, mask_pairwise,
                            pairwise_distance_matrix, distance_to_end_matrix,
                            device).to(device)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_function, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    train_mini_transformer(model, loader, None, opt, lambda_l2, EPOCHS, device)
    print("  full-data model trained.")

    pvals_by_V = {}
    for V in V_LIST:
        print(f"  V={V} (10^{V} permutations / rep, nrepp={nrepp}) ...")
        t0 = time.time()
        _, _, _, _, pmat = statistical_testing(
            model, data, p, target_idx, nrepp, V,
            return_pval_mat=True, seed=seed_test,
        )
        print(f"    done in {time.time() - t0:.1f}s")
        pvals_by_V[V] = pmat.detach().cpu().numpy()

    # Aggregate
    summary_rows = []
    for V, mat in pvals_by_V.items():
        for j in range(p):
            ks_stat, ks_pval = stats.kstest(mat[j], "uniform")
            summary_rows.append({
                "V": V, "var": j, "name": var_names[j],
                "mean_p":  float(mat[j].mean()),
                "std_p":   float(mat[j].std()),
                "ks_stat": float(ks_stat),
                "ks_pval": float(ks_pval),
                "rej_05":  float((mat[j] < 0.05).mean()),
                "rej_01":  float((mat[j] < 0.01).mean()),
            })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, "v_sweep_summary.csv"),
                  index=False, float_format="%.5f")

    with open(os.path.join(OUT_DIR, "v_sweep_pvals.pkl"), "wb") as f:
        pickle.dump({
            "pvals_by_V": pvals_by_V,
            "var_names":  var_names,
            "config": {
                "data": "ili", "p": p, "predindex": target_idx,
                "nheads": nheads, "ncum": ncum, "EPOCHS": EPOCHS,
                "N_SPLITS": N_SPLITS, "CV_RANDOM_STATE": CV_RANDOM_STATE,
                "V_LIST": V_LIST, "nrepp": nrepp,
                "n_params_mt": n_params_mt,
            },
            "gate":      gate,
        }, f)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    target_color = "#AE232F"
    other_color  = "#1D4A91"
    metrics = [
        ("mean_p",  "Mean p-value", axes[0]),
        ("rej_05",  r"Empirical rejection rate at $\alpha=0.05$", axes[1]),
        ("ks_stat", "KS distance vs Uniform[0,1]", axes[2]),
    ]
    for col, label, ax in metrics:
        for j in range(p):
            sub = sum_df[sum_df["var"] == j].sort_values("V")
            color = target_color if j == target_idx else other_color
            short_name = var_names[j].split(" ")[0]
            ax.plot(sub["V"], sub[col], marker="o", color=color, alpha=0.85,
                    label=short_name + (" (target)" if j == target_idx else ""))
        ax.set_xlabel("V (visit-sample size)")
        ax.set_ylabel(label)
        ax.set_xticks(V_LIST)
        ax.grid(True, alpha=0.3)
    axes[1].axhline(0.05, color="grey", linestyle="--", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"V-sweep on ILI (target=OT, nrepp={nrepp})  "
                 f"red = target itself  blue = candidate contexts", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plot_path = os.path.join(OUT_DIR, "v_sweep_plots.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Text summary
    lines = []
    lines.append("=== V-sweep + gate on ILI ===")
    lines.append(f"H={nheads}, C={ncum}, EPOCHS={EPOCHS}")
    lines.append("")
    lines.append("(A) Predictability gate (10-fold CV)")
    lines.append(gate.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append(f"(B) V-sweep at predindex={target_idx} (OT), nrepp={nrepp}")
    lines.append(sum_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
