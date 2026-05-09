"""Per-target gate + per-target permutation test on PBC2.

Mirrors the LORA-style two-track convention exactly:
  - Gate: 10-fold CV with per-fold seeds (matches real_data_experiments_pbc2.ipynb).
  - Permutation test: single full-data model with a fixed seed (matches
    real_data_experiments_pbc2_statistical_testing.ipynb).

Difference from the existing pair: instead of fixing predindex = 9 and asking
about that one target, this script treats *every* variable as a candidate
target. For each candidate r:
  1. The gate is evaluated using BOTH baselines: MT must beat the marginal-
     mean baseline AND the per-target regression (the canonical
     calculate_regression_loss).
  2. If r passes the strict gate, a permutation test is run on the full-data
     model with predindex=r and the per-context p-values are reported.

Outputs (under notebooks/results/per_target_analysis_pbc2/):
  per_target_gate.csv          per-target MSEs and gate decisions
  pvalues_<target>.csv         per-context p-values for each gate-passing target
  summary.txt                  human-readable text dump

Run with:  python notebooks/per_target_analysis_pbc2.py
"""
import os
import sys
import time

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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import warnings

from src.data_preparation import collate_function, load_real_data
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
    count_parameters,
)
from src.evaluation import calculate_bench1_loss, calculate_regression_loss
from src.statistical_testing import statistical_testing


# ----------------------------- configuration -------------------------------- #
device = torch.device("cpu")

# Hyperparameters (match real_data_experiments_pbc2.ipynb / LORA convention)
data_str           = "pbc2"
batch_size         = 2
dk                 = 1
dv                 = 1
nheads             = 8
ncum               = 8
maxlen             = 10
learning_rate      = 1e-3
lambda_l2          = 1e-3
EPOCHS             = 150
target_sample_size = 8
nrepp              = 10
seeds              = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]
k_for_cross_val    = 10
seed_full_data     = 12345  # matches the LORA stat-testing convention

OUT_DIR = "notebooks/results/per_target_analysis_pbc2"
os.makedirs(OUT_DIR, exist_ok=True)

VAR_NAMES = [
    "hepatomegaly", "spiders", "edema_present",
    "albumin_low", "alkphos_high", "ast_high", "platelet_low", "protime_high",
    "bili_high",
    "ascites",   # target -> predindex=9 (rare event, ~8.7% prevalence)
]


# --------------------------- per-target MSE helpers ------------------------- #
def per_target_mse_mt(model, eval_data):
    """MiniTransformer per-target test MSE on the last-timestep prediction."""
    model.eval()
    p_local = eval_data[0].shape[1]
    se, n_eval = np.zeros(p_local), 0
    with torch.no_grad():
        for s in eval_data:
            if s.shape[0] < 3:
                continue
            x = s.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            se += (pred[0, -1].numpy() - s[-1].numpy()) ** 2
            n_eval += 1
    return se / max(n_eval, 1)


def per_target_mse_avg(train_data, eval_data):
    """Marginal-mean baseline per target."""
    means = torch.cat(train_data, dim=0).mean(dim=0).numpy()
    eval_last = torch.stack([s[-1] for s in eval_data]).numpy()
    return ((eval_last - means[None, :]) ** 2).mean(axis=0)


# ----------------- per-target AUROC helpers (binary outcomes) -------------- #
def _safe_auroc(y_true, y_score):
    """AUROC if both classes are present; np.nan otherwise. Suppresses
    UndefinedMetricWarning that sklearn would otherwise throw."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(roc_auc_score(y_true, y_score))


def per_target_auroc_mt(model, eval_data):
    """MiniTransformer per-target AUROC on the last-timestep prediction."""
    model.eval()
    p_local = eval_data[0].shape[1]
    y_true = []
    y_score = []
    with torch.no_grad():
        for s in eval_data:
            if s.shape[0] < 3:
                continue
            x = s.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            y_score.append(pred[0, -1].numpy())
            y_true.append(s[-1].numpy())
    y_true = np.stack(y_true)   # (n_eval, p)
    y_score = np.stack(y_score)
    return np.array([_safe_auroc(y_true[:, r], y_score[:, r]) for r in range(p_local)])


def per_target_auroc_avg(train_data, eval_data):
    """Marginal-mean baseline: predicts a constant -> AUROC undefined,
    by convention 0.5 (no discrimination)."""
    p_local = eval_data[0].shape[1]
    return np.full(p_local, 0.5)


def per_target_auroc_repeat(eval_data):
    """Carry-forward baseline: y_score = previous-visit value (binary)."""
    p_local = eval_data[0].shape[1]
    y_true = []
    y_score = []
    for s in eval_data:
        if s.shape[0] < 2:
            continue
        y_score.append(s[-2].numpy())
        y_true.append(s[-1].numpy())
    y_true = np.stack(y_true)
    y_score = np.stack(y_score)
    return np.array([_safe_auroc(y_true[:, r], y_score[:, r]) for r in range(p_local)])


def per_target_auroc_reg(train_data, eval_data):
    """Per-target Gaussian GLM: y_score = continuous regression prediction."""
    p_local = train_data[0].shape[1]
    Xtr, Ytr = [], []
    for s in train_data:
        for t in range(2, s.shape[0] - 1):
            Xtr.append(s[t - 1].numpy()); Ytr.append(s[t].numpy())
    Xtr = sm.add_constant(np.stack(Xtr), has_constant="add")
    Ytr = np.stack(Ytr)
    Xev, Yev = [], []
    for s in eval_data:
        if s.shape[0] < 3:
            continue
        Xev.append(s[-2].numpy()); Yev.append(s[-1].numpy())
    Xev = sm.add_constant(np.stack(Xev), has_constant="add")
    Yev = np.stack(Yev)
    aurocs = np.zeros(p_local)
    for r in range(p_local):
        m = sm.GLM(Ytr[:, r], Xtr, family=sm.families.Gaussian()).fit()
        aurocs[r] = _safe_auroc(Yev[:, r], m.predict(Xev))
    return aurocs


# --------------------------- training helpers ------------------------------- #
def build_model(p_local):
    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)
    return MiniTransformer(
        p_local, nheads, dk, dv, ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
    ).to(device)


def train_one_fold(train_data, eval_data, seed):
    """Train a MiniTransformer on one fold using `seed` for init."""
    torch.manual_seed(seed)
    model = build_model(train_data[0].shape[1]).to(device)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    eval_loader = DataLoader(
        eval_data, batch_size=len(eval_data), shuffle=False,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=lambda_l2)
    train_mini_transformer(model, train_loader, eval_loader, optimizer,
                           lambda_l2, EPOCHS, device)
    return model


# ----------------------------- main ----------------------------------------- #
def main():
    torch.set_printoptions(sci_mode=False, precision=4)

    print(f"=== Per-target gate + permutation test on {data_str} ===")
    print(f"H={nheads}, C={ncum}, batch={batch_size}, EPOCHS={EPOCHS}, "
          f"k_for_cross_val={k_for_cross_val}, nrepp={nrepp}, "
          f"target_sample_size={target_sample_size}\n")

    # 1. Load data + make folds (same protocol as LORA real-data notebooks)
    data, _ = load_real_data(data_str)
    p = data[0].shape[1]
    print(f"Loaded {len(data)} sequences, p={p}")
    if p != len(VAR_NAMES):
        raise ValueError(f"VAR_NAMES has {len(VAR_NAMES)} entries but p={p}")

    kf = KFold(n_splits=k_for_cross_val, shuffle=True, random_state=42)
    folds = []
    for tr_idx, ev_idx in kf.split(data):
        folds.append(([data[i] for i in tr_idx], [data[i] for i in ev_idx]))

    # 2. CV training: one model per fold with seeds[fold]
    print(f"\n--- 10-fold CV (one seed per fold) ---")
    mse_mt_per_fold  = np.zeros((k_for_cross_val, p))
    mse_avg_per_fold = np.zeros((k_for_cross_val, p))
    mse_reg_per_fold = np.zeros((k_for_cross_val, p))
    mse_rep_per_fold = np.zeros((k_for_cross_val, p))
    auc_mt_per_fold  = np.zeros((k_for_cross_val, p))
    auc_avg_per_fold = np.zeros((k_for_cross_val, p))
    auc_reg_per_fold = np.zeros((k_for_cross_val, p))
    auc_rep_per_fold = np.zeros((k_for_cross_val, p))
    t_cv = time.time()
    for f, (train_data, eval_data) in enumerate(folds):
        t0 = time.time()
        model = train_one_fold(train_data, eval_data, seed=seeds[f])
        # MSEs
        mse_mt_per_fold[f]  = per_target_mse_mt(model, eval_data)
        mse_avg_per_fold[f] = per_target_mse_avg(train_data, eval_data)
        _, _, reg_per_target = calculate_regression_loss(
            train_data, eval_data, predindex=0, return_per_target=True,
        )
        mse_reg_per_fold[f] = reg_per_target
        # Repeat (carry-forward) MSE
        rep_mt = []
        for s in eval_data:
            if s.shape[0] < 2:
                continue
            rep_mt.append(((s[-2].numpy() - s[-1].numpy()) ** 2))
        mse_rep_per_fold[f] = np.stack(rep_mt).mean(axis=0)
        # AUROCs
        auc_mt_per_fold[f]  = per_target_auroc_mt(model, eval_data)
        auc_avg_per_fold[f] = per_target_auroc_avg(train_data, eval_data)
        auc_rep_per_fold[f] = per_target_auroc_repeat(eval_data)
        auc_reg_per_fold[f] = per_target_auroc_reg(train_data, eval_data)
        print(f"  fold {f+1}/{k_for_cross_val} (seed={seeds[f]}): "
              f"MT MSE={mse_mt_per_fold[f].mean():.4f} "
              f"AUC={np.nanmean(auc_mt_per_fold[f]):.3f}  "
              f"reg MSE={mse_reg_per_fold[f].mean():.4f} "
              f"AUC={np.nanmean(auc_reg_per_fold[f]):.3f}  "
              f"[{time.time()-t0:.1f}s]")
    print(f"  CV total: {time.time()-t_cv:.1f}s")

    # 3. Per-target gate (strict: beats avg AND beats reg, on average across folds)
    mt_mean  = mse_mt_per_fold.mean(axis=0)
    mt_std   = mse_mt_per_fold.std(axis=0)
    avg_mean = mse_avg_per_fold.mean(axis=0)
    reg_mean = mse_reg_per_fold.mean(axis=0)
    rep_mean = mse_rep_per_fold.mean(axis=0)
    beats_avg = mt_mean < avg_mean
    beats_reg = mt_mean < reg_mean
    gate_strict = beats_avg & beats_reg

    # AUROC means (NaN-safe; folds where one class is absent return NaN)
    auc_mt_mean  = np.nanmean(auc_mt_per_fold,  axis=0)
    auc_mt_std   = np.nanstd(auc_mt_per_fold,   axis=0)
    auc_avg_mean = np.nanmean(auc_avg_per_fold, axis=0)
    auc_reg_mean = np.nanmean(auc_reg_per_fold, axis=0)
    auc_rep_mean = np.nanmean(auc_rep_per_fold, axis=0)

    # Per-fold gate counts (out of 10)
    folds_beat_avg = (mse_mt_per_fold < mse_avg_per_fold).sum(axis=0)
    folds_beat_reg = (mse_mt_per_fold < mse_reg_per_fold).sum(axis=0)
    folds_pass_strict = ((mse_mt_per_fold < mse_avg_per_fold) &
                         (mse_mt_per_fold < mse_reg_per_fold)).sum(axis=0)

    gate_df = pd.DataFrame({
        "target_var":   np.arange(p),
        "name":         VAR_NAMES,
        "MSE_MT_mean":  mt_mean, "MSE_MT_std": mt_std,
        "MSE_avg_mean": avg_mean,
        "MSE_reg_mean": reg_mean,
        "MSE_rep_mean": rep_mean,
        "AUROC_MT_mean":  auc_mt_mean, "AUROC_MT_std": auc_mt_std,
        "AUROC_avg_mean": auc_avg_mean,
        "AUROC_reg_mean": auc_reg_mean,
        "AUROC_rep_mean": auc_rep_mean,
        "beats_avg_mean":  beats_avg,
        "beats_reg_mean":  beats_reg,
        "gate_strict":     gate_strict,
        "folds_beat_avg":  folds_beat_avg,
        "folds_beat_reg":  folds_beat_reg,
        "folds_pass_strict": folds_pass_strict,
    })
    gate_path = os.path.join(OUT_DIR, "per_target_gate.csv")
    gate_df.to_csv(gate_path, index=False, float_format="%.5f")

    print("\n--- Per-target gate (strict: MT < avg AND MT < reg, CV mean) ---")
    print(gate_df[[
        "name", "MSE_MT_mean", "MSE_avg_mean", "MSE_reg_mean", "MSE_rep_mean",
        "beats_avg_mean", "beats_reg_mean", "gate_strict", "folds_pass_strict",
    ]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n--- Per-target AUROC (CV mean across 10 folds) ---")
    auroc_view = gate_df[[
        "name", "AUROC_MT_mean", "AUROC_avg_mean", "AUROC_reg_mean", "AUROC_rep_mean",
        "gate_strict",
    ]].copy()
    auroc_view.columns = ["name", "AUROC_MT", "AUROC_avg", "AUROC_reg", "AUROC_rep", "gate_strict"]
    print(auroc_view.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("  (AUROC_avg = 0.500 by construction: marginal-mean baseline cannot rank)")

    gate_passing = [r for r in range(p) if gate_strict[r]]
    print(f"\nGate-passing targets ({len(gate_passing)}/{p}): "
          f"{[VAR_NAMES[r] for r in gate_passing]}")
    print(f"Saved gate table to {gate_path}")

    # 4. Train ONE model on the full cohort for the permutation test
    print("\n--- Training full-data model for the permutation test ---")
    print(f"  seed={seed_full_data}")
    torch.manual_seed(seed_full_data)
    model_full = build_model(p).to(device)
    full_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model_full.parameters(), lr=learning_rate)
    t0 = time.time()
    train_mini_transformer(model_full, full_loader, None, optimizer,
                           lambda_l2, EPOCHS, device)
    print(f"  full-data model trained in {time.time()-t0:.1f}s "
          f"(parameters: {count_parameters(model_full)})")

    # 5. For each gate-passing target, run the permutation test
    if not gate_passing:
        print("\nNo gate-passing targets -- skipping permutation tests.")
    else:
        print(f"\n--- Permutation tests for {len(gate_passing)} gate-passing target(s) ---")
        all_pvals = {}
        for r in gate_passing:
            print(f"\n>>> Target {r} = {VAR_NAMES[r]}")
            t0 = time.time()
            avepval, stdpval, _, _, pmat = statistical_testing(
                model_full, data, p, r, nrepp, target_sample_size,
                return_pval_mat=True, seed=seed_full_data,
            )
            print(f"    [{time.time()-t0:.1f}s]")
            ave_np = avepval.numpy()
            std_np = stdpval.numpy()
            order = np.argsort(ave_np)
            df_p = pd.DataFrame({
                "context_var":  order,
                "context_name": [VAR_NAMES[j] for j in order],
                "mean_p":       ave_np[order],
                "std_p":        std_np[order],
            })
            csv_p = os.path.join(OUT_DIR, f"pvalues_{VAR_NAMES[r]}.csv")
            df_p.to_csv(csv_p, index=False, float_format="%.5f")
            all_pvals[r] = df_p
            print("    sorted p-values:")
            for _, row in df_p.iterrows():
                marker = "  *" if row["mean_p"] < 0.05 else "   "
                print(f"   {marker} {row['context_name']:<22s} "
                      f"mean p={row['mean_p']:.4f} +- {row['std_p']:.4f}")

    # 6. Text summary
    lines = []
    lines.append(f"=== Per-target gate + permutation test on {data_str} ===")
    lines.append(f"Hyperparameters: H={nheads} C={ncum} batch={batch_size} "
                 f"EPOCHS={EPOCHS} learning_rate={learning_rate} "
                 f"lambda_l2={lambda_l2}")
    lines.append(f"CV: KFold(n_splits={k_for_cross_val}, random_state=42), "
                 f"per-fold seeds {seeds[:k_for_cross_val]}")
    lines.append(f"Permutation test: full-data model (seed={seed_full_data}), "
                 f"target_sample_size={target_sample_size}, nrepp={nrepp}")
    lines.append("")
    lines.append("Gate (CV-mean MT < CV-mean avg AND CV-mean reg):")
    lines.append(gate_df.to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append(f"Gate-passing targets: "
                 f"{[VAR_NAMES[r] for r in gate_passing]}")
    if gate_passing:
        lines.append("")
        for r in gate_passing:
            lines.append(f"--- p-values for target = {VAR_NAMES[r]} ---")
            lines.append(all_pvals[r].to_string(
                index=False, float_format=lambda v: f"{v:.4f}",
            ))
            lines.append("")
    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\nSaved summary to {os.path.join(OUT_DIR, 'summary.txt')}")
    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
