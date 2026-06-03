"""Companion to notebooks/full_transformer_overfit_lora.py.

Trains the paper's MiniTransformer (H=8 heads, C=8 cumulants, paper §3.2
defaults) on LORA D1 for 150 epochs per fold across all ten folds of the
same KFold(n_splits=10, shuffle=True, random_state=42) split used in
Section 3.2 of the manuscript. For each fold the per-epoch training loss
and per-epoch validation MSE (averaged over variables and on the GHQ-b
target) are recorded, so that the resulting fold-averaged trajectory can
be overlaid directly on the over-parameterised vanilla-transformer curve
produced by notebooks/full_transformer_overfit_lora.py.

Outputs (notebooks/results/mt_overfit_lora/):
    loss_curves_10folds.npz   per-fold (n_folds x EPOCHS) arrays: train, val_all, val_target
    loss_curves.csv            fold-averaged means + stds per epoch
    summary.txt                fold-averaged min epoch / value / final value
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

from src.data_preparation import load_real_data, collate_function
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer_one_epoch,
)


device = torch.device("cpu")

# Configuration
DATA_STR = os.environ.get("OVERFIT_DATA", "ghq_b_sum")  # "ghq_b_sum" = LORA D1, "ghq_sum" = LORA D2
TARGET_IDX = 9
MAXLEN = 10
EPOCHS = int(os.environ.get("MT_EPOCHS", 150))
BATCH_SIZE = 2  # paper §3.2 setting for LORA
LR = 1e-3
LAMBDA_L2 = 1e-3
# Canonical paper seeds, one per fold (matches run_baselines_real_data.py and
# the rest of the evaluation; one seed per fold, NOT SEED+f_idx).
SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]
N_SPLITS = 10
CV_RANDOM_STATE = 42

# Paper §3.2 MiniTransformer hyperparameters
NHEADS = 8
NCUM = 8
DK = 1
DV = 1

OUT_DIR = ("notebooks/results/mt_overfit_lora"
           if DATA_STR == "ghq_b_sum"
           else f"notebooks/results/mt_overfit_lora_{DATA_STR}")
os.makedirs(OUT_DIR, exist_ok=True)


def load_d1():
    tensors, _ = load_real_data(DATA_STR)
    return [s if s.shape[0] <= MAXLEN else s[-MAXLEN:] for s in tensors]


def val_loss_and_target_mse(model, val_data, p):
    """Validation: per-variable squared error on the last visit averaged
    across patients, returning both overall mean MSE and per-target MSE.
    Mirrors the metric used in notebooks/full_transformer_overfit_lora.py
    so the trajectories are directly comparable."""
    model.eval()
    se_tot, se_tar, n_tot, n_tar = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for seq in val_data:
            if seq.shape[0] < 3:
                continue
            x = seq.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            last_pred = pred[0, -1].numpy()
            last_true = seq[-1].numpy()
            se_tot += float(((last_pred - last_true) ** 2).sum())
            n_tot += p
            se_tar += float((last_pred[TARGET_IDX] - last_true[TARGET_IDX]) ** 2)
            n_tar += 1
    return se_tot / max(n_tot, 1), se_tar / max(n_tar, 1)


def main():
    t0 = time.time()
    print(f"=== MiniTransformer overfit curve on LORA {DATA_STR} (10-fold CV) ===")
    print(f"Hyperparameters: H={NHEADS}, C={NCUM}, dk={DK}, dv={DV} "
          f"(paper §3.2 defaults)")
    print(f"Train: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, "
          f"lambda_l2={LAMBDA_L2}")

    tensors = load_d1()
    p = tensors[0].shape[1]
    n_total = len(tensors)
    print(f"Loaded LORA {DATA_STR}: n={n_total}, p={p}")

    mask_pairwise = create_custom_mask_pair(MAXLEN, device)
    distance_to_end_matrix = create_distance_to_end_matrix(MAXLEN, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(MAXLEN, device)

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                       random_state=CV_RANDOM_STATE).split(np.arange(n_total)))
    assert len(folds) == len(SEEDS)

    train_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)
    val_all_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)
    val_tar_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)

    n_params_ref = None
    for f_idx, ((tr_idx, te_idx), seed) in enumerate(zip(folds, SEEDS)):
        train_data = [tensors[i] for i in tr_idx]
        val_data = [tensors[i] for i in te_idx]
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MiniTransformer(
            p, NHEADS, DK, DV, NCUM,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
        ).to(device)
        if n_params_ref is None:
            n_params_ref = sum(par.numel() for par in model.parameters()
                               if par.requires_grad)
            print(f"Model parameters: {n_params_ref}")
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_function, num_workers=0)
        opt = optim.Adam(model.parameters(), lr=LR)
        print(f"\n--- fold {f_idx+1}/{N_SPLITS} (train={len(train_data)}, "
              f"val={len(val_data)}) ---")
        for ep in range(EPOCHS):
            tr_loss, _ = train_mini_transformer_one_epoch(
                model, loader, opt, LAMBDA_L2, device,
            )
            v_all, v_tar = val_loss_and_target_mse(model, val_data, p)
            train_curves[f_idx, ep] = float(tr_loss)
            val_all_curves[f_idx, ep] = v_all
            val_tar_curves[f_idx, ep] = v_tar
            if (ep + 1) in (1, 5, 10, 50, 100, EPOCHS):
                print(f"  epoch {ep+1:>4}: train={tr_loss:.4f}  "
                      f"val_all={v_all:.4f}  val_target={v_tar:.4f}")
        min_ep = int(np.argmin(val_tar_curves[f_idx])) + 1
        min_val = float(val_tar_curves[f_idx].min())
        final = float(val_tar_curves[f_idx, -1])
        print(f"  fold {f_idx+1} val_target min: epoch {min_ep}, "
              f"value {min_val:.4f}; final {final:.4f}")

    # Persist raw per-fold curves
    np.savez(
        os.path.join(OUT_DIR, "loss_curves_10folds.npz"),
        train=train_curves, val_all=val_all_curves, val_target=val_tar_curves,
    )

    # Aggregate
    epochs = np.arange(1, EPOCHS + 1)
    df = pd.DataFrame({
        "epoch": epochs,
        "train_mean": train_curves.mean(axis=0),
        "train_std": train_curves.std(axis=0),
        "val_all_mean": val_all_curves.mean(axis=0),
        "val_all_std": val_all_curves.std(axis=0),
        "val_target_mean": val_tar_curves.mean(axis=0),
        "val_target_std": val_tar_curves.std(axis=0),
    })
    df.to_csv(os.path.join(OUT_DIR, "loss_curves.csv"), index=False,
              float_format="%.5f")

    # Summary
    min_ep_tar = int(df["val_target_mean"].idxmin() + 1)
    min_val_tar = float(df["val_target_mean"].min())
    min_val_tar_std = float(df.loc[df["val_target_mean"].idxmin(),
                                   "val_target_std"])
    final_val_tar = float(df.iloc[-1]["val_target_mean"])
    final_val_tar_std = float(df.iloc[-1]["val_target_std"])

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("=== MiniTransformer overfit curve, LORA D1 (10-fold CV) ===\n")
        f.write(f"Dataset: LORA {DATA_STR} (n={n_total}, p={p}, maxlen={MAXLEN})\n")
        f.write(f"Hyperparameters: H={NHEADS}, C={NCUM}, dk={DK}, dv={DV}\n")
        f.write(f"Parameters: {n_params_ref}\n")
        f.write(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}\n")
        f.write(f"Splits: {N_SPLITS}-fold KFold (random_state={CV_RANDOM_STATE})\n\n")
        f.write(f"Fold-averaged val MSE_target min: epoch {min_ep_tar}, "
                f"value {min_val_tar:.4f} +/- {min_val_tar_std:.4f}\n")
        f.write(f"Fold-averaged val MSE_target final (epoch {EPOCHS}): "
                f"{final_val_tar:.4f} +/- {final_val_tar_std:.4f}\n")

    print(f"\nFold-averaged val MSE_target min: epoch {min_ep_tar}, "
          f"value {min_val_tar:.4f} +/- {min_val_tar_std:.4f}")
    print(f"Fold-averaged val MSE_target final (epoch {EPOCHS}): "
          f"{final_val_tar:.4f} +/- {final_val_tar_std:.4f}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    print(f"Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
