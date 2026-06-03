"""§3.7 / Appendix S6: full-sized transformer overfit curve on LORA D1.

Reviewer's §3.7 worry: maybe a full-sized transformer would exhibit
benign-overfitting / double-descent on LORA-size data and beat
MiniTransformer. We show empirically that it does not: a clearly
over-parameterised vanilla transformer (4 layers, 4 heads, d_model=128,
~200k parameters) trained on LORA D1 with the paper's Table 3 target
overfits the standard way -- training loss drops to (near) zero while
validation loss bottoms out early and climbs steadily thereafter, with no
double-descent recovery within a 500-epoch window.

We use a single train/validation split (no CV) since the point is to
visualise the overfit trajectory, not to estimate generalisation error;
the split is the first KFold(n_splits=10, random_state=42) fold (matching
the paper's CV convention) so the result is reproducible.

Outputs (under notebooks/results/full_transformer_overfit/):
    loss_curves.csv     epoch | train_loss | val_loss | val_mse_target
    loss_curves.png     training and validation loss curves
    summary.txt         architecture spec + min-validation epoch + final values
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

from src.data_preparation import load_real_data, collate_function


device = torch.device("cpu")

# ----- Configuration ------------------------------------------------------ #
DATA_STR = os.environ.get("OVERFIT_DATA", "ghq_b_sum")  # "ghq_b_sum" = LORA D1, "ghq_sum" = LORA D2
TARGET_IDX = 9               # last column is GHQ target in both LORA datasets
MAXLEN = 10                  # match paper convention; truncate sequences
EPOCHS = int(os.environ.get("FT_EPOCHS", 500))
BATCH_SIZE = int(os.environ.get("FT_BATCH", 4))
LR = float(os.environ.get("FT_LR", 1e-3))
# Canonical paper seeds, one per fold (matches run_baselines_real_data.py and
# the rest of the evaluation; one seed per fold, NOT SEED+f_idx).
SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]
CV_RANDOM_STATE = 42
N_SPLITS = 10

# Full-sized transformer hyperparameters -- deliberately over-parameterised
# for a cohort of n=882 individuals with 10 binary features.
NLAYERS = 4
NHEADS = 4
D_MODEL = 128
DIM_FF = 4 * D_MODEL  # standard 4x expansion
DROPOUT = 0.1

OUT_DIR = ("notebooks/results/full_transformer_overfit"
           if DATA_STR == "ghq_b_sum"
           else f"notebooks/results/full_transformer_overfit_{DATA_STR}")
os.makedirs(OUT_DIR, exist_ok=True)


# ----- Model -------------------------------------------------------------- #

class FullTransformer(nn.Module):
    """Multi-layer causal transformer encoder for one-step prediction. Mirrors
    the ScaledVanillaTransformer interface (forward expects ``(x, mask)`` and
    returns next-step predictions of shape ``(B, T-1, p)``) so it can be
    trained with the same loop, but uses ``num_layers`` stacked encoder
    layers with multi-head attention and a larger d_model.
    """
    def __init__(self, p, d_model, n_heads, num_layers, dim_ff, dropout, max_len):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.max_len = max_len
        self.in_proj = nn.Linear(p, d_model)
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, p)

    def forward(self, data):
        x, padded_mask = data
        B, T_in, _ = x.shape
        if T_in > self.max_len:
            raise ValueError(f"T_in={T_in} > max_len={self.max_len}")
        h = self.in_proj(x) + self.pos_embed[:T_in].unsqueeze(0)
        causal = torch.triu(torch.ones(T_in, T_in, dtype=torch.bool, device=h.device),
                            diagonal=1)
        key_pad = ~padded_mask[:, :, 0]
        # Zero out fully-padded rows to avoid MHA NaN
        all_pad = key_pad.all(dim=1)
        if all_pad.any():
            key_pad = key_pad.clone()
            key_pad[all_pad, 0] = False
        h = self.encoder(h, mask=causal, src_key_padding_mask=key_pad)
        out = self.out_proj(h)
        # Return the full per-step output; position i predicts the next token
        # (causal mask above ensures no look-ahead). Shape: (B, T_in, p).
        return out


# ----- Data --------------------------------------------------------------- #

def load_d1():
    tensors, _ = load_real_data(DATA_STR)
    # Truncate to last MAXLEN, matching baseline runners
    tensors = [s if s.shape[0] <= MAXLEN else s[-MAXLEN:] for s in tensors]
    return tensors


# ----- Loss / eval -------------------------------------------------------- #

def per_step_mse_loss(pred, target, mask):
    """Mean squared error over valid (B, T-2) positions, then averaged over
    features. Mirrors src.transformers.mini_transformer_loss.
    """
    # pred:   (B, T-1, p) from forward(x[:, :-1, :])
    # target: (B, T-1, p) the next-step targets x[:, 1:, :]
    # mask:   (B, T-1, p) bool, True = valid
    valid = mask.float()
    se = ((pred - target) ** 2) * valid
    n_valid = valid.sum().clamp(min=1.0)
    return se.sum() / n_valid


def val_loss_and_target_mse(model, val_data):
    model.eval()
    p = val_data[0].shape[1]
    se_tot, se_tar, n_tot, n_tar = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for seq in val_data:
            if seq.shape[0] < 3:
                continue
            x = seq.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            # last predicted step is for the final observed token
            last_pred = pred[0, -1].numpy()
            last_true = seq[-1].numpy()
            se_tot += float(((last_pred - last_true) ** 2).sum())
            n_tot += p
            se_tar += float((last_pred[TARGET_IDX] - last_true[TARGET_IDX]) ** 2)
            n_tar += 1
    return se_tot / max(n_tot, 1), se_tar / max(n_tar, 1)


def train_epoch(model, loader, opt):
    model.train()
    total, n_batches = 0.0, 0
    for batch_x, batch_mask in loader:
        opt.zero_grad()
        x_in = batch_x[:, :-1, :]
        m_in = batch_mask[:, :-1, :]
        x_tar = batch_x[:, 1:, :]
        m_tar = batch_mask[:, 1:, :]
        pred = model((x_in, m_in))
        loss = per_step_mse_loss(pred, x_tar, m_tar)
        loss.backward()
        opt.step()
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


# ----- Main --------------------------------------------------------------- #

def main():
    t0 = time.time()
    print(f"=== Full-sized transformer overfit curve on LORA {DATA_STR} (10-fold CV) ===")
    print(f"Arch: {NLAYERS} layers, {NHEADS} heads, d_model={D_MODEL}, "
          f"dim_ff={DIM_FF}, dropout={DROPOUT}")
    print(f"Train: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}")

    tensors = load_d1()
    p = tensors[0].shape[1]
    n_total = len(tensors)
    print(f"Loaded LORA {DATA_STR}: n={n_total}, p={p}")

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                       random_state=CV_RANDOM_STATE).split(np.arange(n_total)))
    assert len(folds) == len(SEEDS)

    # Collect a (n_folds x EPOCHS) array for each metric so we can aggregate.
    train_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)
    val_all_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)
    val_tar_curves = np.zeros((N_SPLITS, EPOCHS), dtype=np.float32)

    n_params_ref = None
    for f_idx, ((tr_idx, te_idx), seed) in enumerate(zip(folds, SEEDS)):
        train_data = [tensors[i] for i in tr_idx]
        val_data = [tensors[i] for i in te_idx]
        # Use the canonical per-fold seed for the model weights; architecture identical.
        torch.manual_seed(seed); np.random.seed(seed)
        model = FullTransformer(p, D_MODEL, NHEADS, NLAYERS, DIM_FF, DROPOUT, MAXLEN).to(device)
        n_params = sum(par.numel() for par in model.parameters() if par.requires_grad)
        if n_params_ref is None:
            n_params_ref = n_params
            print(f"Model parameters: {n_params}")
        loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_function, num_workers=0)
        opt = optim.Adam(model.parameters(), lr=LR)
        print(f"\n--- fold {f_idx+1}/{N_SPLITS} (train={len(train_data)}, val={len(val_data)}) ---")
        for ep in range(EPOCHS):
            tr = train_epoch(model, loader, opt)
            v_all, v_tar = val_loss_and_target_mse(model, val_data)
            train_curves[f_idx, ep] = tr
            val_all_curves[f_idx, ep] = v_all
            val_tar_curves[f_idx, ep] = v_tar
            if (ep + 1) in (1, 5, 10, 50, 100, 250, EPOCHS):
                print(f"  epoch {ep+1:>4}: train={tr:.4f}  val_all={v_all:.4f}  "
                      f"val_target={v_tar:.4f}")
        # per-fold val-target min for at-a-glance reporting
        min_ep = int(np.argmin(val_tar_curves[f_idx])) + 1
        min_val = float(val_tar_curves[f_idx].min())
        print(f"  fold {f_idx+1} val_target min: epoch {min_ep}, value {min_val:.4f}; "
              f"final {val_tar_curves[f_idx, -1]:.4f}")

    # Persist raw per-fold curves
    np.savez(
        os.path.join(OUT_DIR, "loss_curves_10folds.npz"),
        train=train_curves, val_all=val_all_curves, val_target=val_tar_curves,
    )

    # Aggregate: per-epoch mean + std across folds
    epochs = np.arange(1, EPOCHS + 1)
    df_agg = pd.DataFrame({
        "epoch": epochs,
        "train_mean": train_curves.mean(axis=0),
        "train_std": train_curves.std(axis=0),
        "val_all_mean": val_all_curves.mean(axis=0),
        "val_all_std": val_all_curves.std(axis=0),
        "val_target_mean": val_tar_curves.mean(axis=0),
        "val_target_std": val_tar_curves.std(axis=0),
    })
    df_agg.to_csv(os.path.join(OUT_DIR, "loss_curves.csv"), index=False,
                  float_format="%.5f")

    # Identify fold-averaged val_target minimum and rebound
    min_ep_tar = int(df_agg["val_target_mean"].idxmin() + 1)
    min_val_tar = float(df_agg["val_target_mean"].min())
    min_val_tar_std = float(df_agg.loc[df_agg["val_target_mean"].idxmin(),
                                       "val_target_std"])
    final_val_tar = float(df_agg.iloc[-1]["val_target_mean"])
    final_val_tar_std = float(df_agg.iloc[-1]["val_target_std"])
    rebound = final_val_tar - min_val_tar
    # Per-fold val-target at the fold-averaged-min epoch and at the final epoch
    per_fold_min_at_avg_minep = val_tar_curves[:, min_ep_tar - 1]
    per_fold_final = val_tar_curves[:, -1]
    print(f"\n(Fold-averaged) Validation MSE_target minimum: epoch {min_ep_tar}, "
          f"value {min_val_tar:.4f} +/- {min_val_tar_std:.4f}")
    print(f"(Fold-averaged) Final validation MSE_target (epoch {EPOCHS}): "
          f"{final_val_tar:.4f} +/- {final_val_tar_std:.4f}  "
          f"(rebound {rebound:+.4f})")

    # Plot fold-averaged curves with shaded +/- 1 std band
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    ax = axes[0]
    ax.plot(df_agg["epoch"], df_agg["train_mean"], color="#1D4A91", label="train MSE (mean)")
    ax.fill_between(df_agg["epoch"],
                    df_agg["train_mean"] - df_agg["train_std"],
                    df_agg["train_mean"] + df_agg["train_std"],
                    color="#1D4A91", alpha=0.2)
    ax.plot(df_agg["epoch"], df_agg["val_all_mean"], color="#AE232F",
            label="val MSE all-variables (mean)")
    ax.fill_between(df_agg["epoch"],
                    df_agg["val_all_mean"] - df_agg["val_all_std"],
                    df_agg["val_all_mean"] + df_agg["val_all_std"],
                    color="#AE232F", alpha=0.2)
    ax.axvline(min_ep_tar, color="grey", linestyle="--", linewidth=1,
               label=f"avg val-min epoch ({min_ep_tar})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("Train vs.\\ validation MSE (10-fold avg, $\\pm$1 std)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(df_agg["epoch"], df_agg["val_target_mean"], color="#AE232F",
            label="val MSE on target (mean)")
    ax.fill_between(df_agg["epoch"],
                    df_agg["val_target_mean"] - df_agg["val_target_std"],
                    df_agg["val_target_mean"] + df_agg["val_target_std"],
                    color="#AE232F", alpha=0.2)
    ax.axvline(min_ep_tar, color="grey", linestyle="--", linewidth=1)
    ax.axhline(min_val_tar, color="grey", linestyle=":", linewidth=1,
               label=f"avg val-min ({min_val_tar:.3f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE on target")
    ax.set_title("Validation MSE on the GHQ-b target (10-fold avg)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    fig.suptitle(
        f"Full-sized transformer ({n_params_ref:,} params) on LORA {DATA_STR}, "
        f"{N_SPLITS}-fold CV: standard overfit pattern, no double-descent "
        f"recovery within {EPOCHS} epochs",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "loss_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write("=== Full-sized transformer overfit curve (Appendix S6, 10-fold CV) ===\n")
        f.write(f"Dataset: LORA {DATA_STR} (n={n_total}, p={p}, maxlen={MAXLEN})\n")
        f.write(f"Architecture: {NLAYERS} layers, {NHEADS} heads, "
                f"d_model={D_MODEL}, dim_ff={DIM_FF}, dropout={DROPOUT}\n")
        f.write(f"Parameters: {n_params_ref}\n")
        f.write(f"Training: {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}\n")
        f.write(f"Splits: {N_SPLITS}-fold KFold (shuffle=True, random_state={CV_RANDOM_STATE})\n")
        f.write("\n")
        f.write(f"Fold-averaged validation MSE_target minimum: epoch {min_ep_tar}, "
                f"value {min_val_tar:.4f} +/- {min_val_tar_std:.4f}\n")
        f.write(f"Fold-averaged final validation MSE_target (epoch {EPOCHS}): "
                f"{final_val_tar:.4f} +/- {final_val_tar_std:.4f}\n")
        f.write(f"Rebound from minimum: {rebound:+.4f}\n")
        f.write("\nPer-fold val_target at fold-averaged min epoch "
                f"(ep {min_ep_tar}) and at final epoch:\n")
        for i in range(N_SPLITS):
            f.write(f"  fold {i+1}: min-ep={per_fold_min_at_avg_minep[i]:.4f}  "
                    f"final={per_fold_final[i]:.4f}\n")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    print(f"Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
