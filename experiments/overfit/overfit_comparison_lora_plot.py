"""Overlay plot for the Appendix S6 overfit-curve comparison on LORA D1 and D2.

Loads the 10-fold per-epoch train and validation MSE curves from
notebooks/results/full_transformer_overfit/loss_curves_10folds.npz (D1, 150 ep),
notebooks/results/mt_overfit_lora/loss_curves_10folds.npz (D1, 150 ep),
notebooks/results/full_transformer_overfit_ghq_sum/loss_curves_10folds.npz (D2,
500 ep; truncated to the first 150 epochs for direct comparison with D1) and
notebooks/results/mt_overfit_lora_ghq_sum/loss_curves_10folds.npz (D2, 150 ep),
then produces a 2x2 figure: rows = dataset (D1 top, D2 bottom), columns =
training MSE / validation MSE on the target.

The training loss is normalised onto a per-element MSE scale for the
MiniTransformer (its training loop returns the total sum of squared errors
over a batch), so the two models' training curves are directly comparable on
the same axis.

Output: figures/overfit_comparison_lora.pdf and .png.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(_PROJECT_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.data_preparation import load_real_data


DATASETS = [
    {
        "key": "ghq_b_sum",
        "title": "LORA D1",
        "target_label": "GHQ-b",
        "ft_npz": "notebooks/results/full_transformer_overfit/loss_curves_10folds.npz",
        "mt_npz": "notebooks/results/mt_overfit_lora/loss_curves_10folds.npz",
    },
    {
        "key": "ghq_sum",
        "title": "LORA D2",
        "target_label": "GHQ",
        "ft_npz": "notebooks/results/full_transformer_overfit_ghq_sum/loss_curves_10folds.npz",
        "mt_npz": "notebooks/results/mt_overfit_lora_ghq_sum/loss_curves_10folds.npz",
    },
]
PLOT_EPOCHS = 150  # truncate D2 FT (which ran 500 epochs) to match D1's window

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

COL_FT = "#AE232F"   # full transformer = red
COL_MT = "#1D4A91"   # MiniTransformer  = blue
BAND_ALPHA = 0.18


def _normalise_mt_train(mt_train_sum_per_epoch, data_key):
    """Convert MT's sum-of-squared-errors-per-batch training loss to a
    per-element MSE scale comparable to the full transformer's MSE.
    Scaling factor: (avg sequence length - 2) * p, where (T-2) is the number
    of predicted timesteps per row and p is the number of features.
    """
    tensors, _ = load_real_data(data_key)
    tensors = [s if s.shape[0] <= 10 else s[-10:] for s in tensors]
    lens = np.array([s.shape[0] for s in tensors])
    p = tensors[0].shape[1]
    avg_T = float(lens.mean())
    scaling = (avg_T - 2) * p
    return mt_train_sum_per_epoch / scaling


def _load(ds):
    ft = np.load(ds["ft_npz"])
    mt = np.load(ds["mt_npz"])
    ft_train = ft["train"][:, :PLOT_EPOCHS]
    ft_val_tar = ft["val_target"][:, :PLOT_EPOCHS]
    mt_train_raw = mt["train"][:, :PLOT_EPOCHS]
    mt_val_tar = mt["val_target"][:, :PLOT_EPOCHS]
    mt_train = _normalise_mt_train(mt_train_raw, ds["key"])
    return ft_train, ft_val_tar, mt_train, mt_val_tar


def _plot_pair(ax_train, ax_val, ft_train, ft_val_tar, mt_train, mt_val_tar,
               title_prefix, target_label):
    epochs = np.arange(1, ft_train.shape[1] + 1)

    # Training MSE
    for arr, col, label in [
        (ft_train, COL_FT, "Full transformer (797k params)"),
        (mt_train, COL_MT, "MiniTransformer (420 params)"),
    ]:
        mu = arr.mean(axis=0); sd = arr.std(axis=0)
        ax_train.plot(epochs, mu, color=col, label=label, linewidth=1.8)
        ax_train.fill_between(epochs, mu - sd, mu + sd, color=col, alpha=BAND_ALPHA)
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Training MSE")
    ax_train.set_title(f"{title_prefix}: training MSE")
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(fontsize=8, loc="upper right")

    # Validation MSE on target
    for arr, col, label in [
        (ft_val_tar, COL_FT, "Full transformer (797k params)"),
        (mt_val_tar, COL_MT, "MiniTransformer (420 params)"),
    ]:
        mu = arr.mean(axis=0); sd = arr.std(axis=0)
        ax_val.plot(epochs, mu, color=col, label=label, linewidth=1.8)
        ax_val.fill_between(epochs, mu - sd, mu + sd, color=col, alpha=BAND_ALPHA)
    ft_min_ep = int(np.argmin(ft_val_tar.mean(axis=0))) + 1
    mt_min_ep = int(np.argmin(mt_val_tar.mean(axis=0))) + 1
    ax_val.axvline(ft_min_ep, color=COL_FT, linestyle="--", linewidth=1, alpha=0.6)
    ax_val.axvline(mt_min_ep, color=COL_MT, linestyle="--", linewidth=1, alpha=0.6)
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel(f"Validation MSE on target ({target_label})")
    ax_val.set_title(f"{title_prefix}: validation MSE on the target")
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(fontsize=8, loc="lower right")

    return {
        "ft_min_ep": ft_min_ep,
        "ft_min_val": float(ft_val_tar.mean(axis=0).min()),
        "ft_min_std": float(ft_val_tar.std(axis=0)[ft_min_ep - 1]),
        "ft_final_val": float(ft_val_tar.mean(axis=0)[-1]),
        "ft_final_std": float(ft_val_tar.std(axis=0)[-1]),
        "mt_min_ep": mt_min_ep,
        "mt_min_val": float(mt_val_tar.mean(axis=0).min()),
        "mt_min_std": float(mt_val_tar.std(axis=0)[mt_min_ep - 1]),
        "mt_final_val": float(mt_val_tar.mean(axis=0)[-1]),
        "mt_final_std": float(mt_val_tar.std(axis=0)[-1]),
    }


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8), sharex=True)

    summaries = {}
    for row, ds in enumerate(DATASETS):
        ft_train, ft_val_tar, mt_train, mt_val_tar = _load(ds)
        print(f"{ds['key']}: FT train {ft_train.shape}, MT train {mt_train.shape}")
        summaries[ds["key"]] = _plot_pair(
            axes[row, 0], axes[row, 1],
            ft_train, ft_val_tar, mt_train, mt_val_tar,
            title_prefix=ds["title"], target_label=ds["target_label"],
        )

    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "overfit_comparison_lora.pdf")
    png_path = os.path.join(OUT_DIR, "overfit_comparison_lora.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Summary ---")
    for key, s in summaries.items():
        print(f"\n{key}:")
        print(f"  FT min: epoch {s['ft_min_ep']}, "
              f"value {s['ft_min_val']:.4f} +/- {s['ft_min_std']:.4f}")
        print(f"  FT final (epoch {PLOT_EPOCHS}): "
              f"{s['ft_final_val']:.4f} +/- {s['ft_final_std']:.4f}")
        print(f"  MT min: epoch {s['mt_min_ep']}, "
              f"value {s['mt_min_val']:.4f} +/- {s['mt_min_std']:.4f}")
        print(f"  MT final (epoch {PLOT_EPOCHS}): "
              f"{s['mt_final_val']:.4f} +/- {s['mt_final_std']:.4f}")

    print(f"\nFigure saved to:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()
