"""Baseline comparison on the PBC2 controlled simulation (synthetic ascites target).

Same protocol as notebooks/run_baselines_real_data.py (10-fold CV, paper seeds,
paper hyperparameters), but the data is the binarised PBC2 cohort with the
ascites column overwritten by the synthetic j1 -> j2 -> j3 target used in
notebooks/pbc2_controlled_simulation.py. This produces the PBC2-sim column of
the response letter's S3.1 baseline table for the non-MiniTransformer rows.

The MiniTransformer cell for this column is already reported by
pbc2_controlled_simulation.py (0.097 +/- 0.018); the present script reproduces
it as a sanity check and additionally trains the four parameter-matched
neural baselines plus avg/reg/repeat on the same folds.

Outputs (notebooks/results/baselines_pbc2_controlled_sim/):
    summary_10folds.csv   per-model mean/std MSE and MSE_target across folds
    summary.txt           human-readable summary
"""
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
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
    train_mini_transformer,
    count_parameters,
)
from src.evaluation import calculate_regression_loss
from src.baselines.scaled_vanilla_transformer import (
    ScaledVanillaTransformer, find_matched_d_model,
)
from src.baselines.dlinear import DLinear
from src.baselines.kernel_attention_no_decay import KernelAttentionNoDecay
from src.baselines.itransformer import ITransformer
from src.baselines.rope_attention import RoPEOrDecayMiniTransformer

from src.pbc2_substrate import inject_synthetic_target, COL_ASCITES


device = torch.device("cpu")

SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]
N_SPLITS = 10
CV_RANDOM_STATE = 42
MAXLEN = 10

mt_nheads, mt_ncum, mt_dk, mt_dv = 8, 8, 1, 1
batch_size, learning_rate, lambda_l2 = 2, 1e-3, 1e-3
EPOCHS = int(os.environ.get("BSL_EPOCHS", 150))

OUT_DIR = "notebooks/results/baselines_pbc2_controlled_sim"
os.makedirs(OUT_DIR, exist_ok=True)


def _itransformer_params(p, d_model, history_len, dim_feedforward_mult=2, n_heads=1):
    m = ITransformer(p, d_model=d_model, n_heads=n_heads,
                     history_len=history_len,
                     dim_feedforward=dim_feedforward_mult * d_model,
                     max_len=history_len)
    return sum(par.numel() for par in m.parameters() if par.requires_grad)


def find_itransformer_d_model(p, target_params, history_len, search_range=(2, 16)):
    best_d, best_diff, best_n = None, float("inf"), None
    for d in range(search_range[0], search_range[1] + 1):
        n = _itransformer_params(p, d, history_len)
        diff = abs(n - target_params)
        if diff < best_diff:
            best_d, best_diff, best_n = d, diff, n
    return best_d, best_n


def per_target_mse_avg(train_data, test_data):
    means = torch.cat(train_data, dim=0).mean(dim=0).numpy()
    test_last = torch.stack([s[-1] for s in test_data]).numpy()
    return ((test_last - means[None, :]) ** 2).mean(axis=0)


def per_target_mse_reg(train_data, test_data):
    _, _, per_target = calculate_regression_loss(
        train_data, test_data, predindex=0, return_per_target=True,
    )
    return per_target


def per_target_mse_repeat(test_data):
    p_local = test_data[0].shape[1]
    se, n_eval = np.zeros(p_local), 0
    for s in test_data:
        if s.shape[0] < 2:
            continue
        se += (s[-1].numpy() - s[-2].numpy()) ** 2
        n_eval += 1
    return se / max(n_eval, 1)


def per_target_mse_torch_model(model, test_data):
    model.eval()
    p_local = test_data[0].shape[1]
    se, n_eval = np.zeros(p_local), 0
    with torch.no_grad():
        for seq in test_data:
            if seq.shape[0] < 3:
                continue
            x = seq.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            se += (pred[0, -1].numpy() - seq[-1].numpy()) ** 2
            n_eval += 1
    return se / max(n_eval, 1)


def train_and_eval(model, train_data, test_data):
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_function, num_workers=0)
    # Pass the eval loader exactly as simulation_experiments.ipynb does (see the
    # note in run_baselines_real_data.py): the per-epoch next(iter(eval_loader))
    # consumes the global torch RNG, so passing None would desync the SGD
    # trajectory from the established protocol.
    eval_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False,
                             collate_fn=collate_function, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    train_mini_transformer(model, loader, eval_loader, opt, lambda_l2, EPOCHS, device)
    return per_target_mse_torch_model(model, test_data)


def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    print("=== Baselines on PBC2 controlled simulation (synthetic ascites target) ===")
    print(f"EPOCHS={EPOCHS}  batch={batch_size}  CV(random_state={CV_RANDOM_STATE})")
    print(f"seeds (per fold)={SEEDS}\n")

    raw_tensors, _ = load_real_data("pbc2")
    raw_tensors = [s if s.shape[0] <= MAXLEN else s[-MAXLEN:] for s in raw_tensors]
    p = raw_tensors[0].shape[1]
    assert p == 10, f"expected p=10 binarised PBC2, got p={p}"
    target_idx = COL_ASCITES
    assert target_idx == p - 1

    seqs, _z, y_list = inject_synthetic_target(raw_tensors, seed=42)
    flat_y = np.concatenate(y_list)
    print(f"Synthetic target marginal P(y=1) = {flat_y.mean():.4f} "
          f"(matches notebooks/results/pbc2_controlled_simulation/marginal.txt)\n")

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                       random_state=CV_RANDOM_STATE).split(np.arange(len(seqs))))
    assert len(folds) == len(SEEDS)

    mask_pairwise = create_custom_mask_pair(MAXLEN, device)
    distance_to_end_matrix = create_distance_to_end_matrix(MAXLEN, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(MAXLEN, device)

    torch.manual_seed(0)
    mt0 = MiniTransformer(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                          mask_pairwise, pairwise_distance_matrix,
                          distance_to_end_matrix, device).to(device)
    n_params_mt = count_parameters(mt0)
    d_model_svt, n_params_svt = find_matched_d_model(p, n_params_mt, max_len=MAXLEN)
    d_model_it,  n_params_it  = find_itransformer_d_model(p, n_params_mt, history_len=MAXLEN)
    print(f"Param targets: MT={n_params_mt}  SVT={n_params_svt} (d_model={d_model_svt})  "
          f"iTr={n_params_it} (d_model={d_model_it})\n")

    nd0 = KernelAttentionNoDecay(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                                 mask_pairwise, pairwise_distance_matrix,
                                 distance_to_end_matrix, device).to(device)
    n_params_nd = sum(par.numel() for par in nd0.parameters() if par.requires_grad)
    dl0 = DLinear(p, history_len=MAXLEN, kernel_size=5, max_len=MAXLEN, device=device)
    n_params_dl = sum(par.numel() for par in dl0.parameters() if par.requires_grad)
    rope_dk = 2
    rope0 = RoPEOrDecayMiniTransformer(
        p, mt_nheads, rope_dk, mt_dv, mt_ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
        positional_scheme="rope",
    ).to(device)
    n_params_rope = sum(par.numel() for par in rope0.parameters() if par.requires_grad)
    n_params = {"MiniTransformer": n_params_mt, "NoDecay": n_params_nd,
                "ScaledVanillaTr": n_params_svt, "iTransformer": n_params_it,
                "DLinear": n_params_dl, "RoPEAttention": n_params_rope}

    results = {k: [] for k in [
        "MiniTransformer", "NoDecay", "ScaledVanillaTr",
        "iTransformer", "DLinear", "RoPEAttention",
        "avg", "reg", "repeat",
    ]}

    t_start = time.time()
    for f_idx, ((tr_idx, te_idx), seed) in enumerate(zip(folds, SEEDS)):
        train_data = [seqs[i] for i in tr_idx]
        test_data = [seqs[i] for i in te_idx]
        print(f"--- fold {f_idx+1}/{N_SPLITS}, seed={seed} "
              f"(train={len(train_data)}, test={len(test_data)}) ---")

        torch.manual_seed(seed); np.random.seed(seed)
        mt = MiniTransformer(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                             mask_pairwise, pairwise_distance_matrix,
                             distance_to_end_matrix, device).to(device)
        mse = train_and_eval(mt, train_data, test_data)
        results["MiniTransformer"].append(mse)
        print(f"  MT      MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        torch.manual_seed(seed); np.random.seed(seed)
        nd = KernelAttentionNoDecay(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                                    mask_pairwise, pairwise_distance_matrix,
                                    distance_to_end_matrix, device).to(device)
        mse = train_and_eval(nd, train_data, test_data)
        results["NoDecay"].append(mse)
        print(f"  NoDecay MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        torch.manual_seed(seed); np.random.seed(seed)
        svt = ScaledVanillaTransformer(p, d_model=d_model_svt,
                                       max_len=MAXLEN, device=device)
        mse = train_and_eval(svt, train_data, test_data)
        results["ScaledVanillaTr"].append(mse)
        print(f"  SVT     MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        torch.manual_seed(seed); np.random.seed(seed)
        it = ITransformer(p, d_model=d_model_it, n_heads=1,
                          history_len=MAXLEN, max_len=MAXLEN, device=device)
        mse = train_and_eval(it, train_data, test_data)
        results["iTransformer"].append(mse)
        print(f"  iTr     MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        torch.manual_seed(seed); np.random.seed(seed)
        dl = DLinear(p, history_len=MAXLEN, kernel_size=5, max_len=MAXLEN, device=device)
        mse = train_and_eval(dl, train_data, test_data)
        results["DLinear"].append(mse)
        print(f"  DL      MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        torch.manual_seed(seed); np.random.seed(seed)
        rope = RoPEOrDecayMiniTransformer(
            p, mt_nheads, rope_dk, mt_dv, mt_ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
            positional_scheme="rope",
        ).to(device)
        mse = train_and_eval(rope, train_data, test_data)
        results["RoPEAttention"].append(mse)
        print(f"  RoPE    MSE={mse.mean():.4f}  MSE_target={mse[target_idx]:.4f}")

        results["avg"].append(per_target_mse_avg(train_data, test_data))
        results["reg"].append(per_target_mse_reg(train_data, test_data))
        results["repeat"].append(per_target_mse_repeat(test_data))

    print(f"\nAll folds complete: {time.time() - t_start:.1f}s total\n")

    summary_rows = []
    for name, mse_list in results.items():
        arr = np.stack(mse_list)
        all_mse = arr.mean(axis=1)
        tar_mse = arr[:, target_idx]
        summary_rows.append({
            "model": name,
            "params": n_params.get(name, np.nan),
            "MSE_mean":        all_mse.mean(),
            "MSE_std":         all_mse.std(),
            "MSE_target_mean": tar_mse.mean(),
            "MSE_target_std":  tar_mse.std(),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, "summary_10folds.csv"),
                  index=False, float_format="%.5f")

    lines = []
    lines.append("=== Baselines on PBC2 controlled simulation (10-fold CV) ===")
    lines.append(f"EPOCHS={EPOCHS}  random_state={CV_RANDOM_STATE}  "
                 f"seeds (one per fold)={SEEDS}")
    lines.append(f"Synthetic target marginal P(y=1) = {flat_y.mean():.4f}")
    lines.append("")
    lines.append(f"{'Model':<22s} {'Params':>8s}    "
                 f"{'MSE (10 folds)':>22s}     {'MSE_target (10 folds)':>24s}")
    for r in summary_rows:
        params = "" if pd.isna(r["params"]) else f"{int(r['params'])}"
        lines.append(
            f"{r['model']:<22s} {params:>8s}    "
            f"{r['MSE_mean']:.4f} +/- {r['MSE_std']:.4f}     "
            f"{r['MSE_target_mean']:.4f} +/- {r['MSE_target_std']:.4f}"
        )
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
