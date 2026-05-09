"""§3.1 baseline comparison on the simulation, paper-style:
10 seeds, mean ± std reporting, matching the convention of
notebooks/simulation_experiments.ipynb (Table 1 of the paper).

Models compared (all five neural models trained on each seed):
- MiniTransformer (reference, paper §3.1 hyperparameters)
- KernelAttentionNoDecay (MiniTransformer minus Eq. 1 decay; also covers §3.3)
- ScaledVanillaTransformer (1 layer, 1 head, parameter-matched)
- iTransformer (variable-axis attention, parameter-matched)
- DLinear (Zeng et al. 2023)

Plus the non-neural baselines:
- Marginal mean per target
- Per-target Gaussian regression on t-1 features

For each seed we generate a fresh (train, eval) pair (n_train and n_test fixed),
train all five neural models on the same training set, and record per-target
MSE on the eval set. After all seeds are processed we report mean ± std for
both MSE (averaged over all variables) and MSE_target (predindex=j3=2).
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

from src.data_preparation import SimulatedDataset, collate_function
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


device = torch.device("cpu")

# Paper's seed list (matches simulation_experiments.ipynb)
SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]

# Simulation config (matches paper §3.1 / Table 1, n_train=200 column)
n_train = int(os.environ.get("BSL_N_TRAIN", 200))
n_test  = int(os.environ.get("BSL_N_TEST", 1000))
p = 10
maxlen = 10
predindex = 2  # j3 in the paper

# MiniTransformer hyperparameters (paper §3.1)
mt_nheads = 12
mt_ncum   = 2
mt_dk     = 1
mt_dv     = 1

# Training schedule (paper §3.1)
batch_size    = 1
learning_rate = 1e-3
lambda_l2     = 1e-3
EPOCHS        = int(os.environ.get("BSL_EPOCHS", 100))

OUT_DIR = "notebooks/results/baselines_simulation"
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


def per_target_mse_torch_model(model, test_data):
    model.eval()
    p_local = test_data[0].shape[1]
    se, n_eval = np.zeros(p_local), 0
    with torch.no_grad():
        for seq in test_data:
            if seq.shape[0] < 3: continue
            x = seq.unsqueeze(0)
            mask = torch.ones_like(x, dtype=torch.bool)
            pred = model((x[:, :-1, :], mask[:, :-1, :]))
            se += (pred[0, -1].numpy() - seq[-1].numpy()) ** 2
            n_eval += 1
    return se / max(n_eval, 1)


def train_and_eval(model, train_dataset, test_dataset):
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_function, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    train_mini_transformer(model, loader, None, opt, lambda_l2, EPOCHS, device)
    return per_target_mse_torch_model(model, test_dataset)


def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    print(f"=== §3.1 baseline comparison on simulation, 10-seed paper-style ===")
    print(f"n_train={n_train}, n_test={n_test}, p={p}, EPOCHS={EPOCHS}")
    print(f"seeds={SEEDS}\n")

    mask_pairwise = create_custom_mask_pair(maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(maxlen, device)

    # Determine matched param counts once (from a representative MiniTransformer
    # so they don't depend on the seed)
    torch.manual_seed(0)
    mt0 = MiniTransformer(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                          mask_pairwise, pairwise_distance_matrix,
                          distance_to_end_matrix, device).to(device)
    n_params_mt = count_parameters(mt0)
    d_model_svt, n_params_svt = find_matched_d_model(p, n_params_mt, max_len=maxlen)
    d_model_it,  n_params_it  = find_itransformer_d_model(p, n_params_mt, history_len=maxlen)
    print(f"Param targets:  MT={n_params_mt}  SVT={n_params_svt} (d_model={d_model_svt})  "
          f"iTr={n_params_it} (d_model={d_model_it})\n")

    # storage: model_name -> (n_seeds, p) array of per-target MSEs
    results = {k: [] for k in [
        "MiniTransformer", "NoDecay", "ScaledVanillaTr",
        "iTransformer",    "DLinear", "avg", "reg",
    ]}

    # Build one DLinear / NoDecay to get their param counts (data-independent)
    nd0 = KernelAttentionNoDecay(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                                 mask_pairwise, pairwise_distance_matrix,
                                 distance_to_end_matrix, device).to(device)
    n_params_nd = sum(par.numel() for par in nd0.parameters() if par.requires_grad)
    dl0 = DLinear(p, history_len=maxlen, kernel_size=5, max_len=maxlen, device=device)
    n_params_dl = sum(par.numel() for par in dl0.parameters() if par.requires_grad)
    n_params = {"MiniTransformer": n_params_mt, "NoDecay": n_params_nd,
                "ScaledVanillaTr": n_params_svt, "iTransformer": n_params_it,
                "DLinear": n_params_dl}

    t_start = time.time()
    for k, seed in enumerate(SEEDS):
        print(f"--- seed {seed}  ({k+1}/{len(SEEDS)}) ---")
        # Paper's RNG order: seed -> train data -> test data -> model init
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_ds = SimulatedDataset(n_train, p, maxlen=maxlen, device=device).data
        test_ds  = SimulatedDataset(n_test,  p, maxlen=maxlen, device=device).data

        # --- MiniTransformer ---
        mt = MiniTransformer(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                             mask_pairwise, pairwise_distance_matrix,
                             distance_to_end_matrix, device).to(device)
        mse = train_and_eval(mt, train_ds, test_ds)
        results["MiniTransformer"].append(mse)
        print(f"  MT          MSE={mse.mean():.4f}  MSE_target={mse[predindex]:.4f}")

        # --- KernelAttentionNoDecay ---
        torch.manual_seed(seed); np.random.seed(seed)
        nd = KernelAttentionNoDecay(p, mt_nheads, mt_dk, mt_dv, mt_ncum,
                                    mask_pairwise, pairwise_distance_matrix,
                                    distance_to_end_matrix, device).to(device)
        mse = train_and_eval(nd, train_ds, test_ds)
        results["NoDecay"].append(mse)
        print(f"  NoDecay     MSE={mse.mean():.4f}  MSE_target={mse[predindex]:.4f}")

        # --- ScaledVanillaTransformer ---
        torch.manual_seed(seed); np.random.seed(seed)
        svt = ScaledVanillaTransformer(p, d_model=d_model_svt, max_len=maxlen, device=device)
        mse = train_and_eval(svt, train_ds, test_ds)
        results["ScaledVanillaTr"].append(mse)
        print(f"  SVT         MSE={mse.mean():.4f}  MSE_target={mse[predindex]:.4f}")

        # --- iTransformer ---
        torch.manual_seed(seed); np.random.seed(seed)
        it = ITransformer(p, d_model=d_model_it, n_heads=1,
                          history_len=maxlen, max_len=maxlen, device=device)
        mse = train_and_eval(it, train_ds, test_ds)
        results["iTransformer"].append(mse)
        print(f"  iTr         MSE={mse.mean():.4f}  MSE_target={mse[predindex]:.4f}")

        # --- DLinear ---
        torch.manual_seed(seed); np.random.seed(seed)
        dl = DLinear(p, history_len=maxlen, kernel_size=5, max_len=maxlen, device=device)
        mse = train_and_eval(dl, train_ds, test_ds)
        results["DLinear"].append(mse)
        print(f"  DLinear     MSE={mse.mean():.4f}  MSE_target={mse[predindex]:.4f}")

        # --- non-neural baselines ---
        results["avg"].append(per_target_mse_avg(train_ds, test_ds))
        results["reg"].append(per_target_mse_reg(train_ds, test_ds))

    print(f"\nAll seeds complete: {time.time() - t_start:.1f}s total\n")

    # Aggregate
    summary_rows = []
    for name, mse_list in results.items():
        arr = np.stack(mse_list)              # (n_seeds, p)
        all_mse  = arr.mean(axis=1)           # (n_seeds,) avg over variables
        tar_mse  = arr[:, predindex]          # (n_seeds,) target only
        summary_rows.append({
            "model": name,
            "params": n_params.get(name, np.nan),
            "MSE_mean":         all_mse.mean(),
            "MSE_std":          all_mse.std(),
            "MSE_target_mean":  tar_mse.mean(),
            "MSE_target_std":   tar_mse.std(),
        })
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(os.path.join(OUT_DIR, "summary_10seeds.csv"),
                  index=False, float_format="%.5f")

    # Pretty print + paper-style summary
    lines = []
    lines.append(f"=== §3.1 baseline comparison on simulation ===")
    lines.append(f"n_train={n_train}  EPOCHS={EPOCHS}  seeds={SEEDS}")
    lines.append("")
    lines.append(f"{'Model':<22s} {'Params':>8s}    "
                 f"{'MSE (10 seeds)':>22s}     {'MSE_target (10 seeds)':>24s}")
    for r in summary_rows:
        params = ("" if pd.isna(r["params"]) else f"{int(r['params'])}")
        lines.append(
            f"{r['model']:<22s} {params:>8s}    "
            f"{r['MSE_mean']:.3f} ± {r['MSE_std']:.3f}            "
            f"{r['MSE_target_mean']:.3f} ± {r['MSE_target_std']:.3f}"
        )
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT_DIR, "summary_10seeds.txt"), "w") as f:
        f.write(summary + "\n")

    # Persist raw per-seed per-target arrays for downstream use
    np.savez(os.path.join(OUT_DIR, "per_seed_arrays.npz"),
             **{k: np.stack(v) for k, v in results.items()})
    print(f"\nSaved: summary_10seeds.csv, summary_10seeds.txt, per_seed_arrays.npz")
    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
