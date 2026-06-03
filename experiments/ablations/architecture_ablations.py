"""§3.3 cumulant-head ablations (response to reviewer §3.3 / §3.4).

Compares MiniTransformer against two Eq. 3 ablations:
  - HorizonDecayOff : cumulant pooling kept, horizon-decay factor set to 1
                      (uniform causal pooling). Isolates the horizon-decay term.
  - CumulantOff     : cumulant pooling removed; readout uses only the
                      per-timestep Eq. 2 output at t = T (reviewer's literal
                      request).

Protocols match the paper:
  Simulation: 10 seeds, n_train=200, n_test=1000 (matches
              run_baselines_simulation.py).
  LORA D1, LORA D2, PBC2: 10-fold CV (KFold n_splits=10, random_state=42),
              one paper-seed per fold (matches run_baselines_real_data.py).

Reports per-dataset MSE_target (mean +/- std) for all three model variants.

Run with:
    python notebooks/architecture_ablations.py
Optional env vars:
    ABL_SIM_EPOCHS  (default 100)   epochs for the simulation
    ABL_REAL_EPOCHS (default 150)   epochs for the real cohorts
    ABL_DATASETS    (default "sim,ghq_b_sum,ghq_sum,pbc2")  which datasets to run

Outputs (notebooks/results/architecture_ablations/):
    mse_<dataset>.csv   per-variant MSE_target / MSE_mean
    summary.txt         human-readable summary across datasets
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
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from src.data_preparation import SimulatedDataset, collate_function, load_real_data
from src.transformers import (
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
)
from src.baselines.cumulant_ablations import CumulantAblationMiniTransformer


device = torch.device("cpu")

SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]

# Simulation config (paper Table 1, n_train=200)
SIM_N_TRAIN = 200
SIM_N_TEST = 1000
SIM_P = 10
SIM_MAXLEN = 10
SIM_PREDIDX = 2
SIM_NHEADS, SIM_NCUM, SIM_DK, SIM_DV = 12, 2, 1, 1
SIM_EPOCHS = int(os.environ.get("ABL_SIM_EPOCHS", 100))

# Real-data config (paper §3.2)
REAL_NHEADS, REAL_NCUM, REAL_DK, REAL_DV = 8, 8, 1, 1
REAL_MAXLEN = 10
REAL_EPOCHS = int(os.environ.get("ABL_REAL_EPOCHS", 150))
REAL_PREDIDX = {"ghq_b_sum": 9, "ghq_sum": 9, "pbc2": 9, "pbc2_sim": 9}

BATCH_SIZE = 1
LR = 1e-3
LAMBDA_L2 = 1e-3
N_SPLITS = 10
CV_RANDOM_STATE = 42

MODES = ["full", "pairwise_off", "horizon_off", "cumulant_off"]
MODE_LABEL = {
    "full": "MiniTransformer",
    "pairwise_off": "PairwiseDecayOff",
    "horizon_off": "HorizonDecayOff",
    "cumulant_off": "CumulantOff",
}

DATASETS = os.environ.get("ABL_DATASETS", "sim,ghq_b_sum,ghq_sum,pbc2_sim").split(",")

OUT_DIR = "notebooks/results/architecture_ablations"
os.makedirs(OUT_DIR, exist_ok=True)


def per_target_mse(model, test_data):
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


def train_and_eval(model, train_data, test_data, epochs):
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_function, num_workers=0)
    # NOTE: eval_loader is intentionally None here. The cumulant ablations are
    # NEW in the revision (no published reference to reproduce), so they only
    # need internal consistency among MiniTransformer / HorizonDecayOff /
    # CumulantOff, which a fixed protocol provides. We do NOT pass an eval loader
    # because the per-epoch eval-loss readout in train_mini_transformer assumes
    # the standard MiniTransformer output shape and crashes on the ablation
    # variants (CumulantAblationMiniTransformer). All three variants use the same
    # None convention, so the comparison remains fair.
    opt = optim.Adam(model.parameters(), lr=LR)
    train_mini_transformer(model, loader, None, opt, LAMBDA_L2, epochs, device)
    return per_target_mse(model, test_data)


def make_model(mode, p, nheads, dk, dv, ncum, masks):
    mp, pdm, dte = masks
    return CumulantAblationMiniTransformer(
        p, nheads, dk, dv, ncum, mp, pdm, dte, device, cumulant_mode=mode,
    ).to(device)


def run_simulation():
    print("\n=== Simulation: 10 seeds, cumulant ablations ===")
    masks = (create_custom_mask_pair(SIM_MAXLEN, device),
             create_pairwise_distance_matrix(SIM_MAXLEN, device),
             create_distance_to_end_matrix(SIM_MAXLEN, device))
    results = {m: [] for m in MODES}
    for k, seed in enumerate(SEEDS):
        torch.manual_seed(seed); np.random.seed(seed)
        train_ds = SimulatedDataset(SIM_N_TRAIN, SIM_P, maxlen=SIM_MAXLEN, device=device).data
        test_ds = SimulatedDataset(SIM_N_TEST, SIM_P, maxlen=SIM_MAXLEN, device=device).data
        for mode in MODES:
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(mode, SIM_P, SIM_NHEADS, SIM_DK, SIM_DV, SIM_NCUM, masks)
            mse = train_and_eval(model, train_ds, test_ds, SIM_EPOCHS)
            results[mode].append(mse)
        print(f"  seed {seed} ({k+1}/{len(SEEDS)}): "
              + "  ".join(f"{MODE_LABEL[m]}={results[m][-1][SIM_PREDIDX]:.4f}" for m in MODES))
    return _summarise(results, SIM_PREDIDX, "sim")


def run_real(data_str):
    predindex = REAL_PREDIDX[data_str]
    print(f"\n=== {data_str}: 10-fold CV, cumulant ablations ===")
    load_str = "pbc2" if data_str == "pbc2_sim" else data_str
    tensors, _ = load_real_data(load_str)
    # Truncate each sequence to the last REAL_MAXLEN visits, matching the
    # convention in run_baselines_real_data.py and the v_monotonicity_check_*
    # pipeline (the precomputed positional matrices are sized REAL_MAXLEN).
    tensors = [s if s.shape[0] <= REAL_MAXLEN else s[-REAL_MAXLEN:] for s in tensors]
    if data_str == "pbc2_sim":
        # Overwrite the ascites column with the synthetic j1 -> j2 -> j3 target,
        # matching notebooks/pbc2_controlled_simulation.py (seed=42 to keep the
        # injection deterministic across runs that use this substrate).
        from src.pbc2_substrate import inject_synthetic_target
        tensors, _z, _y = inject_synthetic_target(tensors, seed=42)
    n_total = len(tensors)
    p = tensors[0].shape[1]
    masks = (create_custom_mask_pair(REAL_MAXLEN, device),
             create_pairwise_distance_matrix(REAL_MAXLEN, device),
             create_distance_to_end_matrix(REAL_MAXLEN, device))
    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                       random_state=CV_RANDOM_STATE).split(np.arange(n_total)))
    assert len(folds) == len(SEEDS)
    results = {m: [] for m in MODES}
    for f_idx, ((tr, te), seed) in enumerate(zip(folds, SEEDS)):
        train_ds = [tensors[i] for i in tr]
        test_ds = [tensors[i] for i in te]
        for mode in MODES:
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(mode, p, REAL_NHEADS, REAL_DK, REAL_DV, REAL_NCUM, masks)
            mse = train_and_eval(model, train_ds, test_ds, REAL_EPOCHS)
            results[mode].append(mse)
        print(f"  fold {f_idx+1}/{N_SPLITS} (seed {seed}): "
              + "  ".join(f"{MODE_LABEL[m]}={results[m][-1][predindex]:.4f}" for m in MODES))
    return _summarise(results, predindex, data_str)


def _summarise(results, predindex, tag):
    rows = []
    for mode in MODES:
        arr = np.stack(results[mode])  # (n, p)
        rows.append({
            "model": MODE_LABEL[mode],
            "MSE_target_mean": float(arr[:, predindex].mean()),
            "MSE_target_std": float(arr[:, predindex].std()),
            "MSE_mean_mean": float(arr.mean(axis=1).mean()),
            "MSE_mean_std": float(arr.mean(axis=1).std()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, f"mse_{tag}.csv"), index=False, float_format="%.5f")
    print(df.to_string(index=False))
    return df


def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    t0 = time.time()
    out = {}
    if "sim" in DATASETS:
        out["sim"] = run_simulation()
    for ds in DATASETS:
        if ds == "sim":
            continue
        out[ds] = run_real(ds)

    lines = ["=== Cumulant-head ablations (response §3.3) ===",
             f"Seeds: {SEEDS}", ""]
    for tag, df in out.items():
        lines.append(f"--- {tag} (MSE_target mean +/- std) ---")
        for _, r in df.iterrows():
            lines.append(f"  {r['model']:<16s} {r['MSE_target_mean']:.4f} "
                         f"+/- {r['MSE_target_std']:.4f}")
        lines.append("")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"Total elapsed: {time.time() - t0:.1f}s\nResults in {OUT_DIR}/")


if __name__ == "__main__":
    main()
