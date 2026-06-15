"""§3.1 statistical-testing evaluation on the simulation (paper §2.3 / Section 3.1).

Trains the MiniTransformer on n_train=100 sequences (the most challenging
setting) and runs the permutation test of Section 2.3 on the trained model.

Protocol matches run_baselines_simulation.py:
  - Same seed (42), same data-generation order
  - RNG state saved after data generation; model initialised from that state
  - Real eval_loader passed to train_mini_transformer (consistent RNG advance)
  - nrepp=10 permutation repetitions, V=8 visit samples (paper §2.3)

Outputs:
  notebooks/results/simulation_statistical_testing/pvalues.csv
  notebooks/results/simulation_statistical_testing/summary.txt
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

from src.data_preparation import SimulatedDataset, collate_function
from src.transformers import (
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
)
from src.statistical_testing import statistical_testing

device = torch.device("cpu")

SEED       = int(os.environ.get("ST_SEED", 42))
N_TRAIN    = int(os.environ.get("ST_N_TRAIN", 100))
N_TEST     = int(os.environ.get("ST_N_TEST",  1000))
P          = 10
MAXLEN     = 10
PREDINDEX  = 2          # j3
NHEADS     = 12
NCUM       = 2
DK         = 1
DV         = 1
EPOCHS     = int(os.environ.get("ST_EPOCHS", 100))
BATCH_SIZE = 1
LR         = 1e-3
LAMBDA_L2  = 1e-3
NREPP      = int(os.environ.get("ST_NREPP", 10))   # permutation repetitions
V          = int(os.environ.get("ST_V", 8))         # visit samples

OUT_DIR = "notebooks/results/simulation_statistical_testing"
os.makedirs(OUT_DIR, exist_ok=True)


def run():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    masks = (
        create_custom_mask_pair(MAXLEN, device),
        create_pairwise_distance_matrix(MAXLEN, device),
        create_distance_to_end_matrix(MAXLEN, device),
    )
    mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix = masks

    train_ds = SimulatedDataset(N_TRAIN, P, maxlen=MAXLEN, device=device).data
    eval_ds  = SimulatedDataset(N_TEST,  P, maxlen=MAXLEN, device=device).data

    # Save RNG state after data generation; restore before model init so that
    # training trajectory does not depend on data-generation order.
    rng_state    = torch.get_rng_state()
    np_rng_state = np.random.get_state()
    torch.set_rng_state(rng_state)
    np.random.set_state(np_rng_state)

    model = MiniTransformer(
        P, NHEADS, DK, DV, NCUM,
        mask_pairwise, pairwise_distance_matrix,
        distance_to_end_matrix, device,
    ).to(device)

    loader      = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=collate_function, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=len(eval_ds), shuffle=False,
                             collate_fn=collate_function, num_workers=0)
    optimizer   = optim.Adam(model.parameters(), lr=LR)

    train_mini_transformer(model, loader, eval_loader, optimizer,
                           LAMBDA_L2, EPOCHS, device)

    avepval, stdpval, _ctx, _tgt = statistical_testing(
        model, train_ds, p=P, predindex=PREDINDEX,
        nrepp=NREPP, target_sample_size=V,
    )
    avepval = np.array(avepval.numpy() if hasattr(avepval, "numpy") else avepval)
    stdpval = np.array(stdpval.numpy() if hasattr(stdpval, "numpy") else stdpval)

    df = pd.DataFrame({
        "variable":  [f"v{j}" for j in range(P)],
        "is_signal": [j in (0, 1, 2) for j in range(P)],
        "mean_pval": avepval,
        "std_pval":  stdpval,
    })
    df.to_csv(os.path.join(OUT_DIR, "pvalues.csv"), index=False, float_format="%.5f")

    lines = [
        "=== Simulation statistical testing (Section 2.3) ===",
        f"Seed: {SEED}   n_train: {N_TRAIN}   n_test: {N_TEST}   p: {P}",
        f"nrepp: {NREPP}   V: {V}   epochs: {EPOCHS}",
        "",
        f"{'Variable':<10} {'Signal':<8} {'mean p-val':<12} {'std'}",
    ]
    for j in range(P):
        sig = "yes" if j in (0, 1, 2) else "no"
        lines.append(f"v{j:<9} {sig:<8} {avepval[j]:.4f}       ± {stdpval[j]:.4f}")
    lines += [
        "",
        f"Signal variables (j1=v0, j2=v1, j3=v2): "
        f"{avepval[0]:.4f}, {avepval[1]:.4f}, {avepval[2]:.4f}",
        f"Null range (v3..v9): [{avepval[3:].min():.4f}, {avepval[3:].max():.4f}]",
        f"Total elapsed: {time.time() - t0:.1f}s",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nResults saved to {OUT_DIR}/")
    return df


if __name__ == "__main__":
    run()
