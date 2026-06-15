"""Statistical testing on the LORA real-data cohorts (Section 3.2 / Table 4).

Trains the MiniTransformer on the full dataset (no CV split -- the same
design as the original notebooks, which test population-level context effects
on the best-available fitted model) and runs the permutation test of
Section 2.3.

Protocol matches run_baselines_real_data.py:
  - Real eval_loader passed to train_mini_transformer (consistent RNG advance)
  - Same architecture: nheads=8, ncum=8, batch_size=2, 150 epochs
  - Same seed: 12345
  - nrepp=10, V=8 (matching original notebooks / Table 4)

Usage:
    python experiments/statistical_testing/real_data_statistical_testing.py ghq_b_sum
    python experiments/statistical_testing/real_data_statistical_testing.py ghq_sum

Outputs:
    notebooks/results/statistical_testing_real_data/<DATA_STR>/pvalues.csv
    notebooks/results/statistical_testing_real_data/<DATA_STR>/summary.txt
    minitransformer_paper/figures/context_target_effect_<DATA_STR>.png
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

from src.data_preparation import collate_function, load_real_data
from src.transformers import (
    MiniTransformer,
    count_parameters,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
)
from src.statistical_testing import (
    statistical_testing,
    get_context_predindex_pair_effect,
    plot_context_predindex_pair_effect,
)

device = torch.device("cpu")

DATA_STR   = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ST_DATA", "ghq_b_sum")
SEED       = int(os.environ.get("ST_SEED", 12345))
NHEADS     = 8
NCUM       = 8
DK         = 1
DV         = 1
MAXLEN     = 10
BATCH_SIZE = 2
LR         = 1e-3
LAMBDA_L2  = 1e-3
EPOCHS     = int(os.environ.get("ST_EPOCHS", 150))
PREDINDEX  = 9
NREPP      = int(os.environ.get("ST_NREPP", 10))
V          = int(os.environ.get("ST_V", 8))

VAR_NAMES = {
    "ghq_b_sum": [
        "dh_10", "dh_35", "dh_37", "dh_38", "dh_45",
        "dh_53", "le_8",  "le_17", "le_22", "ghq_b_sum",
    ],
    "ghq_sum": [
        "dh_11", "dh_31", "dh_37", "dh_38", "dh_42",
        "dh_46", "le_1",  "le_16", "le_17", "ghq_sum",
    ],
}

OUT_DIR = f"notebooks/results/statistical_testing_real_data/{DATA_STR}"
os.makedirs(OUT_DIR, exist_ok=True)


def run():
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data, _ = load_real_data(DATA_STR)
    p = data[0].shape[1]
    print(f"=== Real-data statistical testing: {DATA_STR} ===")
    print(f"n={len(data)}  p={p}  seed={SEED}  nrepp={NREPP}  V={V}  epochs={EPOCHS}\n")

    def truncate(seq):
        return seq if seq.shape[0] <= MAXLEN else seq[-MAXLEN:]
    data = [truncate(s) for s in data]

    mask_pairwise           = create_custom_mask_pair(MAXLEN, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(MAXLEN, device)
    distance_to_end_matrix  = create_distance_to_end_matrix(MAXLEN, device)

    model = MiniTransformer(
        p, NHEADS, DK, DV, NCUM,
        mask_pairwise, pairwise_distance_matrix,
        distance_to_end_matrix, device,
    ).to(device)
    print(f"Parameters: {count_parameters(model)}")

    loader = DataLoader(
        data, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_function, num_workers=0,
    )
    eval_loader = DataLoader(
        data, batch_size=len(data), shuffle=False,
        collate_fn=collate_function, num_workers=0,
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_mini_transformer(model, loader, eval_loader, optimizer,
                           LAMBDA_L2, EPOCHS, device)

    print(f"\nRunning statistical_testing (nrepp={NREPP}, V={V}) ...")
    avepval, stdpval, ctx, tgt = statistical_testing(
        model, data, p=p, predindex=PREDINDEX,
        nrepp=NREPP, target_sample_size=V,
    )
    avepval = np.array(avepval.numpy() if hasattr(avepval, "numpy") else avepval)
    stdpval = np.array(stdpval.numpy() if hasattr(stdpval, "numpy") else stdpval)

    names = VAR_NAMES.get(DATA_STR, [f"v{j}" for j in range(p)])

    df = pd.DataFrame({
        "rank":     np.argsort(avepval) + 1,
        "variable": names,
        "mean_pval": avepval,
        "std_pval":  stdpval,
    }).sort_values("mean_pval").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df.to_csv(os.path.join(OUT_DIR, "pvalues.csv"), index=False, float_format="%.5f")

    # Legacy format matching paper_results .txt files
    legacy_lines = [f"For combination sample size: {V} "]
    for j in range(p):
        legacy_lines.append(f"Variable {j+1}: {avepval[j]:.4f} ± {stdpval[j]:.2f}")
    legacy_txt = "\n".join(legacy_lines)

    lines = [
        f"=== Real-data statistical testing: {DATA_STR} ===",
        f"n={len(data)}  p={p}  seed={SEED}  nrepp={NREPP}  V={V}  epochs={EPOCHS}",
        "",
        f"{'Rank':<6} {'Variable':<14} {'mean p-val':<12} {'std'}",
    ]
    for _, row in df.iterrows():
        lines.append(f"{int(row['rank']):<6} {row['variable']:<14} {row['mean_pval']:.4f}       ± {row['std_pval']:.4f}")
    lines += [
        "",
        legacy_txt,
        f"\nTotal elapsed: {time.time() - t0:.1f}s",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nResults saved to {OUT_DIR}/")

    # Heatmap of context-target effect matrix S
    cpe = get_context_predindex_pair_effect(model, p, ctx, tgt)
    fig_dir = "minitransformer_paper/figures"
    plot_context_predindex_pair_effect(cpe, DATA_STR, fig_dir)
    print(f"Heatmap saved to {fig_dir}/context_target_effect_{DATA_STR}.png")

    return df


if __name__ == "__main__":
    run()
