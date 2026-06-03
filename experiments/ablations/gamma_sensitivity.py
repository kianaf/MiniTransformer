"""Minor #3 (gamma sensitivity): sweep the shape parameter of the temporal-decay
kernel exp(-(w*|dt|)^gamma) over {1, 2, 5, 10, learned} and compare predictive
performance plus, on the simulation, the §2.3 test outcome.

- gamma is hard-coded as **5 in src/transformers.MultiHeadAttention.exponential_decay_*;
  we subclass to make it a constructor arg, optionally a learned scalar.
- Protocols match the paper:
    Simulation: 10 seeds (paper-style), n_train/n_test/p/maxlen as in
    notebooks/run_baselines_simulation.py.
    LORA D1, LORA D2: 10-fold CV (KFold n_splits=10, shuffle=True,
    random_state=42), one paper-seed per fold, as in
    notebooks/run_baselines_real_data.py.
- For the simulation we additionally run the §2.3 permutation test once per
  gamma using a fixed test seed; we report mean p-value per variable so the
  ranking can be compared across gamma. Ground truth: j1=0, j2=1, j3=predindex=2
  are signal; 3..9 are null.

Outputs:
- notebooks/results/gamma_sensitivity/mse_simulation.csv
- notebooks/results/gamma_sensitivity/mse_lora_d1.csv
- notebooks/results/gamma_sensitivity/mse_lora_d2.csv
- notebooks/results/gamma_sensitivity/test_simulation.csv     # §2.3 p-values
- notebooks/results/gamma_sensitivity/summary.txt              # plain-text summary
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

from src.data_preparation import SimulatedDataset, collate_function, load_real_data
from src.transformers import (
    MultiHeadAttention,
    MiniTransformer,
    create_custom_mask_pair,
    create_distance_to_end_matrix,
    create_pairwise_distance_matrix,
    train_mini_transformer,
)
from src.evaluation import calculate_regression_loss  # noqa: F401  (parity import)
from src.statistical_testing import statistical_testing


device = torch.device("cpu")

# Gamma values to sweep. "learned" trains gamma as a free scalar parameter.
GAMMAS = [1.0, 2.0, 5.0, 10.0, "learned"]

# Paper seed list (matches simulation_experiments.ipynb / real_data_experiments_*.ipynb)
SEEDS = [0, 1, 11, 42, 123, 999, 1337, 2025, 9999, 12345]

# Simulation config (paper §3.1, n_train=200 column of Table 1)
SIM_N_TRAIN = int(os.environ.get("GS_SIM_N_TRAIN", 200))
SIM_N_TEST  = int(os.environ.get("GS_SIM_N_TEST", 1000))
SIM_P       = 10
SIM_MAXLEN  = 10
SIM_PREDIDX = 2  # j3

# Simulation MiniTransformer hyperparameters (paper §3.1)
SIM_NHEADS = 12
SIM_NCUM   = 2
SIM_DK     = 1
SIM_DV     = 1

# Real-data MiniTransformer hyperparameters (paper §3.2)
REAL_NHEADS = 8
REAL_NCUM   = 8
REAL_DK     = 1
REAL_DV     = 1
REAL_MAXLEN = 10  # LORA D1/D2 max sequence length

# Training schedule
BATCH_SIZE    = 1
LR            = 1e-3
LAMBDA_L2     = 1e-3
SIM_EPOCHS    = int(os.environ.get("GS_SIM_EPOCHS", 100))
REAL_EPOCHS   = int(os.environ.get("GS_REAL_EPOCHS", 150))

# §2.3 test config on the simulation (matches notebooks/null_calibration.py)
TEST_NREPP  = int(os.environ.get("GS_TEST_NREPP", 500))
TEST_V      = int(os.environ.get("GS_TEST_V", 7))
TEST_SEED   = 123456789

# 10-fold CV config
N_SPLITS = 10
CV_RANDOM_STATE = 42

OUT_DIR = "notebooks/results/gamma_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Gamma-parameterised attention / model                                       #
# --------------------------------------------------------------------------- #

class GammaMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with gamma made explicit. Defaults to gamma=5 (the
    paper's hard-coded value). When `learned=True`, gamma becomes a learnable
    scalar initialised at `gamma_init`.

    The pairwise- and horizon-decay weights are re-initialised on a smaller
    range when gamma is large, so the kernel exponent (dist * exp(w))^gamma
    does not explode to inf at step 0. The original code's init range [-1, 1]
    is fine at gamma=5 (the paper's choice) but produces inf gradients at
    gamma=10 where (9 * exp(1))^10 ~ 1e14 saturates the additive logit.
    """
    def __init__(self, *args, gamma=5.0, learned=False, gamma_init=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        if learned:
            # Softplus-parameterised so gamma stays positive
            inv = float(np.log(np.expm1(gamma_init)))
            self._gamma_raw = nn.Parameter(torch.tensor(inv, dtype=torch.float32))
            self._learned = True
            gamma_eff = float(gamma_init)
        else:
            self.register_buffer("_gamma_fixed", torch.tensor(float(gamma)))
            self._learned = False
            gamma_eff = float(gamma)
        # Rescale the weight init so (max_dist * exp(w_init))^gamma_eff stays
        # well below the _LOGIT_MAX clamp at step 0. For gamma <= 5 keep the
        # paper's init range; otherwise shrink.
        if gamma_eff > 5.0:
            # Aim for max exponent value around exp(2) at step 0: i.e.
            # (max_dist * exp(w))^gamma ~ 2 -> exp(w) ~ 2^(1/gamma) / max_dist
            max_dist = 10.0
            target = (2.0 ** (1.0 / gamma_eff)) / max_dist
            w_init = float(np.log(target))
            nn.init.uniform_(self.distance_between_two_positions_weight,
                             w_init - 0.1, w_init + 0.1)
            nn.init.uniform_(self.distance_to_end_weight,
                             w_init - 0.1, w_init + 0.1)

    @property
    def gamma(self):
        if self._learned:
            return nn.functional.softplus(self._gamma_raw)
        return self._gamma_fixed

    # The paper's kernel is exp(-(w*|dt|)^gamma). The original code hard-codes
    # gamma=5 (odd integer) and writes (-dist*exp(weight))**5 so the sign comes
    # out negative. For arbitrary gamma we use the sign-safe form: take the
    # non-negative magnitude to the power of gamma and negate explicitly. We
    # clamp the exponent value itself (not the base) so the additive logit
    # stays bounded for any gamma, avoiding inf/NaN gradients when gamma is
    # large and `weight` drifts up during training. exp(-50) is already deep
    # underflow, so this clamp does not affect the kernel meaningfully.
    _LOGIT_MAX = 50.0
    _BASE_EPS = 1e-6

    def _exponent(self, dist, weight):
        # Clamp the BASE (not the result) before raising to gamma, so the
        # gradient does not flow through inf**(gamma-1) at huge dist values
        # (the distance_to_end_matrix uses 1e9 as a "beyond" sentinel).
        # The cap is chosen so base**gamma <= _LOGIT_MAX; the saturated region
        # is exp(-LOGIT_MAX) ~ 0 anyway, so the kernel value is unchanged in
        # the regime where the clamp engages.
        b_max = self._LOGIT_MAX ** (1.0 / float(
            self.gamma if not self._learned else self.gamma.detach()))
        base = (dist * torch.exp(weight)).clamp(max=b_max) + self._BASE_EPS
        return base ** self.gamma

    def exponential_decay_pred(self, dist, weight):
        return torch.exp(-self._exponent(dist, weight))

    def exponential_decay_pair(self, dist, weight):
        return -self._exponent(dist, weight)


class GammaMiniTransformer(MiniTransformer):
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise,
                 pairwise_distance_matrix, distance_to_end_matrix, device,
                 gamma=5.0, learned_gamma=False):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.device = device
        self.multiheadattn = GammaMultiHeadAttention(
            d_model, num_heads, dk, dv, ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
            gamma=gamma, learned=learned_gamma,
        )
        self.prediction = nn.Linear(self.ncum, self.d_model)


def make_model(gamma_spec, p, nheads, dk, dv, ncum,
               mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix):
    """gamma_spec: float or the string 'learned'."""
    if gamma_spec == "learned":
        return GammaMiniTransformer(
            p, nheads, dk, dv, ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
            gamma=5.0, learned_gamma=True,
        ).to(device)
    return GammaMiniTransformer(
        p, nheads, dk, dv, ncum,
        mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
        gamma=float(gamma_spec), learned_gamma=False,
    ).to(device)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

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


def train_and_eval(model, train_data, test_data, epochs):
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_function, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=LR)
    train_mini_transformer(model, loader, None, opt, LAMBDA_L2, epochs, device)
    return per_target_mse_torch_model(model, test_data)


def gamma_label(g):
    return "learned" if g == "learned" else f"{g:g}"


# --------------------------------------------------------------------------- #
# Simulation sweep                                                            #
# --------------------------------------------------------------------------- #

def run_simulation():
    print("\n=== Simulation: 10 seeds, gamma sweep ===")
    mask_pairwise = create_custom_mask_pair(SIM_MAXLEN, device)
    distance_to_end_matrix = create_distance_to_end_matrix(SIM_MAXLEN, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(SIM_MAXLEN, device)

    # per-gamma: list of per-target MSE arrays (one per seed)
    mse = {gamma_label(g): [] for g in GAMMAS}
    # §2.3 test: per gamma -> (mean p-value vector over variables, single run with TEST_SEED)
    pvals = {gamma_label(g): None for g in GAMMAS}
    # learned gamma final values per seed (only for the 'learned' condition)
    learned_gammas = []

    for k, seed in enumerate(SEEDS):
        print(f"--- seed {seed}  ({k+1}/{len(SEEDS)}) ---")
        # Paper's RNG order: seed -> train -> test -> model init
        torch.manual_seed(seed); np.random.seed(seed)
        train_ds = SimulatedDataset(SIM_N_TRAIN, SIM_P, maxlen=SIM_MAXLEN, device=device).data
        test_ds  = SimulatedDataset(SIM_N_TEST,  SIM_P, maxlen=SIM_MAXLEN, device=device).data

        for g in GAMMAS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(g, SIM_P, SIM_NHEADS, SIM_DK, SIM_DV, SIM_NCUM,
                               mask_pairwise, pairwise_distance_matrix,
                               distance_to_end_matrix)
            per_var_mse = train_and_eval(model, train_ds, test_ds, SIM_EPOCHS)
            mse[gamma_label(g)].append(per_var_mse)
            extra = ""
            if g == "learned":
                lg = float(model.multiheadattn.gamma.detach().cpu())
                learned_gammas.append(lg)
                extra = f"  learned_gamma={lg:.2f}"
            print(f"  gamma={gamma_label(g):>7}  MSE_target={per_var_mse[SIM_PREDIDX]:.4f}"
                  f"  MSE_mean={per_var_mse.mean():.4f}{extra}")

            # §2.3 test only on the LAST seed (so all gammas see the same model
            # population); using one seed keeps cost manageable while giving a
            # ground-truth comparison
            if k == len(SEEDS) - 1:
                avepval, _stdpval, _ctx, _tg = statistical_testing(
                    model, train_ds, p=SIM_P, predindex=SIM_PREDIDX,
                    nrepp=TEST_NREPP, target_sample_size=TEST_V,
                    return_pval_mat=False, seed=TEST_SEED,
                )
                pvals[gamma_label(g)] = avepval
                print(f"    §2.3 p-values: " + ", ".join(
                    f"v{j}={avepval[j]:.3f}" for j in range(SIM_P)))

    # write MSE CSV
    rows = []
    for g in GAMMAS:
        arr = np.stack(mse[gamma_label(g)])  # (n_seeds, p)
        rows.append({
            "gamma": gamma_label(g),
            "MSE_target_mean": arr[:, SIM_PREDIDX].mean(),
            "MSE_target_std":  arr[:, SIM_PREDIDX].std(),
            "MSE_mean_mean":   arr.mean(axis=1).mean(),
            "MSE_mean_std":    arr.mean(axis=1).std(),
        })
    df_mse = pd.DataFrame(rows)
    df_mse.to_csv(os.path.join(OUT_DIR, "mse_simulation.csv"), index=False)

    # write §2.3 p-values CSV (per-gamma per-variable)
    pv_rows = []
    for g in GAMMAS:
        pv = pvals[gamma_label(g)]
        if pv is None: continue
        for j in range(SIM_P):
            pv_rows.append({"gamma": gamma_label(g), "variable": j,
                             "is_signal": j in (0, 1, 2),
                             "mean_pvalue": float(pv[j])})
    pd.DataFrame(pv_rows).to_csv(os.path.join(OUT_DIR, "test_simulation.csv"),
                                  index=False)

    print(f"\nSimulation summary:\n{df_mse.to_string(index=False)}")
    if learned_gammas:
        print(f"Learned gamma across seeds: "
              f"mean={np.mean(learned_gammas):.2f}  "
              f"std={np.std(learned_gammas):.2f}  "
              f"values={[round(x,2) for x in learned_gammas]}")
    return df_mse


# --------------------------------------------------------------------------- #
# Real-data sweep (LORA D1, LORA D2)                                          #
# --------------------------------------------------------------------------- #

def run_real(data_str, predindex):
    print(f"\n=== {data_str}: 10-fold CV, gamma sweep ===")
    tensors, maxlen = load_real_data(data_str)
    n_total = len(tensors)
    p = tensors[0].shape[1]
    print(f"  loaded n={n_total} sequences, p={p}, maxlen={maxlen}")
    # Cap maxlen at REAL_MAXLEN for parity with paper convention
    eff_maxlen = max(REAL_MAXLEN, maxlen)
    mask_pairwise = create_custom_mask_pair(eff_maxlen, device)
    distance_to_end_matrix = create_distance_to_end_matrix(eff_maxlen, device)
    pairwise_distance_matrix = create_pairwise_distance_matrix(eff_maxlen, device)

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True,
                        random_state=CV_RANDOM_STATE).split(np.arange(n_total)))
    assert len(folds) == len(SEEDS)

    mse = {gamma_label(g): [] for g in GAMMAS}
    learned_gammas = []

    for f_idx, ((tr_idx, te_idx), seed) in enumerate(zip(folds, SEEDS)):
        train_ds = [tensors[i] for i in tr_idx]
        test_ds  = [tensors[i] for i in te_idx]
        print(f"--- fold {f_idx+1}/{N_SPLITS}, seed={seed} "
              f"(n_train={len(train_ds)}, n_test={len(test_ds)}) ---")

        for g in GAMMAS:
            torch.manual_seed(seed); np.random.seed(seed)
            model = make_model(g, p, REAL_NHEADS, REAL_DK, REAL_DV, REAL_NCUM,
                                mask_pairwise, pairwise_distance_matrix,
                                distance_to_end_matrix)
            per_var_mse = train_and_eval(model, train_ds, test_ds, REAL_EPOCHS)
            mse[gamma_label(g)].append(per_var_mse)
            extra = ""
            if g == "learned":
                lg = float(model.multiheadattn.gamma.detach().cpu())
                learned_gammas.append(lg)
                extra = f"  learned_gamma={lg:.2f}"
            print(f"  gamma={gamma_label(g):>7}  "
                  f"MSE_target={per_var_mse[predindex]:.4f}{extra}")

    rows = []
    for g in GAMMAS:
        arr = np.stack(mse[gamma_label(g)])
        rows.append({
            "gamma": gamma_label(g),
            "MSE_target_mean": arr[:, predindex].mean(),
            "MSE_target_std":  arr[:, predindex].std(),
            "MSE_mean_mean":   arr.mean(axis=1).mean(),
            "MSE_mean_std":    arr.mean(axis=1).std(),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, f"mse_{data_str}.csv")
    df.to_csv(out, index=False)
    print(f"\n{data_str} summary:\n{df.to_string(index=False)}")
    if learned_gammas:
        print(f"Learned gamma across folds: "
              f"mean={np.mean(learned_gammas):.2f}  "
              f"std={np.std(learned_gammas):.2f}  "
              f"values={[round(x,2) for x in learned_gammas]}")
    return df


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    torch.set_printoptions(sci_mode=False, precision=6)
    t0 = time.time()
    df_sim = run_simulation()
    df_d1  = run_real("ghq_b_sum", predindex=2)
    df_d2  = run_real("ghq_sum",   predindex=2)

    # Plain-text summary the response letter can quote
    lines = []
    lines.append("=== Gamma sensitivity (Minor #3) ===")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"Simulation: n_train={SIM_N_TRAIN}, n_test={SIM_N_TEST}, "
                 f"EPOCHS={SIM_EPOCHS}")
    lines.append(f"Real data: 10-fold CV, EPOCHS={REAL_EPOCHS}, "
                 f"random_state={CV_RANDOM_STATE}")
    lines.append("")
    for name, df in [("Simulation MSE_target (j_3)", df_sim),
                     ("LORA D1 MSE_target",         df_d1),
                     ("LORA D2 MSE_target",         df_d2)]:
        lines.append(name + ":")
        for _, r in df.iterrows():
            lines.append(f"  gamma={r['gamma']:>7}  "
                         f"{r['MSE_target_mean']:.4f} ± {r['MSE_target_std']:.4f}")
        lines.append("")
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
