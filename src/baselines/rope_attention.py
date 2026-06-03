"""MiniTransformer variant with the temporal-decay positional scheme replaced
by Rotary Position Embedding (RoPE; Su et al., 2024).

This is the §3.3 ablation that targets the positional parameterisation:

    "Replace the authors' positional parameterisation (w_dist, w_horizon,
    gamma) with a rotary positional encoding (RoPE; Su et al., 2024)."

What changes relative to MiniTransformer:

- The pairwise temporal-decay kernel exp(-(w_dist * |t_i - t_l|)^gamma) is
  removed; relative position is instead injected by rotating the per-head
  query/key projections by an angle that depends on the visit index.
- The prediction-horizon decay exp(-(w_horizon * |t_{T+1} - t_i|)^gamma) is
  also removed; the cumulant pooling becomes a uniform sum over history
  positions, weighted only by `cum_weights`. This gives the cleanest "swap
  the positional scheme out" comparison; if you want to keep the horizon
  decay (only Eq. 1 replaced), pass `keep_horizon_decay=True`.

The three scalar parameters (w_dist, w_horizon, gamma) of MiniTransformer
disappear; RoPE adds no learnable parameters. Everything else of
MiniTransformer (scalar-valued attention heads, cumulant pooling shape,
linear regression readout, training schedule) is preserved.

The same class can be used as a *flag* to compare against the original
decay scheme: pass `positional_scheme='decay'` and it behaves exactly like
the paper's MiniTransformer; pass `positional_scheme='rope'` and you get
the RoPE variant. This is the easiest way to run the ablation as a single
sweep without maintaining two separate models.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from src.transformers import MiniTransformer, MultiHeadAttention


def _rope_angles(seq_len: int, dim: int, base: float = 10000.0,
                  device=None) -> torch.Tensor:
    """Return the (seq_len, dim) tensor of rotation angles m * theta_k.

    For each position m in [0, seq_len) and each dim index d in [0, dim),
    the angle is m * theta_{d//2} where theta_k = base ** (-2k / dim).
    The same angle is used for both elements of a (2k, 2k+1) pair, so the
    returned tensor has duplicated columns.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even head dimension; got dim={dim}")
    half = dim // 2
    # theta_k for k = 0, ..., half-1
    k = torch.arange(half, device=device, dtype=torch.float32)
    theta = base ** (-2.0 * k / float(dim))                      # (half,)
    positions = torch.arange(seq_len, device=device,
                              dtype=torch.float32)               # (seq_len,)
    # outer product -> (seq_len, half)
    angles_half = positions[:, None] * theta[None, :]
    # duplicate so the angle for dim 2k and 2k+1 is the same
    angles = torch.repeat_interleave(angles_half, repeats=2, dim=1)  # (seq_len, dim)
    return angles


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleave-rotate trick used by the canonical RoPE implementation:
    swap (x_{2k}, x_{2k+1}) -> (-x_{2k+1}, x_{2k}) along the last dim.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.stack([-x_odd, x_even], dim=-1)
    return out.flatten(-2)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to x.

    Args:
        x: shape (..., seq_len, dim)
        angles: shape (seq_len, dim), produced by _rope_angles.

    Returns:
        Tensor of the same shape as x with the rotation applied.
    """
    # broadcast angles to match x's batch dims
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # Align shapes: angles is (seq_len, dim); x is (..., seq_len, dim).
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Multi-head attention with selectable positional scheme                      #
# --------------------------------------------------------------------------- #


class _RoPEorDecayMultiHeadAttention(MultiHeadAttention):
    """MultiHeadAttention with a `positional_scheme` flag.

    - 'decay': behaves like the paper's MultiHeadAttention (additive decay
      kernel in attention, decay kernel in the cumulant pooling).
    - 'rope': pairwise decay is replaced by a RoPE rotation of the per-head
      Q and K projections; the cumulant pooling becomes uniform (no horizon
      decay) unless `keep_horizon_decay=True`.
    """
    def __init__(self, *args, positional_scheme: str = "decay",
                 keep_horizon_decay: bool = False, rope_base: float = 10000.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if positional_scheme not in ("decay", "rope"):
            raise ValueError(
                f"positional_scheme must be 'decay' or 'rope', got {positional_scheme!r}")
        self.positional_scheme = positional_scheme
        self.keep_horizon_decay = keep_horizon_decay
        self.rope_base = float(rope_base)

        if positional_scheme == "rope":
            # RoPE replaces the two scalar positional parameters; mark them
            # as non-trainable so the parameter count drops cleanly.
            self.distance_between_two_positions_weight.requires_grad_(False)
            self.distance_to_end_weight.requires_grad_(False)
            if self.dk % 2 != 0:
                raise ValueError(
                    f"RoPE requires even dk; got dk={self.dk}. Pass dk=2 or "
                    f"another even value when constructing the model.")

    # --- attention computation (overrides MultiHeadAttention.forward) ----- #

    def forward(self, data):
        if self.positional_scheme == "decay":
            return super().forward(data)
        return self._forward_rope(data)

    def _forward_rope(self, data):
        x = data[0]
        padding_mask = data[1]
        batch_size, seq_len, _ = x.size()

        # Standard Q, K, V projection + head split
        Q, K, V = self.qkv(x)  # shapes (B, H, T, dk/dv)

        # Apply RoPE to Q and K (per head, per position)
        angles = _rope_angles(seq_len, self.dk, base=self.rope_base,
                              device=x.device)  # (T, dk)
        Q = apply_rope(Q, angles)
        K = apply_rope(K, angles)

        # Standard scaled-dot-product attention, masked the same way the
        # paper's MultiHeadAttention does (causal-ish via mask_pairwise,
        # padding via padding_mask). RoPE replaces the additive decay
        # bias; no other change.
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.dk)
        scores = scores.masked_fill(
            (padding_mask[:, :, 0].unsqueeze(1).unsqueeze(-1)
                 .expand(batch_size, self.num_heads, seq_len, seq_len)) != True,
            -1e9,
        )
        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(
            batch_size, self.num_heads, seq_len, seq_len)
        attention_scores = torch.nn.functional.softmax(
            (scores + mask_pairwise)[:, :, 1:seq_len, :seq_len], dim=-1
        )

        head_output = (attention_scores.matmul(V)).transpose(1, 2).squeeze(dim=-1)

        # Cumulant pooling: by default, uniform (no horizon decay). When
        # keep_horizon_decay=True, retain the paper's exponential horizon decay.
        if self.keep_horizon_decay:
            pooling_weights = self.exponential_decay_pred(
                self.distance_to_end_matrix[:seq_len - 1, :seq_len - 1],
                self.distance_to_end_weight[0, 0],
            )
        else:
            T_pool = seq_len - 1
            pooling_weights = torch.ones(
                T_pool, T_pool, device=x.device, dtype=head_output.dtype)
            # Causal lower-triangular mask so position i only pools positions <= i
            causal = torch.tril(torch.ones_like(pooling_weights))
            pooling_weights = pooling_weights * causal
            # Normalise so each row sums to 1 (uniform average over valid history)
            row_sum = pooling_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooling_weights = pooling_weights / row_sum

        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)
        return head_outputs_cum


# --------------------------------------------------------------------------- #
# MiniTransformer wrapper                                                     #
# --------------------------------------------------------------------------- #


class RoPEOrDecayMiniTransformer(MiniTransformer):
    """MiniTransformer with a `positional_scheme` flag.

    Pass `positional_scheme='decay'` (default) for the paper's behaviour.
    Pass `positional_scheme='rope'` for the §3.3 RoPE ablation.

    For RoPE, `dk` must be even (the paper's default dk=1 is odd, so pass
    dk=2 when using RoPE).
    """
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise,
                 pairwise_distance_matrix, distance_to_end_matrix, device,
                 positional_scheme: str = "decay",
                 keep_horizon_decay: bool = False,
                 rope_base: float = 10000.0):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.device = device
        self.positional_scheme = positional_scheme
        self.multiheadattn = _RoPEorDecayMultiHeadAttention(
            d_model, num_heads, dk, dv, ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
            positional_scheme=positional_scheme,
            keep_horizon_decay=keep_horizon_decay,
            rope_base=rope_base,
        )
        self.prediction = nn.Linear(self.ncum, self.d_model)
