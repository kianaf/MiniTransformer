"""MiniTransformer variant with the pairwise temporal-decay kernel removed.

This implements the reviewer's request from §3.1:

    "A linear / kernel attention baseline *without* the temporal-decay kernel
    in Eq. 1, so the decay term's contribution can be attributed cleanly."

It also serves as the §3.3 ablation that targets Eq. 1's exponential-decay
multiplier specifically.

The change relative to MiniTransformer is minimal: in Eq. 1,

    g(x_ti, x_tl; .) = exp( x_ti' W_q . x_tl' W_k )
                     * exp( -(w_dist . |t_i - t_l|)^gamma )

we set the second exponential to 1, leaving the standard softmax kernel-
attention term. All other ingredients of MiniTransformer (multi-head structure,
cumulant pooling, prediction head) are preserved, so the parameter count drops
by exactly one (the ``distance_between_two_positions_weight`` parameter), and
the comparison directly attributes any predictive difference to the decay term
rather than to a wholesale architectural change.

The Eq. 3 prediction-horizon decay is *retained*. To remove that as well,
flip ``include_horizon_decay=False`` in the constructor; that more aggressive
ablation is also exposed for §3.3.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.transformers import (
    MiniTransformer,
    MultiHeadAttention,
)


class _MultiHeadAttentionNoDecay(MultiHeadAttention):
    """MiniTransformer attention with the pairwise temporal-decay multiplier in
    Eq. 1 disabled. Optionally also disables the prediction-horizon decay
    multiplier in Eq. 3 (``include_horizon_decay=False``)."""

    def __init__(self, *args, include_horizon_decay: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_horizon_decay = include_horizon_decay
        # The pairwise-distance-weight parameter is unused now; freeze it so
        # the optimiser sees no gradient. We keep the attribute around so
        # state_dict shapes stay compatible with MiniTransformer checkpoints.
        self.distance_between_two_positions_weight.requires_grad_(False)
        if not include_horizon_decay:
            self.distance_to_end_weight.requires_grad_(False)

    def forward(self, data):
        x = data[0]
        padding_mask = data[1]

        batch_size, seq_len, _ = x.size()

        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(
            batch_size, self.num_heads, seq_len, seq_len
        )

        # Standard softmax attention (no temporal-decay multiplier in Eq. 1).
        Q, K, V = self.qkv(x)
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.dk)
        scores = scores.masked_fill(
            (padding_mask[:, :, 0].unsqueeze(1).unsqueeze(-1)
             .expand(batch_size, self.num_heads, seq_len, seq_len)) != True,
            -1e9,
        )
        # No `+ dist_weight` term; only the causal mask is added.
        attention_scores = torch.nn.functional.softmax(
            (scores + mask_pairwise)[:, :, 1:seq_len, :seq_len],
            dim=-1,
        )

        head_output = (attention_scores.matmul(V)).transpose(1, 2).squeeze(dim=-1)

        # Eq. 3 cumulant pooling: include the prediction-horizon decay unless
        # explicitly disabled (then use uniform pooling weights with the
        # causal mask only).
        if self.include_horizon_decay:
            pooling_weights = self.exponential_decay_pred(
                self.distance_to_end_matrix[:seq_len - 1, :seq_len - 1],
                self.distance_to_end_weight[0, 0],
            )
        else:
            # Uniform causal pooling: ones on/below the diagonal, zeros above.
            ones = torch.ones(
                seq_len - 1, seq_len - 1, device=head_output.device,
                dtype=head_output.dtype,
            )
            pooling_weights = torch.tril(ones)

        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)
        return head_outputs_cum


class KernelAttentionNoDecay(MiniTransformer):
    """Drop-in replacement for MiniTransformer with the Eq. 1 temporal-decay
    multiplier removed (and optionally the Eq. 3 horizon-decay multiplier as
    well, via ``include_horizon_decay=False``).

    All other architectural ingredients (multi-head scalar attention, cumulant
    head, prediction layer) are unchanged. The parameter count is identical
    to MiniTransformer's apart from one frozen scalar (``w_dist``), so this
    baseline targets the *contribution of the decay term itself* rather than
    a different architecture.
    """

    def __init__(self, d_model, num_heads, dk, dv, ncum,
                 mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix,
                 device, *, include_horizon_decay: bool = True):
        # Initialise as a MiniTransformer but swap in the no-decay attention.
        nn.Module.__init__(self)  # bypass MiniTransformer's __init__ to avoid
                                  # building the standard attention twice
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.device = device
        self.multiheadattn = _MultiHeadAttentionNoDecay(
            d_model, num_heads, dk, dv, ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix,
            device,
            include_horizon_decay=include_horizon_decay,
        )
        self.prediction = nn.Linear(self.ncum, self.d_model)


__all__ = ["KernelAttentionNoDecay"]
