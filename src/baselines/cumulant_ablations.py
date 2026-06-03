"""Eq. 3 cumulant-head ablations for the §3.3 / §3.4 response.

Two variants, both subclassing the paper's MultiHeadAttention so that only the
cumulant-pooling stage (Eq. 3) changes; everything else (the Eq. 1 pairwise
decay, scalar attention, readout) is identical to MiniTransformer.

- HorizonDecayOff: keep the cumulant pooling but replace the
  prediction-horizon decay factor exp(-(w_horizon * |t_{T+1} - t_i|)^gamma)
  with a uniform (causal) average over the valid history. Isolates the
  horizon-decay term that the reviewer doubts in §3.4, while leaving the act
  of cumulative pooling intact.

- CumulantOff: drop the cumulant pooling entirely; the readout uses only the
  per-timestep Eq. 2 output at the final history position t = T. This is the
  reviewer's literal §3.3 request ("remove Eq. 3, use only Eq. 2").

The paper's pooling lives in MultiHeadAttention.forward as:

    pooling_weights = self.exponential_decay_pred(distance_to_end[:T-1,:T-1], w)
    head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
    head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)

We override forward to change only the construction of `pooling_weights`
(HorizonDecayOff) or to bypass pooling (CumulantOff).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.transformers import MiniTransformer, MultiHeadAttention


class _CumulantAblationAttention(MultiHeadAttention):
    """MultiHeadAttention with a `cumulant_mode` flag.

    cumulant_mode:
      - 'full'        : paper behaviour (pairwise + horizon decay). Default.
      - 'pairwise_off': Eq. 1 pairwise temporal-decay multiplier set to 1
                        (standard softmax attention); cumulant pooling and its
                        horizon decay retained.
      - 'horizon_off' : uniform causal pooling (Eq. 3 horizon decay set to 1);
                        Eq. 1 pairwise decay retained.
      - 'cumulant_off': no pooling; use only the t=T per-timestep output.
    """
    def __init__(self, *args, cumulant_mode: str = "full", **kwargs):
        super().__init__(*args, **kwargs)
        if cumulant_mode not in ("full", "pairwise_off", "horizon_off", "cumulant_off"):
            raise ValueError(f"bad cumulant_mode: {cumulant_mode!r}")
        self.cumulant_mode = cumulant_mode
        if cumulant_mode == "pairwise_off":
            # The pairwise-distance scalar (w_dist) is unused; freeze it.
            self.distance_between_two_positions_weight.requires_grad_(False)
        elif cumulant_mode in ("horizon_off", "cumulant_off"):
            # The horizon-decay scalar (w_horizon) is unused; freeze it so the
            # parameter count is reported honestly (one fewer trainable scalar
            # than MiniTransformer).
            self.distance_to_end_weight.requires_grad_(False)

    def forward(self, data):
        if self.cumulant_mode == "full":
            return super().forward(data)

        if self.cumulant_mode == "pairwise_off":
            return self._forward_pairwise_off(data)

        x = data[0]
        padding_mask = data[1]
        batch_size, seq_len, _ = x.size()

        # --- identical attention computation to the paper (Eq. 1-2) ---------
        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(
            batch_size, self.num_heads, seq_len, seq_len)
        attention_scores, V = self.get_attention(
            x,
            self.exponential_decay_pair(
                self.pairwise_distance_matrix[:seq_len, :seq_len],
                self.distance_between_two_positions_weight[0, 0],
            ).expand(batch_size, self.num_heads, seq_len, seq_len),
            mask_pairwise,
            padding_mask,
        )
        head_output = (attention_scores.matmul(V)).transpose(1, 2).squeeze(dim=-1)
        # head_output: (batch, T-1, num_heads)

        T_pool = seq_len - 1

        if self.cumulant_mode == "cumulant_off":
            # No pooling: take only the last history position's per-timestep
            # output (the Eq. 2 output at t = T). Shape (batch, num_heads).
            last = head_output[:, -1, :]
            head_outputs_cum = self.cum_weights(last)
            return head_outputs_cum

        # cumulant_mode == "horizon_off":
        # Uniform causal pooling. distance_to_end_matrix[i,j] is finite (a valid
        # distance) for j <= i and 1e9 for j > i, so "valid" = (< 1e9).
        dte = self.distance_to_end_matrix[:T_pool, :T_pool]
        valid = (dte < 1e8).to(head_output.dtype)           # (T_pool, T_pool)
        row_sum = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooling_weights = valid / row_sum                   # row-normalised uniform
        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        head_outputs_cum = self.cum_weights(head_output_weighted_sum_pool)
        return head_outputs_cum

    def _forward_pairwise_off(self, data):
        """Eq. 1 pairwise temporal-decay multiplier set to 1: standard softmax
        attention, with the full Eq. 3 horizon-decay cumulant pooling retained.
        Mirrors KernelAttentionNoDecay (the reviewer's Eq. 1 ablation)."""
        x = data[0]
        padding_mask = data[1]
        batch_size, seq_len, _ = x.size()

        mask_pairwise = self.mask_pairwise[:seq_len, :seq_len].expand(
            batch_size, self.num_heads, seq_len, seq_len)

        Q, K, V = self.qkv(x)
        scores = Q.matmul(K.transpose(2, 3)) / math.sqrt(self.dk)
        scores = scores.masked_fill(
            (padding_mask[:, :, 0].unsqueeze(1).unsqueeze(-1)
             .expand(batch_size, self.num_heads, seq_len, seq_len)) != True,
            -1e9,
        )
        attention_scores = torch.nn.functional.softmax(
            (scores + mask_pairwise)[:, :, 1:seq_len, :seq_len], dim=-1,
        )
        head_output = (attention_scores.matmul(V)).transpose(1, 2).squeeze(dim=-1)

        # Full Eq. 3 horizon-decay pooling (unchanged from the paper).
        pooling_weights = self.exponential_decay_pred(
            self.distance_to_end_matrix[:seq_len - 1, :seq_len - 1],
            self.distance_to_end_weight[0, 0],
        )
        head_output_weighted_sum_pool = pooling_weights.matmul(head_output)
        return self.cum_weights(head_output_weighted_sum_pool)


class CumulantAblationMiniTransformer(MiniTransformer):
    """MiniTransformer with a `cumulant_mode` flag selecting the Eq. 3 ablation.

    cumulant_mode='full' reproduces MiniTransformer exactly; 'horizon_off' and
    'cumulant_off' are the two §3.3 ablations.
    """
    def __init__(self, d_model, num_heads, dk, dv, ncum, mask_pairwise,
                 pairwise_distance_matrix, distance_to_end_matrix, device,
                 cumulant_mode: str = "full"):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ncum = ncum
        self.dk = dk
        self.dv = dv
        self.device = device
        self.cumulant_mode = cumulant_mode
        self.multiheadattn = _CumulantAblationAttention(
            d_model, num_heads, dk, dv, ncum,
            mask_pairwise, pairwise_distance_matrix, distance_to_end_matrix, device,
            cumulant_mode=cumulant_mode,
        )
        self.prediction = nn.Linear(self.ncum, self.d_model)
