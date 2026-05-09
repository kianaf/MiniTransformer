"""iTransformer baseline (Liu et al., ICLR 2024) adapted for cohort data.

Original idea
-------------
The standard transformer for time-series forecasting attends across *time*:
each time step is a token. iTransformer inverts this: each *variable's*
full timeseries is treated as a single token of length T, and attention is
applied across the p variables at each position. The original architecture
is:

    1. Each variable v's length-T history is projected to a d_model embedding,
       producing p tokens of dimension d_model.
    2. A transformer block attends across these p variable-tokens (sequence
       length = p), letting each variable's representation incorporate cross-
       variable information.
    3. A per-variable output head projects each token back to the forecast
       horizon.

This factorisation is one of the strongest small-footprint architectures on
multivariate-forecasting benchmarks at the time of writing (cf. Liu et al.
2024). The reviewer cited it as a recommended compact baseline (§3.1).

Adaptation to MiniTransformer's cohort setup
--------------------------------------------
MiniTransformer's setup is one-step prediction at every position: given a
cohort sequence of length ``T``, output[t] predicts position t+2 from history
positions 0..t+1. To match this, our adaptation applies iTransformer at every
prediction position independently:

- For each prediction position t = 1, ..., T_in - 1:
    * Take the causal history ``x[:, :t+1, :]`` (positions 0..t).
    * Pad on the left to a fixed window ``history_len``.
    * Embed each variable's length-``history_len`` history with a *shared*
      linear projection to ``d_model``, giving (B, p, d_model).
    * Apply one transformer encoder layer with attention over the p variables
      (no temporal mask is needed -- the sequence dimension is the variable
      axis here).
    * Project each variable's output token to a scalar one-step prediction.
- Stack outputs to shape (B, T_in - 1, p).

This preserves iTransformer's variable-axis attention while fitting the
cohort one-step prediction loop. Forward signature matches MiniTransformer
so it is a drop-in replacement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ITransformer(nn.Module):
    """Variable-axis attention transformer adapted for cohort one-step
    prediction.

    Parameters
    ----------
    p : int
        Number of variables (channels).
    d_model : int
        Embedding dimension for each variable's history.
    n_heads : int
        Number of attention heads (across the variable axis).
    history_len : int
        Fixed history-window length used to embed each variable. Histories
        shorter than ``history_len`` are left-padded with zeros.
    dim_feedforward : int | None
        Standard transformer FFN width. Defaults to ``2 * d_model`` to keep
        the baseline parameter-light at small ``d_model``.
    dropout : float
    max_len : int
        Sanity bound on the input sequence length.
    """

    def __init__(
        self,
        p: int,
        d_model: int = 8,
        n_heads: int = 1,
        history_len: int = 10,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        max_len: int = 10,
        device=None,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.history_len = history_len
        self.max_len = max_len
        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        # Per-variable history embedding: (B, p, history_len) -> (B, p, d_model)
        # Shared across variables (one Linear, applied to each variable's
        # history vector). This is the canonical iTransformer choice.
        self.embed = nn.Linear(history_len, d_model)

        # Transformer block whose sequence dimension is the variable axis.
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        # Per-variable output: (B, p, d_model) -> (B, p, 1) one-step prediction.
        # Shared linear, applied per variable.
        self.head = nn.Linear(d_model, 1)

        if device is not None:
            self.to(device)

    def _predict_one_step(self, history):
        """history: (B, L, p) -- left-pad to history_len and produce (B, 1, p)
        one-step-ahead prediction."""
        B, L, p = history.shape
        if L < self.history_len:
            pad = torch.zeros(B, self.history_len - L, p,
                              device=history.device, dtype=history.dtype)
            history_padded = torch.cat([pad, history], dim=1)
        elif L > self.history_len:
            history_padded = history[:, -self.history_len:, :]
        else:
            history_padded = history

        # Move variables to the sequence axis: (B, p, history_len)
        h = history_padded.transpose(1, 2)
        # Embed each variable's history: (B, p, d_model)
        h = self.embed(h)
        # Attention across variables (no causal mask: variables are unordered)
        h = self.encoder_layer(h)
        # Per-variable head: (B, p, 1) -> (B, 1, p)
        out = self.head(h).squeeze(-1).unsqueeze(1)
        return out

    def forward(self, data):
        """Args:
            data (tuple): ``(x, padded_mask)``, both shape ``(B, T_in, p)``.

        Returns:
            torch.Tensor of shape ``(B, T_in - 1, p)``.
        """
        x, _padded_mask = data
        B, T_in, p = x.shape
        if T_in > self.max_len:
            raise ValueError(
                f"sequence length {T_in} exceeds max_len {self.max_len}."
            )
        outs = []
        for t in range(1, T_in):
            history = x[:, :t + 1, :]
            outs.append(self._predict_one_step(history))
        return torch.cat(outs, dim=1)


__all__ = ["ITransformer"]
