"""DLinear baseline (Zeng et al., 2023, "Are Transformers Effective for Time
Series Forecasting?", AAAI 2023).

DLinear is a simple but strong forecasting baseline that decomposes the input
into a trend (moving average) and a remainder, then applies a per-channel
linear map from the historical length to the prediction length on each
component. The original paper shows that this simple model is competitive with
or better than several transformer-based forecasters on standard TSF
benchmarks; the reviewer cited it as a recommended compact baseline (§3.1).

Adaptation to MiniTransformer's setup
-------------------------------------
The original DLinear is designed for fixed-length forecasting (history of L
points, predict horizon of H points). MiniTransformer's setup is one-step-
ahead prediction at every position (output[t] predicts the value at position
t+2, given history positions 0..t+1). To match this, our DLinear:

- Operates on inputs of shape ``(B, T_in, p)`` (same as MiniTransformer's
  forward signature ``(x, padded_mask)``).
- For each prediction position ``t = 1, ..., T_in - 1``, takes the history
  ``x[:, :t+1, :]`` (causal), pads it on the left to a fixed window length
  ``L``, applies the standard DLinear (decomposition + per-channel linear
  with a horizon of 1), and outputs the one-step-ahead prediction.
- Returns shape ``(B, T_in - 1, p)`` so it is a drop-in replacement in the
  existing training and evaluation harness.

Per-channel: each output feature has its own linear weights (no cross-channel
mixing). This is the channel-independent design DLinear is known for.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MovingAverage1D(nn.Module):
    """Moving-average smoothing along the time axis with reflective padding so
    that the output has the same length as the input. ``kernel_size`` should be
    odd; the smoother is applied independently per channel."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        # x: (B, L, C). Permute to (B, C, L) for 1D pooling.
        x_t = x.transpose(1, 2)
        # Reflective padding to keep length equal.
        x_pad = F.pad(x_t, (self.pad, self.pad), mode="replicate")
        x_smooth = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        return x_smooth.transpose(1, 2)


class _SeriesDecomposition(nn.Module):
    """Decompose a series into (trend, residual). Trend is the moving average
    along time; residual is the input minus the trend."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.ma = _MovingAverage1D(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        residual = x - trend
        return residual, trend


class DLinear(nn.Module):
    """Channel-independent DLinear adapted for one-step causal prediction.

    Parameters
    ----------
    p : int
        Number of input/output channels (features).
    history_len : int
        The fixed history window passed into the per-channel linear maps.
        Histories shorter than ``history_len`` are left-padded with zeros.
    kernel_size : int
        Moving-average kernel size for the trend/residual decomposition.
    max_len : int
        Maximum input sequence length (used only for the API parity with
        MiniTransformer; not architecturally relevant).
    """

    def __init__(
        self,
        p: int,
        history_len: int = 10,
        kernel_size: int = 5,
        max_len: int = 10,
        device=None,
    ):
        super().__init__()
        self.p = p
        self.history_len = history_len
        self.max_len = max_len
        self.decomp = _SeriesDecomposition(kernel_size)

        # Channel-independent linear maps from history_len -> 1 (one-step
        # horizon), one set of weights per channel, no cross-channel mixing.
        # This is the design choice DLinear is known for (cf. Zeng et al. 2023);
        # implemented via plain Parameters + einsum rather than nn.Conv1d with
        # groups=p purely for code readability.
        self.weight_residual = nn.Parameter(torch.zeros(p, history_len))
        self.bias_residual   = nn.Parameter(torch.zeros(p))
        self.weight_trend    = nn.Parameter(torch.zeros(p, history_len))
        self.bias_trend      = nn.Parameter(torch.zeros(p))
        # Initialise as identity-ish (last position dominates), which is a
        # reasonable prior for one-step-ahead prediction.
        with torch.no_grad():
            self.weight_residual[:, -1] = 1.0
            self.weight_trend[:, -1]    = 1.0

        if device is not None:
            self.to(device)

    def _predict_one_step(self, history):
        """history: (B, L<=history_len, p) -- left-pad to history_len and
        produce a single (B, 1, p) one-step-ahead prediction."""
        B, L, p = history.shape
        if L < self.history_len:
            pad_len = self.history_len - L
            pad = torch.zeros(B, pad_len, p, device=history.device,
                              dtype=history.dtype)
            history_padded = torch.cat([pad, history], dim=1)
        elif L > self.history_len:
            history_padded = history[:, -self.history_len:, :]
        else:
            history_padded = history

        residual, trend = self.decomp(history_padded)  # each (B, history_len, p)
        # Channel-independent linear: (B, history_len, p) @ (p, history_len) ?
        # We do per-channel: out_c = sum_l w_c[l] * x[:, l, c] + b_c
        # Implemented via einsum.
        out_res = torch.einsum("blp,pl->bp", residual, self.weight_residual) \
                  + self.bias_residual           # (B, p)
        out_trd = torch.einsum("blp,pl->bp", trend, self.weight_trend) \
                  + self.bias_trend              # (B, p)
        return (out_res + out_trd).unsqueeze(1)  # (B, 1, p)

    def forward(self, data):
        """Args:
            data (tuple): (x, padded_mask), both shape (B, T_in, p). Returns
                a tensor of shape (B, T_in - 1, p) where output[t] is the
                one-step-ahead prediction given causal history x[:, :t+2, :].
        """
        x, _padded_mask = data
        B, T_in, p = x.shape
        outs = []
        # For each prediction position t in 1..T_in-1, the prediction is for
        # the value at position t+1 (relative to the input sequence). We use
        # history x[:, :t+1, :] (positions 0..t) to predict position t+1.
        # The output is aligned to ``target[:, 2:, :]`` in the loss when the
        # caller passes x = data[:, :-1, :] as input: then T_in = T - 1, and
        # output[t] for t = 1..T_in-1 is matched against target[t+1]
        # = data[:, t+1, :], i.e. positions 2..T-1 of the original sequence.
        for t in range(1, T_in):
            history = x[:, :t + 1, :]              # positions 0..t
            outs.append(self._predict_one_step(history))  # (B, 1, p)
        return torch.cat(outs, dim=1)              # (B, T_in - 1, p)


__all__ = ["DLinear"]
