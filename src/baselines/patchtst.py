"""PatchTST baseline (Nie et al., ICLR 2023, "A Time Series is Worth 64 Words")
adapted for cohort data.

Original idea
-------------
PatchTST has two defining ingredients:

  1. **Patching.** Each univariate series is split into (possibly overlapping)
     subseries patches of length ``patch_len`` taken every ``stride`` steps;
     each patch becomes one token. This shortens the attention sequence and
     lets each token summarise a local window.
  2. **Channel independence.** Every variable (channel) is processed by the
     *same* transformer weights, independently of the other variables; there
     is no cross-variable mixing inside the model.

The original experiments use long look-back windows (input length 96 and
above) with ``patch_len=16``, ``stride=8``. The reviewer cited PatchTST as a
recommended compact baseline (Comment 3.1).

Adaptation to MiniTransformer's cohort setup
--------------------------------------------
MiniTransformer's setup is one-step prediction at every position: given a
sequence of length ``T``, output[t] predicts position t+2 from history
positions 0..t+1. To match this, our adaptation applies PatchTST at every
prediction position independently:

- For each prediction position t = 1, ..., T_in - 1:
    * Take the causal history ``x[:, :t+1, :]`` and left-pad to a fixed
      window ``history_len``.
    * For each variable independently, patchify its length-``history_len``
      history into patches of length ``patch_len`` with step ``stride``
      (channel-independent: the same embedding and transformer weights are
      shared across all variables and all patches).
    * Embed each patch (Linear ``patch_len`` -> ``d_model``), add a learned
      positional embedding over patches, run one transformer encoder layer
      attending across the patch tokens, flatten, and project to a scalar
      one-step prediction.
- Stack outputs to shape (B, T_in - 1, p).

Because ``T`` is at most 10 here, we use a short patch (``patch_len=3``,
``stride=2``): on a history window of length 10 this yields 4 patches, the
same patching mechanism as the original model at a scale appropriate to short
clinical sequences. The forward signature matches MiniTransformer so this is a
drop-in replacement in the existing training and evaluation harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Channel-independent patch transformer adapted for cohort one-step
    prediction.

    Parameters
    ----------
    p : int
        Number of variables (channels).
    d_model : int
        Per-patch embedding dimension.
    n_heads : int
        Number of attention heads (across the patch axis).
    history_len : int
        Fixed history-window length each variable is patchified over. Shorter
        histories are left-padded with zeros.
    patch_len : int
        Patch (subseries) length.
    stride : int
        Step between consecutive patches.
    dim_feedforward : int | None
        Transformer FFN width. Defaults to ``2 * d_model`` to stay light.
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
        patch_len: int = 3,
        stride: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        max_len: int = 10,
        device=None,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.history_len = history_len
        self.patch_len = patch_len
        self.stride = stride
        self.max_len = max_len
        if dim_feedforward is None:
            dim_feedforward = 2 * d_model

        # Number of patches over a length-history_len window.
        self.n_patches = (history_len - patch_len) // stride + 1
        if self.n_patches < 1:
            raise ValueError(
                f"patch_len={patch_len}, stride={stride} give <1 patch on "
                f"history_len={history_len}."
            )
        # The patches span exactly this many timesteps. When this is < history_len
        # (because (history_len - patch_len) is not a multiple of stride), a naive
        # left-aligned unfold would DROP the most recent timesteps -- fatal for
        # one-step-ahead prediction. We therefore align the patched window to END
        # at the most recent timestep (see _predict_one_step).
        self.patch_span = (self.n_patches - 1) * stride + patch_len

        # Channel-independent: one shared patch embedding applied to every
        # variable and every patch.
        self.patch_embed = nn.Linear(patch_len, d_model)
        # Learned positional embedding over the patch sequence.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        # Flatten head: (n_patches * d_model) -> 1, shared across variables.
        self.head = nn.Linear(self.n_patches * d_model, 1)

        if device is not None:
            self.to(device)

    def _patchify(self, hist):
        """hist: (B, p, history_len) -> patches (B, p, n_patches, patch_len)."""
        # unfold along the time axis.
        return hist.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def _predict_one_step(self, history):
        """history: (B, L, p) -> (B, 1, p) one-step-ahead prediction.

        The window is aligned to END at the most recent timestep so the final
        patch always contains it: we take the last ``patch_span`` timesteps,
        left-padding with zeros only if the history is shorter than that.
        """
        B, L, p = history.shape
        span = self.patch_span
        if L < span:
            pad = torch.zeros(B, span - L, p,
                              device=history.device, dtype=history.dtype)
            history = torch.cat([pad, history], dim=1)
        else:
            history = history[:, -span:, :]

        # (B, p, patch_span)
        h = history.transpose(1, 2)
        # (B, p, n_patches, patch_len)
        patches = self._patchify(h)
        # Treat (B, p) as the channel-independent batch: (B*p, n_patches, patch_len)
        bp = patches.reshape(B * p, self.n_patches, self.patch_len)
        # Embed each patch + positional embedding: (B*p, n_patches, d_model)
        tokens = self.patch_embed(bp) + self.pos_embed
        # Attention across patches (shared weights for every variable).
        enc = self.encoder_layer(tokens)
        # Flatten patches and project to one scalar: (B*p, 1)
        flat = enc.reshape(B * p, self.n_patches * self.d_model)
        out = self.head(flat)
        # Back to (B, 1, p)
        return out.reshape(B, p).unsqueeze(1)

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
            outs.append(self._predict_one_step(x[:, :t + 1, :]))
        return torch.cat(outs, dim=1)


__all__ = ["PatchTST"]
