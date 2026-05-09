"""Scaled-down vanilla transformer baseline (parameter-matched control).

Built to address reviewer §3.1: a "scaled-down vanilla transformer at matched
parameter count (e.g. 1 layer, 1 head, d_model chosen so that the total number
of parameters equals that of the MiniTransformer)" -- the reviewer's stated
purpose is to "isolate the effect of the *specific* simplifications in
Section 2.2 from the effect of simply having fewer parameters".

Design choices:
- 1 encoder layer, 1 attention head (as the reviewer specified).
- Standard ``nn.TransformerEncoderLayer`` with ``norm_first=False`` and
  ``dim_feedforward = 4 * d_model`` (the canonical default). ``d_model`` is
  chosen via ``find_matched_d_model`` so that the total parameter count is
  closest to the MiniTransformer being compared against.
- Learned positional embeddings up to ``max_len`` (the same precomputed
  positional cap used by MiniTransformer).
- Causal self-attention (lower-triangular ``src_mask``) so every position only
  sees its own past, matching MiniTransformer's autoregressive setup.
- Forward signature mirrors MiniTransformer: ``model((x, padded_mask))`` where
  ``x`` has shape ``(B, T_in, p)`` and the returned tensor has shape
  ``(B, T_in - 1, p)``. The first time-step output is dropped so the alignment
  with ``mini_transformer_loss`` (which compares against ``target[:, 2:, :]``)
  is identical to MiniTransformer's. As a result, this class is a drop-in
  replacement in ``train_mini_transformer`` and the per-target / gate code in
  ``v_sweep_and_gate*.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScaledVanillaTransformer(nn.Module):
    """Causal vanilla transformer with one layer and one attention head."""

    def __init__(
        self,
        p: int,
        d_model: int = 4,
        n_heads: int = 1,
        dim_feedforward: int | None = None,
        max_len: int = 10,
        dropout: float = 0.0,
        device=None,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.in_proj = nn.Linear(p, d_model)
        # Learned positional embedding (one row per position up to max_len).
        # Initialised small to mimic the scale of standard transformer embeds.
        self.pos_embed = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )

        self.out_proj = nn.Linear(d_model, p)

        if device is not None:
            self.to(device)

    def forward(self, data):
        """Args:
            data (tuple): ``(x, padded_mask)`` with
                ``x`` of shape ``(B, T_in, p)`` (float),
                ``padded_mask`` of shape ``(B, T_in, p)`` (bool, True = valid).

        Returns:
            torch.Tensor of shape ``(B, T_in - 1, p)``.
        """
        x, padded_mask = data
        B, T_in, p = x.shape
        if T_in > self.max_len:
            raise ValueError(
                f"sequence length {T_in} exceeds max_len {self.max_len} "
                f"(positional embeddings only cover up to max_len)."
            )

        h = self.in_proj(x) + self.pos_embed[:T_in].unsqueeze(0)  # (B, T_in, d_model)

        # Causal self-attention mask: position i may attend to 0..i. Use bool
        # to match the dtype of ``src_key_padding_mask`` (PyTorch >= 2.1 warns
        # otherwise). True means "this position is masked out".
        causal_mask = torch.triu(
            torch.ones(T_in, T_in, dtype=torch.bool, device=h.device),
            diagonal=1,
        )

        # Per-position padding mask: True means "ignore this key position".
        # ``padded_mask`` is True for valid positions in any feature; the first
        # feature column is a faithful proxy (every feature shares the same
        # pad pattern within a sequence). Invert to get the "ignore" semantics.
        key_padding_mask = ~padded_mask[:, :, 0]  # (B, T_in)

        # If a row in the batch is fully padded (e.g. sequence shorter than
        # T_in), torch's MHA refuses with NaNs. The collate_function used in
        # this project never produces fully-padded rows because every batch is
        # padded only up to the max length actually present, so this is safe.
        h = self.encoder_layer(
            h,
            src_mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        out = self.out_proj(h)  # (B, T_in, p)
        # Drop the first time step's output so the (B, T_in - 1, p) shape lines
        # up with ``target[:, 2:, :]`` in mini_transformer_loss.
        return out[:, 1:, :]


def count_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def find_matched_d_model(
    p: int,
    target_params: int,
    max_len: int = 10,
    n_heads: int = 1,
    dim_feedforward_mult: int = 4,
    search_range=(2, 32),
) -> tuple[int, int]:
    """Search for the ``d_model`` whose ScaledVanillaTransformer has parameter
    count closest to ``target_params``. Returns ``(best_d_model, n_params)``."""
    best_d, best_diff, best_n = None, float("inf"), None
    for d in range(search_range[0], search_range[1] + 1):
        ff = dim_feedforward_mult * d
        m = ScaledVanillaTransformer(
            p, d_model=d, n_heads=n_heads, dim_feedforward=ff, max_len=max_len
        )
        n = count_params(m)
        diff = abs(n - target_params)
        if diff < best_diff:
            best_d, best_diff, best_n = d, diff, n
    assert best_d is not None and best_n is not None
    return best_d, best_n


__all__ = [
    "ScaledVanillaTransformer",
    "count_params",
    "find_matched_d_model",
]
