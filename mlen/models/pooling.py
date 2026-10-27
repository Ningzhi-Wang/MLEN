"""Per-aspect attention pooling for the conditional prediction head."""

import torch
from torch import nn


class AspectAttentionPool(nn.Module):
    """Pool frames into one vector per aspect with learned aspect queries.

    The queries are a [num_aspects, dim] table: a query projection applied
    to a one-hot aspect is a row lookup. Keys and values come from the
    frames. The queries start at zero and the value and output projections
    at identity, so every aspect initially receives the plain mean of the
    frames and the head reproduces mean pooling exactly.
    """

    def __init__(self, num_aspects, dim, num_heads=8):
        super().__init__()
        if num_aspects <= 0 or dim <= 0:
            raise ValueError("num_aspects and dim must be positive")
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.queries = nn.Parameter(torch.zeros(num_aspects, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        with torch.no_grad():
            self.attention.in_proj_weight[2 * dim:].copy_(torch.eye(dim))
            self.attention.in_proj_bias[2 * dim:].zero_()
            self.attention.out_proj.weight.copy_(torch.eye(dim))
            self.attention.out_proj.bias.zero_()

    @property
    def num_aspects(self):
        return self.queries.size(0)

    def forward(self, frames, key_padding_mask=None):
        """frames [B, T, D]; key_padding_mask [B, T] with True for padding.

        Returns [B, num_aspects, D]. A fully padded track attends to all of
        its frames rather than producing NaNs.
        """
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.bool()
            fully_padded = key_padding_mask.all(dim=-1, keepdim=True)
            key_padding_mask = key_padding_mask & ~fully_padded
        queries = self.queries.unsqueeze(0).expand(frames.size(0), -1, -1)
        pooled, _ = self.attention(
            queries, frames, frames,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        return pooled
