"""Device-local waveform masking with MuQ-compatible token indices."""

import torch


def mask_waveform(x, *, mask_hop, hop_length, mask_prob):
    # Keep MuQ's 24 kHz block sizes and leave an incomplete final block alone.
    samples_per_block = int(24000 * mask_hop)
    tokens_per_block = int(24000 / hop_length / 2 / 2 * mask_hop)
    batch, samples = x.shape
    blocks = samples // samples_per_block
    selected = torch.rand(batch, blocks, device=x.device) < mask_prob

    # Retain the [N, 2] integer API used by prediction filtering and RVQ loss.
    # CUDA nonzero still synchronizes to determine N, but this is the small
    # token mask rather than a sample-sized waveform index array.
    token_indices = selected.repeat_interleave(tokens_per_block, dim=1).nonzero()
    masked = x.clone()
    prefix = x[:, :blocks * samples_per_block].reshape(batch, blocks, samples_per_block)
    noise = torch.randn_like(prefix) * 0.1
    masked[:, :blocks * samples_per_block] = torch.where(
        selected.unsqueeze(-1), noise, prefix
    ).reshape(batch, blocks * samples_per_block)
    return masked, token_indices
