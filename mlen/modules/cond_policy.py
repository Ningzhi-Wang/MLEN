"""Condition sampling: which of a track's aspects are active in a batch."""

import torch

COND_POLICIES = ("single", "multi", "all")


def sample_cond(available, policy, drop_prob=0.0, ids=None):
    """Return a multi-hot [B, A] condition chosen from the available aspects.

    available: [B, A] bool, True where the track has a target for the aspect.
    policy: "single" activates one aspect, "multi" a random subset of size
        1..n, "all" every available aspect.
    drop_prob: probability that a track gets the all-zero (unconditional)
        condition instead, whatever the policy.
    ids: optional [B] track ids. When given, the choice is a fixed function
        of the id (for validation); otherwise it is random.
    """
    available = torch.as_tensor(available).bool()
    if available.dim() != 2:
        raise ValueError("available must be [batch, aspects]")
    if policy not in COND_POLICIES:
        raise ValueError(f"Unknown cond policy {policy!r}; choose from {COND_POLICIES}")
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in the range [0.0, 1.0]")
    batch, num_aspects = available.shape
    device = available.device
    if ids is not None:
        ids = torch.as_tensor(ids, device=device).long().flatten()
        if ids.numel() != batch:
            raise ValueError("ids must contain one entry per track")

    count = available.sum(dim=-1)
    if policy == "all":
        k = count
    elif policy == "single":
        k = count.clamp_max(1)
    elif ids is None:
        k = (torch.rand(batch, device=device) * count).long() + 1
    else:
        k = ids % count.clamp_min(1) + 1
    k = torch.minimum(k, count)
    if drop_prob > 0.0:
        k = k.masked_fill(torch.rand(batch, device=device) < drop_prob, 0)

    if ids is None:
        scores = torch.rand(batch, num_aspects, device=device)
    else:
        # Rotate the aspect order by the track id: fixed per track, yet the
        # chosen aspects are spread over the whole set.
        offsets = (torch.arange(num_aspects, device=device)[None] - ids[:, None]) % num_aspects
        scores = -offsets.float()
    scores = scores.masked_fill(~available, float("-inf"))
    rank = scores.argsort(dim=-1, descending=True).argsort(dim=-1)
    return (rank < k[:, None]).to(torch.float32)
