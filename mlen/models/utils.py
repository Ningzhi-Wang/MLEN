import torch
from torch.nn import functional as F


def global_infonce(pred, target, temperature=0.07):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)

    # Shape: [batch, batch]
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)

    return (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    ) / 2


def multi_positive_infonce(pred, target, temperature=0.07, target_mse_atol=1e-6):
    target = target.float()
    pairwise_target_mse = (target[:, None] - target[None, :]).square().mean(dim=-1)

    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target, dim=-1)

    logits = pred @ target.T / temperature
    positive_mask = torch.isclose(
        pairwise_target_mse,
        torch.zeros_like(pairwise_target_mse),
        rtol=0.0,
        atol=target_mse_atol,
    )

    def direction_loss(scores, positives):
        log_denominator = torch.logsumexp(scores, dim=1)
        log_numerator = torch.logsumexp(
            scores.masked_fill(~positives, float("-inf")),
            dim=1,
        )
        return (log_denominator - log_numerator).mean()

    return (
        direction_loss(logits, positive_mask)
        + direction_loss(logits.T, positive_mask.T)
    ) / 2