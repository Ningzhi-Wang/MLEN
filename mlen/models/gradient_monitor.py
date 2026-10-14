"""Detached conditional-head diagnostics; never modify training gradients."""

import torch
from torch.nn import functional as F

from .utils import cosine_similarity_loss


def aspect_gradient_statistics(
    head, hidden, target, aspect_ids, num_aspects, *,
    pooling="mean", attention_mask=None, loss_weight=1.0,
):
    """Return [sample count, count * mean-loss grad norm, batch count] per aspect.

    Norms cover trainable linear-head parameters only, in FP32 before clipping
    or AMP scaling. Pooling before the linear projection is algebraically
    equivalent to the model's projection-before-pooling, including its bias.
    Detached parameter copies keep autograd and DDP training hooks untouched.
    """
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("Conditional gradient diagnostics require a Linear head")
    with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden = hidden.detach().float()
        valid = None
        if pooling == "cls":
            pooled = hidden[:, 0]
        elif pooling == "mean":
            if attention_mask is None:
                pooled = hidden.mean(dim=1)
            else:
                stride = int(attention_mask.size(-1) / hidden.size(1))
                mask = attention_mask.bool()[:, ::stride][:, :hidden.size(1)]
                mask = mask.unsqueeze(-1).float()
                count = mask.sum(dim=1)
                pooled = (hidden * mask).sum(dim=1) / count.clamp_min(1)
                valid = count > 0
        else:
            raise ValueError(f"Unsupported conditional pooling: {pooling}")

        weight = head.weight.detach().float().clone().requires_grad_(head.weight.requires_grad)
        bias = None if head.bias is None else (
            head.bias.detach().float().clone().requires_grad_(head.bias.requires_grad)
        )
        parameters = [p for p in (weight, bias) if p is not None and p.requires_grad]
        prediction = F.linear(pooled, weight, bias)
        if valid is not None:
            prediction = prediction * valid
        prediction = F.normalize(prediction, p=2, dim=-1)
        losses = cosine_similarity_loss(prediction, target.detach(), reduction="none")
        stats = torch.zeros(num_aspects, 3, device=hidden.device, dtype=torch.float64)
        present = aspect_ids.detach().unique().tolist()
        for index, aspect in enumerate(present):
            selected = aspect_ids == aspect
            count = selected.sum()
            mean_loss = loss_weight * losses[selected].mean()
            if parameters:
                gradients = torch.autograd.grad(
                    mean_loss, parameters, retain_graph=index < len(present) - 1,
                )
                norm = torch.stack([g.detach().norm(2) for g in gradients]).norm(2)
            else:
                norm = losses.new_zeros(())
            stats[aspect, 0] = count
            stats[aspect, 1] = count * norm.double()
            stats[aspect, 2] = 1
    return stats.detach()
