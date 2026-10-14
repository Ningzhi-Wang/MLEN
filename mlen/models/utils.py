import torch
import torch.distributed as dist

from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F


def cosine_similarity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *args,
    reduction: str = "mean",
    eps: float = 1e-8,
    **kwargs
) -> torch.Tensor:
    """
    Cosine loss: 1 - cosine_similarity.

    Computation and returned loss are always float32.
    Embedding dimension is assumed to be the last dimension.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    # Disabling autocast prevents normalization and dot products from being
    # silently executed in float16/bfloat16.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = prediction.float()
        target = target.float()

        prediction = F.normalize(prediction, dim=-1, eps=eps)
        target = F.normalize(target, dim=-1, eps=eps)

        loss = 1.0 - (prediction * target).sum(dim=-1)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

    return loss


def whitened_cosine_similarity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    aspect_ids: torch.Tensor,
    *args,
    aspect_means: torch.Tensor,
    whitening_matrices: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """Cosine distance after a fixed, target-derived per-aspect ZCA transform.

    Means [A, D] and matrices [A, D, D] are fitted to L2-normalized targets.
    Both sides use (normalize(x) - mean[a]) @ W[a], followed by normalization.
    Targets and statistics are fixed; only predictions receive gradients.
    Targets at the aspect mean have no residual direction, so use raw cosine
    for those samples. Empty batches return a graph-connected zero mean.
    Every sample uses its own aspect's statistics. Extra arguments from the
    shared MLEN loss call are ignored; this loss adds no RVQ term or key weight.
    """
    aspect_ids = torch.as_tensor(
        aspect_ids, device=prediction.device, dtype=torch.long
    ).flatten()

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = F.normalize(prediction.float(), dim=-1, eps=1e-12)
        target = F.normalize(target.detach().float(), dim=-1, eps=1e-12)
        means = aspect_means.detach()[aspect_ids].float()
        matrices = whitening_matrices.detach()[aspect_ids].float()
        target_residual = target - means
        whitened_prediction = torch.bmm(
            (prediction - means).unsqueeze(1), matrices
        ).squeeze(1)
        whitened_target = torch.bmm(
            target_residual.unsqueeze(1), matrices
        ).squeeze(1)
        whitened_loss = 1.0 - (
            F.normalize(whitened_prediction, dim=-1, eps=eps)
            * F.normalize(whitened_target, dim=-1, eps=eps)
        ).sum(dim=-1)
        raw_loss = 1.0 - (prediction * target).sum(dim=-1)
        loss = torch.where(
            target_residual.norm(dim=-1) > eps, whitened_loss, raw_loss
        )
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.sum() / max(loss.numel(), 1)
        return loss


def gather_embeddings(x):
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return x

    return torch.cat(all_gather(x), dim=0)


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


def multi_positive_infonce(pred, target, temperature=0.07, target_mse_atol=1e-6, *args, **kwargs):
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


def cosine_kernel_infonce(
    pred,
    target,
    temperature=0.05,
    target_temperature=0.03,
    *args,
    **kwargs,
):
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)

    logits = pred @ target.T / temperature
    target_sim = target @ target.T
    soft_targets = F.softmax(target_sim / target_temperature, dim=1).detach()

    def direction_loss(scores):
        log_probs = F.log_softmax(scores, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    return (
        direction_loss(logits)
        + direction_loss(logits.T)
    ) / 2


DEFAULT_ASPECT_COSINE_LOSS_WEIGHTS = (
    1.0000,  # genre
    2.9044,  # key
    0.9105,  # tempo
    3.8277,  # vocality
    2.5636,  # tuning frequency
    3.2315,  # tuning stability
    1.1576,  # instrumentation
    1.3050,  # mood
)


def aspect_scaled_cosine_loss(
    pred,
    target,
    aspect_ids,
    *args,
    aspect_weights=DEFAULT_ASPECT_COSINE_LOSS_WEIGHTS,
    **kwargs,
):
    """Cosine distance with one multiplicative loss weight per aspect.

    The default weights are inverse occurrence-weighted tangent RMS standard
    deviations, relative to genre. Because aspects are sampled uniformly, the
    configured mean weight is used as a fixed normalizer. This preserves the
    relative scaling even in batches that do not contain every aspect.
    """
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim < 2:
        raise ValueError("pred and target must include a batch dimension")

    aspect_ids = torch.as_tensor(
        aspect_ids,
        device=pred.device,
        dtype=torch.long,
    ).flatten()
    if aspect_ids.numel() != pred.size(0):
        raise ValueError(
            "aspect_ids must contain one ID per sample, got "
            f"{aspect_ids.numel()} IDs for batch size {pred.size(0)}"
        )

    aspect_weights = torch.as_tensor(
        aspect_weights,
        device=pred.device,
        dtype=torch.float32,
    ) * 4
    if aspect_weights.ndim != 1 or aspect_weights.numel() == 0:
        raise ValueError(
            "aspect_weights must be a non-empty one-dimensional sequence"
        )
    if not torch.isfinite(aspect_weights).all():
        raise ValueError("aspect_weights must be finite")
    if not (aspect_weights > 0).all():
        raise ValueError("aspect_weights must be positive")
    if (aspect_ids < 0).any() or (
        aspect_ids >= aspect_weights.numel()
    ).any():
        raise ValueError(
            "aspect_ids contains an ID without a configured aspect weight"
        )

    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    per_sample_loss = 1.0 - (pred * target).sum(dim=-1)
    sample_weights = aspect_weights[aspect_ids]

    return (
        per_sample_loss * sample_weights
    ).mean() 


def distributed_multi_positive_infonce(pred, target, *args, **kwargs):
    global_pred = gather_embeddings(pred)
    global_target = gather_embeddings(target.detach())

    return multi_positive_infonce(
        global_pred,
        global_target,
        **kwargs,
    )


def distributed_cosine_kernel_infonce(pred, target, *args, **kwargs):
    global_pred = gather_embeddings(pred)
    global_target = gather_embeddings(target.detach())

    return cosine_kernel_infonce(
        global_pred,
        global_target,
        **kwargs,
    )


DEFAULT_ASPECT_TARGET_TEMPERATURES = (
    0.10,  # genre
    0.01,  # key
    0.03,  # tempo
    0.01,  # vocality
    0.01,  # tuning frequency
    0.01,  # tuning stability
    0.03,  # instrumentation
    0.10,  # mood
)


def distributed_aspect_temperature_cosine_kernel_infonce(
    pred,
    target,
    aspect_ids,
    *args,
    aspect_target_temperatures=DEFAULT_ASPECT_TARGET_TEMPERATURES,
    temperature=0.05,
    **kwargs,
):
    """Cosine-kernel InfoNCE with one target temperature per aspect.

    The prediction temperature remains shared. Target temperatures are
    selected per query row while all aspects remain in the same global
    comparison matrix, preserving cross-aspect target relationships.
    """
    if pred.size(0) != target.size(0):
        raise ValueError(
            "pred and target must have the same local batch size, got "
            f"{pred.size(0)} and {target.size(0)}"
        )

    aspect_ids = torch.as_tensor(
        aspect_ids,
        device=pred.device,
        dtype=torch.long,
    ).flatten()
    if aspect_ids.numel() != pred.size(0):
        raise ValueError(
            "aspect_ids must contain one ID per local sample, got "
            f"{aspect_ids.numel()} IDs for batch size {pred.size(0)}"
        )

    aspect_target_temperatures = torch.as_tensor(
        aspect_target_temperatures,
        device=pred.device,
        dtype=torch.float32,
    )
    if aspect_target_temperatures.ndim != 1:
        raise ValueError("aspect_target_temperatures must be one-dimensional")
    if aspect_target_temperatures.numel() == 0:
        raise ValueError("aspect_target_temperatures must not be empty")
    if not torch.isfinite(aspect_target_temperatures).all():
        raise ValueError("aspect_target_temperatures must be finite")
    if not (aspect_target_temperatures > 0).all():
        raise ValueError("aspect_target_temperatures must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    global_pred = gather_embeddings(pred)
    global_target = gather_embeddings(target.detach())
    global_aspect_ids = gather_embeddings(aspect_ids.detach())

    if global_aspect_ids.numel() != global_pred.size(0):
        raise ValueError(
            "Gathered aspect_ids and embeddings have different batch sizes: "
            f"{global_aspect_ids.numel()} and {global_pred.size(0)}"
        )
    if (global_aspect_ids < 0).any():
        raise ValueError("aspect_ids must be non-negative")
    if (
        global_aspect_ids >= aspect_target_temperatures.numel()
    ).any():
        raise ValueError(
            "aspect_ids contains an ID without a configured target "
            "temperature"
        )

    global_pred = F.normalize(global_pred.float(), dim=-1)
    global_target = F.normalize(global_target.float(), dim=-1)

    logits = global_pred @ global_target.T / temperature
    target_sim = global_target @ global_target.T
    row_temperatures = aspect_target_temperatures[
        global_aspect_ids
    ].unsqueeze(1)
    soft_targets = F.softmax(
        target_sim / row_temperatures,
        dim=1,
    ).detach()

    def direction_loss(scores):
        log_probs = F.log_softmax(scores, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()

    return (
        direction_loss(logits)
        + direction_loss(logits.T)
    ) / 2


def classification_loss(
    pred,
    target,
    aspect_ids,
    aspect_class_id,
    probing_heads,
    multi_label_aspects=(0, 6, 7),
    *args,
    **kwargs,
):
    """
    pred:
        Shared representation [B, D].

    aspect_ids:
        Selected aspect per sample [B].

    aspect_class_id:
        Class IDs [B, L], padded with -1.

    probing_heads:
        One Linear(D, C_aspect) head per aspect.
    """
    del target  # Not used by conventional classification.

    aspect_ids = torch.as_tensor(
        aspect_ids,
        device=pred.device,
        dtype=torch.long,
    ).flatten()

    class_ids = torch.as_tensor(
        aspect_class_id,
        device=pred.device,
        dtype=torch.long,
    )

    if class_ids.ndim == 1:
        class_ids = class_ids.unsqueeze(1)

    if class_ids.size(0) != pred.size(0):
        raise ValueError(
            "aspect_class_id must contain one row per sample"
        )

    multi_label_aspects = set(multi_label_aspects)

    # Different heads have different output widths, so retain a list.
    logits_by_aspect = [
        head(pred)
        for head in probing_heads
    ]

    # Connect every head to the graph, including heads whose aspect
    # is absent from this batch. This avoids unused-parameter issues
    # under DDP.
    graph_anchor = sum(
        logits.sum() * 0.0
        for logits in logits_by_aspect
    )

    aspect_losses = []

    for aspect_id, all_logits in enumerate(logits_by_aspect):
        sample_mask = aspect_ids.eq(aspect_id)

        if not sample_mask.any():
            continue

        logits = all_logits[sample_mask]
        labels = class_ids[sample_mask]
        valid = labels.ge(0)

        if not valid.any(dim=1).all():
            raise ValueError(
                f"Aspect {aspect_id} has samples without labels"
            )

        number_of_classes = logits.size(-1)
        selected_ids = labels[valid]

        if (selected_ids >= number_of_classes).any():
            raise ValueError(
                f"Aspect {aspect_id} received class ID "
                f"{int(selected_ids.max())}, but its head has "
                f"{number_of_classes} outputs"
            )

        if aspect_id in multi_label_aspects:
            targets = logits.new_zeros(logits.shape)

            row_indices = (
                torch.arange(
                    labels.size(0),
                    device=labels.device,
                )
                .unsqueeze(1)
                .expand_as(labels)
            )

            targets[
                row_indices[valid],
                labels[valid],
            ] = 1.0

            pos_weights = None
            if aspect_id == 0:
                pos_weights = logits.new_full(
                    (logits.size(-1),),
                    412.6587,
                )


            aspect_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weights,
            )

        else:
            label_counts = valid.sum(dim=1)

            if not torch.all(label_counts == 1):
                raise ValueError(
                    f"Single-label aspect {aspect_id} received "
                    "multiple class IDs"
                )

            # Exactly one valid class ID per row.
            targets = labels[valid]

            aspect_loss = F.cross_entropy(
                logits,
                targets,
            )

        aspect_losses.append(aspect_loss)

    if not aspect_losses:
        raise ValueError(
            "The batch contains no valid classification labels"
        )

    # Equal weighting for each aspect represented in the batch.
    return torch.stack(aspect_losses).mean() + graph_anchor


def aspect_specific_loss(
    pred,
    target,
    aspect_ids,
    rvq_logits,
    rvq_targets,
    masked_indices,
    target_aspect=1,
    cond_weighting=5,
    *args,
    **kwargs,
):
    cond_idx = aspect_ids == target_aspect
    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred= pred[cond_idx].float()
        target = target[cond_idx].float()

        pred= F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        cond_loss = 1.0 - (pred * target).sum(dim=-1)

    positions = masked_indices.to(aspect_ids.device)
    positions = positions[~cond_idx[positions[:, 0]]]
    batch_idx, time_idx = positions.unbind(dim=1)

    rvq_losses = 0.
    if positions.size(0) > 0:
        for key in rvq_logits:
            selected_logits = rvq_logits[key][batch_idx, time_idx]
            selected_targets = rvq_targets[key][batch_idx, time_idx]
            rvq_loss = F.cross_entropy(
                selected_logits.reshape(-1, 1024),
                selected_targets.reshape(-1).long(),
            )
            rvq_losses += rvq_loss

    return cond_loss.sum() / max(cond_loss.numel(), 1) * cond_weighting + rvq_losses


def whitened_or_rvq_loss(
    prediction,
    target,
    aspect_ids,
    rvq_logits,
    rvq_targets,
    masked_indices,
    *args,
    aspect_means,
    whitening_matrices,
    rvq_aspects=(-1,),
    whitening_weight=1.0,
    rvq_weight=0.1,
    reduction="mean",
    eps=1e-6,
    components=None,
    **kwargs,
):
    """Per-sample loss: RVQ for rvq_aspects, whitened cosine for other samples.

    rvq_aspects: one aspect ID or a sequence of IDs; -1 is the zeroed
    (unconditional) sample produced by aspect dropout.
    rvq_targets[key]: [B, T, K], already reshaped by get_loss().

    Each sample's value is ``whitening_weight`` times its whitened cosine
    distance, or ``rvq_weight`` times the cross-entropy averaged over its
    masked positions and codebooks and summed over logits keys. An RVQ sample
    without masked positions contributes zero. ``reduction="none"`` returns a
    [B] tensor so MLEN can apply per-label weights and per-aspect logging;
    ``"mean"`` divides by B, so the RVQ branch scales with the fraction of
    RVQ samples in the batch; ``"sum"`` sums. When ``components`` is a dict it
    receives the detached, unweighted mean of each branch under ``"whitened"``
    and ``"rvq"`` (zero for an empty branch).
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    aspect_ids = torch.as_tensor(
        aspect_ids, device=prediction.device, dtype=torch.long
    ).flatten()
    batch = prediction.size(0)
    if aspect_ids.numel() != batch:
        raise ValueError(
            "aspect_ids must contain one ID per sample, got "
            f"{aspect_ids.numel()} IDs for batch size {batch}"
        )
    rvq_aspects = torch.as_tensor(
        rvq_aspects, device=prediction.device, dtype=torch.long
    ).flatten()

    use_rvq = torch.isin(aspect_ids, rvq_aspects)
    whitening_rows = (~use_rvq).nonzero(as_tuple=True)[0]
    zeros = prediction.new_zeros(batch, dtype=torch.float32)

    # The helper handles an empty subset; never run it on RVQ rows, whose
    # aspect id (-1) would silently select the last aspect's statistics.
    whitened = whitened_cosine_similarity_loss(
        prediction[whitening_rows],
        target[whitening_rows],
        aspect_ids[whitening_rows],
        aspect_means=aspect_means,
        whitening_matrices=whitening_matrices,
        reduction="none",
        eps=eps,
    )
    whitened_per_sample = zeros.index_put((whitening_rows,), whitened)

    # Keep masked positions belonging to RVQ-selected samples.
    positions = torch.as_tensor(
        masked_indices, device=prediction.device, dtype=torch.long
    ).reshape(-1, 2)
    positions = positions[use_rvq[positions[:, 0]]]
    indices = tuple(positions.t())
    rows = positions[:, 0]
    counts = zeros.index_add(0, rows, torch.ones_like(rows, dtype=torch.float32))

    rvq_per_sample = zeros
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        for key, logits in rvq_logits.items():
            # Preserve a zero gradient when this branch has no samples.
            rvq_per_sample = rvq_per_sample + logits[..., :0].float().sum()

            if positions.size(0) == 0:
                continue

            tokens = rvq_targets[key]
            if tokens.ndim != 3 or tokens.shape[:2] != logits.shape[:2]:
                raise ValueError(
                    f"{key}: expected RVQ targets shaped [B, T, K]"
                )

            num_codebooks = tokens.size(-1)
            if num_codebooks == 0 or logits.size(-1) % num_codebooks:
                raise ValueError(
                    f"{key}: logits width must be divisible by K"
                )

            codebook_size = logits.size(-1) // num_codebooks

            position_losses = F.cross_entropy(
                logits[indices].reshape(-1, codebook_size).float(),
                tokens[indices].reshape(-1).long(),
                reduction="none",
            ).view(-1, num_codebooks).mean(dim=-1)
            rvq_per_sample = rvq_per_sample.index_add(0, rows, position_losses)
    rvq_per_sample = rvq_per_sample / counts.clamp_min(1.0)

    if components is not None:
        components["whitened"] = (
            whitened.detach().mean() if whitened.numel() else zeros.new_zeros(())
        )
        components["rvq"] = (
            rvq_per_sample[use_rvq].detach().mean()
            if use_rvq.any() else zeros.new_zeros(())
        )

    loss = whitening_weight * whitened_per_sample + rvq_weight * rvq_per_sample
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / max(batch, 1)
    return loss


def condition_to_aspect_ids(cond):
    aspect_ids = cond.argmax(dim=-1)
    return aspect_ids.masked_fill(cond.eq(0).all(dim=-1), -1)


def scaled_log1p(loss, reference, alpha=1.0):
    return torch.log1p(
        alpha * loss / reference
    ) / alpha
