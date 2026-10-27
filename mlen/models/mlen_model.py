import torch
from einops import rearrange
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F
from muq.muq.models.muq_model import MuQModel

from .conformer import CondConformerEncoder
from .pooling import AspectAttentionPool
from .masking import mask_waveform
from .utils import (
    multi_positive_infonce, 
    distributed_multi_positive_infonce, 
    distributed_cosine_kernel_infonce,
    distributed_aspect_temperature_cosine_kernel_infonce,
    classification_loss,
    aspect_scaled_cosine_loss,
    cosine_similarity_loss,
    scaled_log1p,
    aspect_specific_loss,
    whitened_cosine_similarity_loss,
    whitened_or_rvq_loss,
    condition_to_aspect_ids,
)


# Conditional losses selectable by name via MLENConfig.cond_loss.
# "raw_centered_cosine_loss" is resolved to the MLENModel method.
COND_LOSSES = {
    fn.__name__: fn
    for fn in (
        classification_loss,
        aspect_scaled_cosine_loss,
        aspect_specific_loss,
        whitened_cosine_similarity_loss,
        whitened_or_rvq_loss,
        cosine_similarity_loss,
        distributed_cosine_kernel_infonce,
        distributed_aspect_temperature_cosine_kernel_infonce,
    )
}
WHITENED_COND_LOSSES = {"whitened_cosine_similarity_loss", "whitened_or_rvq_loss"}
# Losses that accept reduction="none" and return one value per sample, which
# lets MLEN log the conditional loss per aspect.
PER_SAMPLE_COND_LOSSES = {
    "cosine_similarity_loss",
    "whitened_cosine_similarity_loss",
    "whitened_or_rvq_loss",
}
# Losses that give unconditional tracks (all-zero cond, aspect id -1) their
# own term. Other losses never see such tracks in the pair format.
UNCONDITIONAL_COND_LOSSES = {"whitened_or_rvq_loss"}
COND_POOLINGS = ("mean", "cls", "attention")


class MLENModel(MuQModel):
    def __init__(
            self, 
            cond_dim=None, 
            cond_target_dim=512, 
            cond_pooling="mean",
            cond_loss="cosine_similarity_loss",
            flash_attention=True, 
            aspect_centroids=None,
            centroid_weight=0.0,
            aspect_whitening_means=None,
            aspect_whitening_matrices=None,
            cond_label_weights=None,
            cond_loss_kwargs=None,
            cond_pool_heads=8,
            **kwargs
        ):
        super().__init__( **kwargs)
        config = EasyDict(kwargs.get("w2v2_config", {}))
        config.num_hidden_layers = kwargs.get("encoder_depth", 12)
        config.hidden_size = kwargs.get("encoder_dim", 1024)
        if cond_pooling not in COND_POOLINGS:
            raise ValueError(
                f"cond_pooling must be one of {COND_POOLINGS}, got {cond_pooling!r}"
            )
        self.cond_pooling = cond_pooling
        self.cond_loss_components = {}
        # Extra keyword arguments forwarded to every cond loss call; they
        # override the shared defaults built in forward() (MLENConfig.cond_loss_kwargs).
        self.cond_loss_kwargs = dict(cond_loss_kwargs or {})
        # Per-codebook loss/accuracy of the last get_loss call (detached).
        self.rvq_loss_components = {}
        if (aspect_whitening_means is None) != (aspect_whitening_matrices is None):
            raise ValueError("Provide both aspect whitening means and matrices")
        if aspect_whitening_means is not None:
            if cond_dim is None:
                raise ValueError("Whitening requires a conditional model")
            aspect_whitening_means = torch.as_tensor(
                aspect_whitening_means, dtype=torch.float32
            ).detach().clone()
            aspect_whitening_matrices = torch.as_tensor(
                aspect_whitening_matrices, dtype=torch.float32
            ).detach().clone()
            if (
                tuple(aspect_whitening_means.shape) != (cond_dim, cond_target_dim)
                or tuple(aspect_whitening_matrices.shape)
                != (cond_dim, cond_target_dim, cond_target_dim)
            ):
                raise ValueError(
                    "Whitening statistics must match [cond_dim, cond_target_dim] "
                    "and [cond_dim, cond_target_dim, cond_target_dim]"
                )
            if not (
                torch.isfinite(aspect_whitening_means).all()
                and torch.isfinite(aspect_whitening_matrices).all()
            ):
                raise ValueError("Whitening statistics contain non-finite values")
        self.register_buffer("aspect_whitening_means", aspect_whitening_means)
        self.register_buffer("aspect_whitening_matrices", aspect_whitening_matrices)
        # Per-label loss weights indexed by text_index (see the label-weight setup
        # cell); None means every sample weighs 1.
        if cond_label_weights is not None:
            cond_label_weights = torch.as_tensor(cond_label_weights, dtype=torch.float32).detach().clone().flatten()
            if not torch.isfinite(cond_label_weights).all() or (cond_label_weights <= 0).any():
                raise ValueError("cond_label_weights must be finite and positive")
        self.register_buffer("cond_label_weights", cond_label_weights)
        if cond_dim is not None:
            self.cond = True
            self.cond_proj = nn.Linear(kwargs["encoder_dim"], cond_target_dim)
            if cond_pooling == "attention":
                # One query per aspect; the shared cond_proj decodes each pooled vector.
                self.aspect_pool = AspectAttentionPool(
                    cond_dim, kwargs["encoder_dim"], num_heads=cond_pool_heads
                )
            self.cond_loss = self.resolve_cond_loss(cond_loss)
            if cond_label_weights is not None and not self.cond_loss_is_per_sample:
                raise ValueError(f"cond_label_weights need a per-sample loss, not {cond_loss!r}")
            if aspect_centroids is not None:
                expected_shape = (cond_dim, cond_target_dim)
                if tuple(aspect_centroids.shape) != expected_shape:
                    raise ValueError(
                        f"aspect_centroids must have shape {expected_shape}, "
                        f"got {tuple(aspect_centroids.shape)}"
                    )

                if not torch.isfinite(aspect_centroids).all():
                    raise ValueError("aspect_centroids contains non-finite values")

                self.register_buffer("aspect_centroids", aspect_centroids, persistent=True)
                self.centroid_weight = centroid_weight
            else:
                self.aspect_centroids = None
                self.centroid_weight = 0.0

            config.cond_dim = cond_dim
            config.flash_attention = flash_attention
            self.conformer = CondConformerEncoder(config)
        else:
            self.cond = False

    def resolve_cond_loss(self, cond_loss):
        if cond_loss == "raw_centered_cosine_loss":
            return self.raw_centered_cosine_loss
        if cond_loss not in COND_LOSSES:
            options = sorted([*COND_LOSSES, "raw_centered_cosine_loss"])
            raise ValueError(f"Unknown cond_loss {cond_loss!r}; choose from {options}")
        if cond_loss in WHITENED_COND_LOSSES and self.aspect_whitening_means is None:
            raise ValueError(f"cond_loss {cond_loss!r} requires aspect whitening statistics")
        return COND_LOSSES[cond_loss]

    def masking(self, x, attention_mask=None):
        # As in MuQ, padding is filtered downstream rather than here.
        return mask_waveform(
            x, mask_hop=self.mask_hop, hop_length=self.hop_length,
            mask_prob=self.mask_prob,
        )

    @torch.no_grad()
    def preprocessing(self, x, features):
        """extract classic audio features"""
        # check precision
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            precision = 16
        else:
            precision = 32

        out = {}
        for key in features:
            layer = getattr(self, "preprocessor_%s" % key)
            layer.to(x.device)
            dtype = x.dtype
            out[key] = layer(x.float())[..., :-1]
            if precision == 16:
                out[key] = out[key].half()
            if out[key].dtype != dtype:
                out[key] = out[key].to(dtype=dtype)
        return out

    def get_targets(self, x, label=None):
        if label is None:
            return super().get_targets(x, label=label)

        x = self.preprocessing(x, features=self.features)
        x = self.normalize(x)
        x = self.rearrange(x)
        melspec = x['melspec_2048']

        if label.dim() == 2:
            target = label.long()
        elif label.dim() == 3:
            target = rearrange(label, "b n s -> b (n s)").long()
        else:
            raise ValueError(f"Expected label with 2 or 3 dims, got shape {tuple(label.shape)}")
        return {'melspec_2048': target}, melspec

    # FRONTEND_CHANNELS_LAST: layout-only opt-in; checkpoint keys/shapes are unchanged.
    def enable_frontend_channels_last(self):
        """Call after weight loading and before creating the optimizer or DDP."""
        if not self.cond:
            raise ValueError("Frontend channels-last requires the conditional MLEN encoder")
        self.conv.conv.to(memory_format=torch.channels_last)
        self.frontend_channels_last = True

    def encoder_cond(self, x, cond, *, attention_mask=None, is_features_only=False):
        """2-layer conv + w2v-conformer"""
        if getattr(self, "frontend_channels_last", False):
            if x.ndim == 3:
                x = x.unsqueeze(1)
            x = x.to(memory_format=torch.channels_last)
        x = self.conv(x)
        mask_indices = None
        cls_tokens = int(self.cond_pooling == "cls")

        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            skip_n = int(attention_mask.size(-1) / x.size(1))
            attention_mask = attention_mask[:, ::skip_n]
            attention_mask = attention_mask[:, :x.size(1)]
            attention_mask = F.pad(attention_mask, (cls_tokens, 0), value=True)

        if cls_tokens:
            cls_token = self.cls_token.to(dtype=x.dtype)
            cls_token = cls_token.view(1, 1, -1).expand(x.size(0), -1, -1)
            x = torch.cat([cls_token, x], dim=1)

        out = self.conformer(
            x, cond=cond, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden_emb = out["hidden_states"]
        last_emb = out["last_hidden_state"]
        logits = self.linear(self.get_frame_hidden(last_emb))
        interval = self.codebook_size
        logits = {
            key: logits[:, :, i * interval : (i + 1) * interval]
            for i, key in enumerate(self.features)
        }
        return logits, hidden_emb, mask_indices

    def encoder(self, x, *args, attention_mask=None, is_features_only=False):
        if self.cond:
            assert len(args) > 0 and args[0] is not None, "cond is required for MLENModel when self.cond is True"
            return self.encoder_cond(x, args[0], attention_mask=attention_mask, is_features_only=is_features_only)
        else:
            return super().encoder(x, attention_mask=attention_mask, is_features_only=is_features_only)

    def get_predictions(self, x, *args, mask=None, attention_mask=None, return_new_mask=False, is_features_only=False):
        # preprocessing
        x = self.preprocessing(x, features=["melspec_2048"])
        x = self.normalize(x)

        # encoding
        logits, hidden_emb, new_mask = self.encoder(x["melspec_2048"], *args, attention_mask=attention_mask, is_features_only=is_features_only)

        if return_new_mask:
            return logits, hidden_emb, mask if new_mask is None else new_mask
        else:
            return logits, hidden_emb

    def get_latent(self, x, *args, layer_ix=12):
        _, hidden_states = self.get_predictions(x, *args)
        emb = self.get_frame_hidden(hidden_states[layer_ix])
        return emb

    def get_frame_hidden(self, hidden_emb):
        if self.cond and self.cond_pooling == "cls":
            return hidden_emb[:, 1:]
        return hidden_emb

    def get_loss(self, logits, target_tokens, masked_indices):
        losses = {}
        accuracies = {}
        for key in logits.keys():
            if not self.use_rvq_like_target:
                masked_logits = logits[key][tuple(masked_indices.t())]
                masked_tokens = target_tokens[key][tuple(masked_indices.t())]
            else:
                Batch, SeqLen, N_Codebook_x_CodebookSize = logits[key].shape
                Batch, N_Codebook_x_SeqLen = target_tokens[key].shape 
                N_Codebook = int(N_Codebook_x_SeqLen // SeqLen)
                target_tokens[key] = rearrange(target_tokens[key], "b (n s) -> b s n", n=N_Codebook) # Batch, SeqLen=750, N_Codebook=4
                masked_logits = logits[key][tuple(masked_indices.t())] 
                masked_tokens = target_tokens[key][tuple(masked_indices.t())] 
                masked_logits = rearrange(masked_logits, "b (n c) -> (b n) c", n=N_Codebook)
                masked_tokens = rearrange(masked_tokens, "b n -> (b n)", n=N_Codebook) 

            if masked_tokens.numel() == 0:
                losses[key] = masked_logits.float().sum()
                accuracies[key] = logits[key].new_zeros(
                    (), dtype=torch.float32
                )
                if self.use_rvq_like_target:
                    self._set_rvq_loss_components(key, None, None, N_Codebook)
                continue
            losses[key] = self.loss(masked_logits, masked_tokens)
            accuracies[key] = (
                torch.sum(masked_logits.argmax(-1) == masked_tokens)
                / masked_tokens.numel()
            )
            if self.use_rvq_like_target:
                self._set_rvq_loss_components(key, masked_logits, masked_tokens, N_Codebook)
        return losses, accuracies

    @torch.no_grad()
    def _set_rvq_loss_components(self, key, masked_logits, masked_tokens, num_codebooks):
        """Per-codebook CE and accuracy; keys are always present so DDP logging stays in sync."""
        if not hasattr(self, "rvq_loss_components"):
            self.rvq_loss_components = {}
        if masked_logits is None:
            zero = torch.zeros((), dtype=torch.float32)
            for i in range(num_codebooks):
                self.rvq_loss_components[f"{key}_codebook{i}_loss"] = zero
                self.rvq_loss_components[f"{key}_codebook{i}_acc"] = zero
            return
        # Rows are ordered (b n) by the rearrange in get_loss.
        per_row = F.cross_entropy(masked_logits.float(), masked_tokens, reduction="none").view(-1, num_codebooks)
        correct = (masked_logits.argmax(-1) == masked_tokens).float().view(-1, num_codebooks)
        for i in range(num_codebooks):
            self.rvq_loss_components[f"{key}_codebook{i}_loss"] = per_row[:, i].mean().detach()
            self.rvq_loss_components[f"{key}_codebook{i}_acc"] = correct[:, i].mean().detach()
    
    @staticmethod
    def _frame_attention_mask(attention_mask, num_frames):
        """Subsample a sample-level attention mask to frames (True = keep)."""
        attention_mask = attention_mask.bool()
        skip_n = int(attention_mask.size(-1) / num_frames)
        return attention_mask[:, ::skip_n][:, :num_frames]

    def get_cond_prediction(self, last_hidden_emb, attention_mask):
        """Normalised predictions: [B, D] for mean/cls, [B, A, D] for attention."""
        if self.cond_pooling == "cls":
            cond_pred = self.cond_proj(last_hidden_emb[:, 0])
            return F.normalize(cond_pred, p=2, dim=-1)

        if self.cond_pooling == "attention":
            frames = last_hidden_emb
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~self._frame_attention_mask(attention_mask, frames.size(1))
            pooled = self.aspect_pool(frames, key_padding_mask)
            return F.normalize(self.cond_proj(pooled), p=2, dim=-1)

        if attention_mask is not None:
            attention_mask = self._frame_attention_mask(attention_mask, last_hidden_emb.size(1))

            frame_mask = attention_mask.unsqueeze(-1).float()
            cond_pred = self.cond_proj(last_hidden_emb)
            cond_pred = (cond_pred * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp_min(1.0)
        else:
            cond_pred = self.cond_proj(last_hidden_emb.mean(dim=1))
        return F.normalize(cond_pred, p=2, dim=-1)

    def forward(self, x, *args, attention_mask=None, label=None, cond_label=None, feature_only=False, aspect_class_id=None, cond_label_index=None):
        # get target feature tokens
        if self.training or not feature_only:
            target_tokens, melspec = self.get_targets(x, label=label)
            # masking
            x, masked_indices = self.masking(x, attention_mask=attention_mask)
        else:
            masked_indices = None

        # forward
        logits, hidden_emb, masked_indices = self.get_predictions(x, *args, mask=masked_indices, attention_mask=attention_mask, return_new_mask=True)
        
        if not self.training and feature_only:
            return logits, hidden_emb

        if attention_mask is not None:
            valid_audio = attention_mask.to(
                device=x.device, dtype=torch.bool
            )
        
            # Match encoder_cond() and get_cond_prediction().
            num_tokens = next(iter(logits.values())).size(1)
            skip_n = int(valid_audio.size(-1) / num_tokens)
            valid_tokens = valid_audio[:, ::skip_n][:, :num_tokens]
        
            masked_indices = masked_indices.to(
                device=x.device, dtype=torch.long
            )
            keep = valid_tokens[
                masked_indices[:, 0],
                masked_indices[:, 1],
            ]
            masked_indices = masked_indices[keep]

        # get loss
        losses, accuracies = self.get_loss(logits, target_tokens, masked_indices)

        if self.recon_loss_ratio:
            frame_hidden = self.get_frame_hidden(hidden_emb[-1])
            losses["recon_loss"] = self.get_recon_loss(frame_hidden, melspec, masked_indices) * self.recon_loss_ratio
        
        if self.cond:
            assert cond_label is not None, "cond_label is required for MLENModel in conditional mode"
            pair_rows, aspect_ids, pair_target, pair_index = self._cond_pairs(
                args[0], cond_label, cond_label_index
            )
            cond_pred = self.get_cond_prediction(hidden_emb[-1], attention_mask)
            if cond_pred.dim() == 2:
                # One vector per track serves every pair of that track.
                cond_pred = cond_pred[pair_rows]
            else:
                cond_pred = cond_pred[pair_rows, aspect_ids.clamp_min(0)]
            loss_args = (cond_pred, pair_target, aspect_ids, logits, target_tokens, masked_indices)
            loss_kwargs = dict(
                target_aspect=1, cond_weighting=50,
                aspect_means=self.aspect_whitening_means,
                whitening_matrices=self.aspect_whitening_matrices,
                pair_rows=pair_rows,
                # Losses that report branch values write into this dict.
                components=self.cond_loss_components,
            )
            loss_kwargs.update(self.cond_loss_kwargs)
            if self.cond_loss_is_per_sample:
                per_sample = self.cond_loss(*loss_args, reduction="none", **loss_kwargs)
                sample_weights = self.get_cond_sample_weights(pair_index, per_sample)
                if sample_weights is None:
                    losses['cond_loss'] = per_sample.sum() / max(per_sample.numel(), 1)
                else:
                    losses['cond_loss'] = (sample_weights * per_sample).sum() / max(per_sample.numel(), 1)
                # Diagnostics stay unweighted so per-aspect logs are comparable across runs.
                self.cond_loss_per_sample = per_sample.detach()
            else:
                losses['cond_loss'] = self.cond_loss(*loss_args, **loss_kwargs)
                self.cond_loss_per_sample = None
            # One entry per (track, aspect) pair, for logging and validation plots.
            self.cond_aspect_ids = aspect_ids.detach()
            self.cond_pair_rows = pair_rows.detach()
            self.cond_pred = cond_pred.detach()
            self.cond_pair_target = pair_target.detach()

        return logits, hidden_emb, losses, accuracies

    def _cond_pairs(self, cond, cond_label, cond_label_index):
        """Supervised (track, aspect) pairs: rows, aspect ids, targets, label indices.

        cond_label [B, D] (one aspect per track): the aspect is the cond's
        argmax, or -1 for an all-zero cond; pairs are the rows themselves.
        cond_label [B, A, D] (a target for every aspect): the pairs are the
        nonzero entries of the multi-hot cond. Tracks whose cond is all zero
        become aspect -1 pairs with a zero target when the cond loss has a
        term for unconditional tracks, and are left out otherwise.
        """
        if cond_label.dim() == 2:
            rows = torch.arange(cond.size(0), device=cond.device)
            return rows, condition_to_aspect_ids(cond), cond_label, cond_label_index
        if cond_label.dim() != 3 or cond_label.shape[:2] != cond.shape:
            raise ValueError(
                "cond_label must be [B, D] or [B, A, D] matching the [B, A] cond, got "
                f"{tuple(cond_label.shape)} for cond {tuple(cond.shape)}"
            )
        rows, aspect_ids = cond.nonzero(as_tuple=True)
        if self.cond_loss_handles_unconditional:
            unconditional = cond.eq(0).all(dim=-1).nonzero(as_tuple=True)[0]
            rows = torch.cat([rows, unconditional])
            aspect_ids = torch.cat([aspect_ids, torch.full_like(unconditional, -1)])
        lookup = aspect_ids.clamp_min(0)
        conditional = aspect_ids >= 0
        target = cond_label[rows, lookup].masked_fill(~conditional.unsqueeze(-1), 0)
        index = None
        if cond_label_index is not None:
            index = torch.as_tensor(cond_label_index, device=cond.device).long()
            index = index[rows, lookup].masked_fill(~conditional, -1)
            if (conditional & (index < 0)).any():
                raise ValueError("cond activates an aspect the track has no target for")
        return rows, aspect_ids, target, index

    def get_cond_sample_weights(self, cond_label_index, per_sample):
        """Per-sample weights from the label table, or None when unweighted."""
        weights = getattr(self, "cond_label_weights", None)
        if weights is None or cond_label_index is None:
            return None
        index = torch.as_tensor(cond_label_index, device=per_sample.device).flatten().long()
        if index.numel() != per_sample.numel():
            raise ValueError("cond_label_index must have one entry per sample")
        known = (index >= 0) & (index < weights.numel())
        return torch.where(known, weights[index.clamp(0, weights.numel() - 1)], torch.ones_like(per_sample)).to(per_sample.dtype)

    @property
    def cond_loss_is_per_sample(self):
        return getattr(self.cond_loss, "__name__", None) in PER_SAMPLE_COND_LOSSES

    @property
    def cond_loss_handles_unconditional(self):
        return getattr(self.cond_loss, "__name__", None) in UNCONDITIONAL_COND_LOSSES

    def raw_centered_cosine_loss(self, pred, target, aspect_ids, *args, **kwargs):
        if self.aspect_centroids is None or self.centroid_weight == 0.0:
            return 1 - (pred * target).sum(dim=-1).mean()
        else:
            return self._raw_centered_cosine_loss(pred, target, aspect_ids)

    def _raw_centered_cosine_loss(
        self,
        pred,
        target,
        aspect_ids,
        *args,
        **kwargs
    ):
        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1)

        centroid = self.aspect_centroids[aspect_ids]

        raw_loss = 1 - (pred * target).sum(dim=-1).mean()
        raw_loss = scaled_log1p(raw_loss, reference=1., alpha=1.0)

        pred_residual = F.normalize(pred - centroid, dim=-1, eps=0.05)
        target_residual = F.normalize(target - centroid, dim=-1, eps=0.05)

        centered_loss = (
            1
            - (pred_residual * target_residual)
            .sum(dim=-1)
            .mean()
        )
        centered_loss = scaled_log1p(centered_loss, reference=1., alpha=1.0)
        self.cond_loss_components['raw_loss'] = raw_loss.detach().clone()
        self.cond_loss_components['centered_loss'] = centered_loss.detach().clone()

        return (1-self.centroid_weight) * raw_loss + self.centroid_weight * centered_loss
