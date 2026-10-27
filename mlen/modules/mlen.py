import torch 
import numpy as np

import matplotlib.pyplot as plt
import torch.distributed as dist

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from dataclasses import dataclass, field
from typing import List, Optional
from matplotlib.patches import Ellipse

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from ..models import MLENModel
from ..models.gradient_monitor import aspect_gradient_statistics
from ..models.utils import cosine_similarity_loss, condition_to_aspect_ids
from .cond_policy import COND_POLICIES, sample_cond

@dataclass
class MLENConfig:
    label_rate:int = field(default=25)
    num_codebooks:int = field(default=1)
    codebook_dim:int = field(default=16)
    codebook_size:int = field(default=4096)
    features:List[str] = field(default_factory=lambda:["melspec_2048"])
    hop_length:int = field(default=240)
    n_mels:int = field(default=128)
    conv_dim:int = field(default=512)
    encoder_dim:int = field(default=1024)
    encoder_depth:int = field(default=12)
    mask_hop:float = field(default=0.4)
    mask_prob:float = field(default=0.6)
    is_flash:bool = field(default=False)
    stat:Optional[dict] = field(default_factory=dict)
    w2v2_config:Optional[dict] = field(default_factory=dict)
    use_rvq_target:bool = field(default=False)
    use_vq_target:bool = field(default=False)
    use_encodec_target:bool = field(default=False)
    rvq_ckpt_path: Optional[str] = field(default=None)
    recon_loss_ratio: Optional[float] = field(default=None)
    resume_checkpoint: Optional[str] = None
    rvq_n_codebooks:int = field(default=8)
    rvq_multi_layer_num:int = field(default=1)
    cond_dim: Optional[int] = field(default=8)
    cond_target_dim: Optional[int] = field(default=512)
    # "mean" or "cls" give one prediction per track; "attention" pools once
    # per aspect with learned queries and is required for multi-aspect conds.
    cond_pooling: str = field(default="mean")
    cond_pool_heads: int = field(default=8)
    # Which of a track's available aspects condition it during training:
    # "single" (one random aspect), "multi" (random subset of 1..n), "all".
    cond_policy: str = field(default="single")
    # Probability that a training track gets the all-zero (unconditional)
    # cond instead; pair with cond_loss = "whitened_or_rvq_loss".
    cond_drop_prob: float = field(default=0.0)
    # Validation policy; a fixed function of the track id, never dropped.
    cond_val_policy: str = field(default="single")
    # Name of a loss in mlen.models.mlen_model.COND_LOSSES, or "raw_centered_cosine_loss".
    cond_loss: str = field(default="cosine_similarity_loss")
    cond_loss_weight: Optional[float] = field(default=1.0)
    # Keyword arguments forwarded to the cond loss, overriding the defaults set
    # in MLENModel.forward, e.g. {"rvq_weight": 1.0, "whitening_weight": 1.0,
    # "rvq_aspects": (-1,)} for whitened_or_rvq_loss.
    cond_loss_kwargs: dict = field(default_factory=dict)
    rvq_loss_weight: Optional[float] = field(default=1.0)
    flash_attention: Optional[bool] = field(default=True)
    use_padding_mask: bool = field(default=True)
    target_geometry_path: Optional[str] = None
    aspect_centroids: Optional[torch.Tensor] = None
    centered_loss_weight: float = 0.0
    aspect_whitening_means: Optional[torch.Tensor] = None
    aspect_whitening_matrices: Optional[torch.Tensor] = None
    # [num_texts] per-label cond-loss weights (metadata target_statistics/label_weights); None = unweighted.
    cond_label_weights: Optional[torch.Tensor] = None
    log_cond_grad_norms: bool = field(default=False)
    # Optimizer steps between train/cond_aspect_loss logs; 0 disables them.
    cond_aspect_log_interval: int = field(default=1000)


class MLEN(LightningModule):
    def __init__(self, config: MLENConfig, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLENModel(
            num_codebooks=config.num_codebooks,
            codebook_dim=config.codebook_dim,
            codebook_size=config.codebook_size,
            features=config.features,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            conv_dim=config.conv_dim,
            encoder_dim=config.encoder_dim,
            encoder_depth=config.encoder_depth,
            mask_hop=config.mask_hop,
            mask_prob=config.mask_prob,
            is_flash=config.is_flash,
            stat=config.stat,
            w2v2_config=config.w2v2_config,
            use_rvq_target=config.use_rvq_target,
            use_vq_target=config.use_vq_target,
            use_encodec_target=config.use_encodec_target,
            rvq_ckpt_path=config.rvq_ckpt_path,
            recon_loss_ratio=config.recon_loss_ratio,
            label_rate=config.label_rate,
            rvq_n_codebooks=config.rvq_n_codebooks,
            rvq_multi_layer_num=config.rvq_multi_layer_num,
            cond_dim=config.cond_dim,
            cond_pooling=config.cond_pooling,
            cond_loss=config.cond_loss,
            cond_target_dim=config.cond_target_dim,
            flash_attention=config.flash_attention,
            aspect_centroids=config.aspect_centroids,
            centroid_weight=config.centered_loss_weight,
            aspect_whitening_means=config.aspect_whitening_means,
            aspect_whitening_matrices=config.aspect_whitening_matrices,
            cond_label_weights=config.cond_label_weights,
            cond_loss_kwargs=config.cond_loss_kwargs,
            cond_pool_heads=config.cond_pool_heads,
        )
        for name in ("cond_policy", "cond_val_policy"):
            if getattr(config, name) not in COND_POLICIES:
                raise ValueError(f"{name} must be one of {COND_POLICIES}, got {getattr(config, name)!r}")
        if not 0.0 <= config.cond_drop_prob <= 1.0:
            raise ValueError("cond_drop_prob must be in the range [0.0, 1.0]")
        self.cond_policy = config.cond_policy
        self.cond_drop_prob = config.cond_drop_prob
        self.cond_val_policy = config.cond_val_policy
        self.rvq_loss_weight = config.rvq_loss_weight
        self.cond_loss_weight = config.cond_loss_weight 
        self.log_cond_grad_norms = config.log_cond_grad_norms
        # Rank-local, detached CPU counters, deliberately NOT DDP buffers.
        # Intervals start fresh on resume and end at non-sanity validation.
        self._cond_grad_interval = torch.zeros(config.cond_dim or 0, 3, dtype=torch.float64)
        # Per-aspect (sum, count) of the unweighted conditional loss, for the
        # training interval and for the current validation pass.
        self._cond_loss_interval = torch.zeros(config.cond_dim or 0, 2, dtype=torch.float64)
        self._valid_cond_loss_interval = torch.zeros(config.cond_dim or 0, 2, dtype=torch.float64)
        self.cond_aspect_log_interval = config.cond_aspect_log_interval
        self._cond_loss_last_flush_step = -1
        self.flash_attention = config.flash_attention
        self.use_padding_mask = config.use_padding_mask
        if config.target_geometry_path is None:
            self.target_geometry = None
        else:
            artifact = torch.load(
                config.target_geometry_path,
                map_location="cpu",
                weights_only=True,
            )
            self.target_geometry = artifact["aspects"]
        # self.valid_pca = {}
        # self.target_means = {}

    def _padding_to_attention_mask(self, padding_mask):
        """Apply the run's padding policy to attention, pooling, and RVQ loss."""
        if padding_mask is None or not self.use_padding_mask or not padding_mask.any():
            return None
        return ~padding_mask.bool()

    def forward(self, x, *args, padding_mask=None, label=None, cond_label=None, feature_only=False, aspect_class_id=None, cond_label_index=None):
        attention_mask = self._padding_to_attention_mask(padding_mask)
        return self.model(
            x, *args, attention_mask=attention_mask, label=label, cond_label=cond_label, feature_only=feature_only,
            aspect_class_id=aspect_class_id, cond_label_index=cond_label_index,
        )

    def _batch_cond(self, batch, *, training):
        """The [B, A] cond for a batch: given explicitly, or sampled per policy.

        Batches from AudioTextRVQDataset carry no cond; the active aspects are
        drawn here from text_index, randomly in training and as a fixed
        function of the track id in validation.
        """
        cond = batch.get('aspect_cond', None)
        if cond is not None or not self.model.cond:
            return cond
        available = batch['text_index'] >= 0
        if training:
            return sample_cond(available, self.cond_policy, drop_prob=self.cond_drop_prob)
        return sample_cond(available, self.cond_val_policy, ids=batch.get('id', None))

    @torch.no_grad()
    def predict_aspects(self, x, cond, padding_mask=None):
        """Normalised predictions [B, A, D] for the aspects active in ``cond``.

        One encoder pass per track serves every active aspect; inactive
        aspects are zero. With mean or cls pooling every active aspect of a
        track receives the same vector.
        """
        was_training = self.model.training
        self.model.eval()
        dtype = torch.bfloat16 if self.flash_attention else torch.float32
        try:
            with torch.autocast(device_type=x.device.type, dtype=dtype):
                attention_mask = self._padding_to_attention_mask(padding_mask)
                _, hidden_emb = self.model(x, cond, attention_mask=attention_mask, feature_only=True)
                prediction = self.model.get_cond_prediction(hidden_emb[-1], attention_mask)
        finally:
            self.model.train(was_training)
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(1).expand(-1, cond.size(1), -1)
        return prediction.float() * cond.ne(0).unsqueeze(-1).to(prediction.device)

    def extract_features(
        self,
        x,
        *args,
        padding_mask=None,
        ret_layer=True,
        return_cls_token=False,
        **kwargs,
    ):
        """Return layer features as [batch, layers, time, hidden].

        When ``return_cls_token`` is true, the time dimension has size one.
        """
        if return_cls_token and (not ret_layer or self.model.cond_pooling != "cls"):
            raise ValueError(
                "return_cls_token=True requires ret_layer=True and cond_pooling='cls'"
            )

        was_training = self.model.training

        self.model.eval()
        dtype = torch.bfloat16 if self.flash_attention else torch.float32
        try:
            with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=dtype):
                attention_mask = self._padding_to_attention_mask(padding_mask)
                logits, hidden_emb = self.model(x, *args, attention_mask=attention_mask, feature_only=True)
        finally:
            self.model.train(was_training)
        if ret_layer:
            if return_cls_token:
                return torch.stack(
                    [hidden[:, :1] for hidden in hidden_emb[1:]],
                    dim=1,
                )
            return torch.stack(
                [self.model.get_frame_hidden(hidden) for hidden in hidden_emb[1:]],
                dim=1,
            )
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['source']
        batch_size = x.size(0)
        cond = self._batch_cond(batch, training=True)
        padding_mask = batch.get('padding_mask', None)
        label = batch.get('target', None)
        cond_label = batch.get('aspect_target', None)
        class_id = batch.get('aspect_class_id', None)
        logits, hidden_emb, losses, accuracies = self(x, cond, padding_mask=padding_mask, label=label, cond_label=cond_label, aspect_class_id=class_id, cond_label_index=batch.get('text_index', None))
        if 'cond_loss' in losses:
            self._record_cond_gradients(hidden_emb[-1], cond_label, cond, padding_mask)
            self._accumulate_cond_losses(getattr(self, "_cond_loss_interval", None))
            losses['cond_loss'] *= self.cond_loss_weight
        if 'melspec_2048' in losses:
            losses['melspec_2048'] *= self.rvq_loss_weight
        loss = sum(losses.values())
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in accuracies.items():
            self.log(f"train/{name}_acc", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        for name, value in self.model.cond_loss_components.items():
            self.log(
                f"train/cond_{name}_loss",
                value,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        for name, value in getattr(self.model, "rvq_loss_components", {}).items():
            self.log(f"train/rvq_{name}", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # After backward/step, so global_step already counts this batch's
        # window and the flush happens outside the DDP-wrapped forward.
        self._maybe_flush_train_cond_losses()

    def _maybe_flush_train_cond_losses(self):
        """Flush every `cond_aspect_log_interval` optimizer steps.

        global_step is identical on every rank, so all ranks flush together
        (the all-reduce inside is collective). The last-flush guard makes the
        flush fire once per step even though every micro-batch of an
        accumulation window sees the same global_step.
        """
        every = getattr(self, "cond_aspect_log_interval", 0)
        step = self.global_step
        if not every or step <= 0 or step % every or step == self._cond_loss_last_flush_step:
            return
        self._cond_loss_last_flush_step = step
        self._flush_cond_losses(getattr(self, "_cond_loss_interval", None), "train", on_step=True)

    @staticmethod
    def _aspect_metric_name(aspect):
        names = (
            "genre", "key", "tempo", "vocality", "tuning_frequency",
            "tuning_stability", "instrumentation", "mood",
        )
        return names[aspect] if aspect < len(names) else f"aspect_{aspect}"

    def _reduce_cond_grad_statistics(self, statistics):
        result = statistics.to(self.device).clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def _log_cond_grad_statistics(self, statistics, *, interval):
        prefix = "valid" if interval else "train"
        for aspect, (count, weighted_norm, batches) in enumerate(statistics):
            name = self._aspect_metric_name(aspect)
            options = dict(on_step=not interval, on_epoch=interval, sync_dist=False)
            # Statistics are already globally reduced; do not average rank means.
            self.log(f"{prefix}/cond_grad_samples/{name}", count, **options)
            if interval:
                self.log(f"{prefix}/cond_grad_batches/{name}", batches, **options)
            if count > 0:
                self.log(f"{prefix}/cond_grad_norm/{name}", weighted_norm / count, **options)

    def _record_cond_gradients(self, hidden, target, cond, padding_mask):
        if not self.log_cond_grad_norms or self.model.cond_loss is not cosine_similarity_loss:
            return
        if self.model.cond_pooling not in ("mean", "cls"):
            return  # The monitor models the linear head over a per-track pooling only.
        attention_mask = self._padding_to_attention_mask(padding_mask)
        rows = getattr(self.model, "cond_pair_rows", None)
        if rows is None:
            aspect_ids = condition_to_aspect_ids(cond)
        else:
            # One entry per (track, aspect) pair, as the loss saw them.
            hidden, target, aspect_ids = hidden[rows], self.model.cond_pair_target, self.model.cond_aspect_ids
            attention_mask = None if attention_mask is None else attention_mask[rows]
        statistics = aspect_gradient_statistics(
            self.model.cond_proj, 
            hidden, 
            target, 
            aspect_ids,
            self._cond_grad_interval.size(0), 
            pooling=self.model.cond_pooling,
            attention_mask=attention_mask,
            loss_weight=self.cond_loss_weight,
        )
        self._cond_grad_interval += statistics.cpu()
        self._log_cond_grad_statistics(
            self._reduce_cond_grad_statistics(statistics), interval=False,
        )

    def _flush_cond_gradients(self):
        if not self.log_cond_grad_norms or not self._cond_grad_interval.numel():
            return
        statistics = self._reduce_cond_grad_statistics(self._cond_grad_interval)
        if statistics[:, 0].sum() > 0:
            self._log_cond_grad_statistics(statistics, interval=True)
        self._cond_grad_interval.zero_()

    def _accumulate_cond_losses(self, interval):
        """Add this batch's per-sample cond losses to a rank-local interval.

        Nothing is logged here: aspects missing from a batch differ between
        ranks, and a sync_dist log with mismatched keys deadlocks DDP.
        """
        per_sample = getattr(self.model, "cond_loss_per_sample", None)
        if interval is None or per_sample is None or not interval.numel():
            return
        aspect_ids = self.model.cond_aspect_ids.detach().long()
        if per_sample.dim() != 1 or per_sample.numel() != aspect_ids.numel():
            return
        # Accumulate on the loss's device so no per-step host sync is needed;
        # the table only moves at flush time.
        if interval.device != per_sample.device:
            interval.data = interval.data.to(per_sample.device)
        # Unconditional (zeroed) samples carry aspect id -1 and are skipped.
        valid = (aspect_ids >= 0) & (aspect_ids < interval.size(0))
        values = per_sample.detach().double()[valid]
        aspect_ids = aspect_ids.to(interval.device)[valid]
        interval[:, 0].index_add_(0, aspect_ids, values)
        interval[:, 1].index_add_(0, aspect_ids, torch.ones_like(values))

    def _flush_cond_losses(self, interval, prefix, *, on_step=False):
        """All-reduce an interval on every rank, log per-aspect means, reset it."""
        if interval is None or not interval.numel():
            return
        statistics = self._reduce_cond_grad_statistics(interval)
        # Already all-reduced, so every rank logs identical values and keys;
        # sync_dist=True is a no-op mean that keeps Lightning's DDP warning quiet.
        # Note: Lightning re-emits an on_step value at every logging step until
        # the epoch resets, so W&B shows plateaus between flushes; only the
        # points at multiples of cond_aspect_log_interval are new values.
        for aspect, (total, count) in enumerate(statistics.tolist()):
            if count > 0:
                self.log(
                    f"{prefix}/cond_aspect_loss/{self._aspect_metric_name(aspect)}",
                    total / count, on_step=on_step, on_epoch=not on_step, sync_dist=True,
                )
        interval.zero_()

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        # Sanity-check batches must not leak into the first real validation.
        interval = getattr(self, "_valid_cond_loss_interval", None)
        if interval is not None:
            interval.zero_()

    def validation_step(self, batch, batch_idx):
        x = batch['source']
        batch_size = x.size(0)
        cond = self._batch_cond(batch, training=False)
        padding_mask = batch.get('padding_mask', None)
        label = batch.get('target', None)
        cond_label = batch.get('aspect_target', None)
        class_id = batch.get('aspect_class_id', None)
        logits, hidden_emb, losses, accuracies = self(x, cond, padding_mask=padding_mask, label=label, cond_label=cond_label, aspect_class_id=class_id, feature_only=False, cond_label_index=batch.get('text_index', None))
        output = {}
        if 'cond_loss' in losses:
            if getattr(self.model, "cond_pair_rows", None) is not None:
                # One entry per (track, aspect) pair, exactly as the loss saw them.
                output = {
                    "cond_target": self.model.cond_pair_target.float().cpu(),
                    "aspect_id": self.model.cond_aspect_ids.long().cpu(),
                    "cond_pred": self.model.cond_pred.float().cpu(),
                }
            else:
                # Models without pair bookkeeping: one aspect per track.
                attention_mask = self._padding_to_attention_mask(padding_mask)
                cond_pred = self.model.get_cond_prediction(hidden_emb[-1], attention_mask)
                output = {
                    "cond_target": cond_label.detach().float().cpu(),
                    "aspect_id": condition_to_aspect_ids(cond).detach().long().cpu(),
                    "cond_pred": cond_pred.detach().float().cpu(),
                }
            self._accumulate_cond_losses(getattr(self, "_valid_cond_loss_interval", None))
            losses['cond_loss'] *= self.cond_loss_weight
        if 'melspec_2048' in losses:
            losses['melspec_2048'] *= self.rvq_loss_weight
        for name, value in losses.items():
            self.log(f"valid/{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in accuracies.items():
            self.log(f"valid/{name}_acc", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        for name, value in self.model.cond_loss_components.items():
            self.log(
                f"valid/cond_{name}_loss",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        for name, value in getattr(self.model, "rvq_loss_components", {}).items():
            self.log(f"valid/rvq_{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        if output:
            self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.validation_outputs.clear()
            return
        self._flush_cond_gradients()
        self._flush_cond_losses(getattr(self, "_valid_cond_loss_interval", None), "valid")
        if not self.model.cond or self.target_geometry is None:
            self.validation_outputs.clear()
            return

        gathered = self._gather_validation_outputs()
        self.validation_outputs.clear()

        # Only rank zero continues past here.
        if gathered is None:
            return

        aspect_ids = gathered["aspect_id"]
        cond_preds = gathered["cond_pred"]
        cond_targets = gathered["cond_target"]
        for aspect_id, aspect_name in enumerate(["genre", "key"]):
            mask = aspect_ids == aspect_id
            if not mask.any():
                continue
            preds = cond_preds[mask]
            targets = cond_targets[mask]
            geometry = self.target_geometry[aspect_name]
            basis = geometry['centers']
            inferred_ids = (targets @ basis.T).argmax(dim=-1)
            inferred_pred_ids = (preds @ basis.T).argmax(dim=-1)
            self._plot_confusion_matrix(
                aspect_name,
                inferred_ids,
                inferred_pred_ids,
                number_of_classes=len(basis),
            )

            self._plot_pca(
                aspect_name, preds, geometry, inferred_ids
            )

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.hparams.warmup_steps, max(total_steps - 1, 1))
        decay_steps = max(total_steps - warmup_steps, 1)
        # The RVQ head is trained from scratch, so it gets its own learning
        # rate; cond_proj and the encoder share the fine-tuning rate.
        head_params = list(self.model.linear.parameters())
        head_ids = {id(param) for param in head_params}
        encoding_params = [param for param in self.parameters() if id(param) not in head_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.hparams.get("head_lr", self.hparams.lr)},
                {"params": encoding_params, "lr": self.hparams.lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )
        warm_up = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=decay_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warm_up, decay],
            milestones=[warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _grad_norm(self, module):
        return torch.norm(
            torch.stack([
                p.grad.detach().norm(2)
                for p in module.parameters()
                if p.grad is not None
            ]),
            2,
        )

    def on_before_optimizer_step(self, optimizer):
        total_norm = self._grad_norm(self)
        self.log("train/grad_norm", total_norm, on_step=True)
        linear_norm = self._grad_norm(self.model.linear)
        self.log("train/linear_grad_norm", linear_norm, on_step=True)
        if self.model.cond:
            cond_norm = self._grad_norm(self.model.cond_proj)
            self.log("train/cond_grad_norm", cond_norm, on_step=True)

    def _gather_validation_outputs(self):
        if self.validation_outputs:
            local_output = {
                "aspect_id": torch.cat([
                    x["aspect_id"] for x in self.validation_outputs
                ]),
                "cond_pred": torch.cat([
                    x["cond_pred"] for x in self.validation_outputs
                ]),
                "cond_target": torch.cat([
                    x["cond_target"] for x in self.validation_outputs
                ]),
            }

            # Only retain genre and key before transferring.
            keep = (
                (local_output["aspect_id"] == 0)
                | (local_output["aspect_id"] == 1)
            )

            local_output = {
                name: value[keep]
                for name, value in local_output.items()
            }
        else:
            local_output = None

        if dist.is_available() and dist.is_initialized():
            gathered_outputs = [None] * dist.get_world_size()

            # Every rank must execute this call.
            dist.all_gather_object(
                gathered_outputs,
                local_output,
            )
        else:
            gathered_outputs = [local_output]

        gathered_outputs = [
            output for output in gathered_outputs
            if output is not None and len(output["aspect_id"]) > 0
        ]

        # Nonzero ranks participated in gathering but do not plot.
        if not self.trainer.is_global_zero:
            return None

        if not gathered_outputs:
            return None

        return {
            name: torch.cat([
                output[name] for output in gathered_outputs
            ])
            for name in ("aspect_id", "cond_pred", "cond_target")
        }

    def _plot_confusion_matrix(
        self,
        aspect_name,
        target_ids,
        prediction_ids,
        number_of_classes,
    ):
        labels = np.arange(number_of_classes)

        cm = confusion_matrix(
            target_ids.numpy(),
            prediction_ids.numpy(),
            labels=labels,
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(9, 8))

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        ).plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            colorbar=False,
        )

        ax.set_title(f"{aspect_name.capitalize()} confusion matrix")
        fig.tight_layout()

        self._log_validation_figure(
            f"valid/{aspect_name}/confusion_matrix",
            fig,
        )

    @staticmethod
    def _project_saved_pca(embeddings, pca_state):
        mean = pca_state["mean"]
        components = pca_state["components"]

        return ((embeddings - mean) @ components.T).numpy()


    def _plot_pca(self, aspect_name, preds, geometry, basis_ids):
        preds_2d = self._project_saved_pca(
            preds,
            geometry["pca"],
        )

        target_2d = geometry["target_centers_2d"].numpy()
        covariances = geometry[
            "target_covariances_2d"
        ].numpy()

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab20(
            np.linspace(0, 1, len(target_2d))
        )
        distances = np.linalg.norm(
            target_2d[:, None, :] - target_2d[None, :, :],
            axis=-1,
        )
        np.fill_diagonal(distances, np.inf)
        nearest_distances = distances.min(axis=1)
        nstd = 2
        
 
        for class_id, color in enumerate(colors):
            point_mask = basis_ids.numpy() == class_id
            class_preds = preds_2d[point_mask]

            ax.scatter(class_preds[:, 0], class_preds[:, 1], color=color, alpha=0.8, s=10, label=f"Class {class_id}")
            covariance = covariances[class_id].copy()
            largest_eigenvalue = max(
                float(np.linalg.eigvalsh(covariance).max()),
                1e-12,
            )

            current_major_radius = (
                nstd * np.sqrt(largest_eigenvalue)
            )
            maximum_major_radius = (
                0.6 * nearest_distances[class_id]
            )
            if current_major_radius > maximum_major_radius:
                scale = (
                    maximum_major_radius / current_major_radius
                ) ** 2
                covariance *= scale
            self._add_target_gaussian_contour(ax, target_2d[class_id], covariance, color, n_std=nstd)

        explained = 100 * geometry['pca']['explained_variance_ratio'].numpy()

        ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
        ax.set_title(
            f"{aspect_name.capitalize()} predictions with {len(preds)} samples"
        )
        ax.legend(fontsize=7, ncols=2)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        self._log_validation_figure(
            f"valid/{aspect_name}/pca",
            fig,
        )    

    @staticmethod
    def _add_target_gaussian_contour(
        ax,
        center,
        covariance,
        color,
        n_std=2.0,
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = eigenvalues.argsort()[::-1]
        eigenvalues = np.maximum(
            eigenvalues[order],
            1e-10,
        )
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(
            np.arctan2(
                eigenvectors[1, 0],
                eigenvectors[0, 0],
            )
        )

        width, height = (
            2 * n_std * np.sqrt(eigenvalues)
        )

        ax.add_patch(
            Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                linestyle="--",
                alpha=1.0,
            )
        )

    def _log_validation_figure(self, name, figure):
        if self.trainer.is_global_zero and self.logger is not None:
            import wandb

            self.logger.experiment.log(
                {
                    name: wandb.Image(figure),
                    "epoch": self.current_epoch,
                }
            )

        plt.close(figure)
