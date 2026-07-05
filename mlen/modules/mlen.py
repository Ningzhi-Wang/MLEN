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
    cond_loss_weight: Optional[float] = field(default=1.0)
    rvq_loss_weight: Optional[float] = field(default=1.0)
    flash_attention: Optional[bool] = field(default=True)
    target_geometry_path: Optional[str] = None


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
            cond_target_dim=config.cond_target_dim,
            flash_attention=config.flash_attention,
        )
        self.rvq_loss_weight = config.rvq_loss_weight
        self.cond_loss_weight = config.cond_loss_weight 
        self.flash_attention = config.flash_attention
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

    @staticmethod
    def _padding_to_attention_mask(padding_mask):
        if padding_mask is None or not padding_mask.any():
            return None
        return ~padding_mask.bool()

    def forward(self, x, *args, padding_mask=None, label=None, cond_label=None, feature_only=False):
        attention_mask = self._padding_to_attention_mask(padding_mask)
        return self.model(
            x, *args, attention_mask=attention_mask, label=label, cond_label=cond_label, feature_only=feature_only
        )

    def extract_features(self, x, *args, padding_mask=None, ret_layer=True, **kwargs):
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
            return torch.stack([hidden for hidden in hidden_emb[1:]], dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['source']
        batch_size = x.size(0)
        cond = batch.get('aspect_cond', None)
        padding_mask = batch.get('padding_mask', None)
        label = batch.get('target', None)
        cond_label = batch.get('aspect_target', None)
        logits, hidden_emb, losses, accuracies = self(x, cond, padding_mask=padding_mask, label=label, cond_label=cond_label)
        if 'cond_loss' in losses:
            losses['cond_loss'] *= self.cond_loss_weight
        if 'melspec_2048' in losses:
            losses['melspec_2048'] *= self.rvq_loss_weight
        loss = sum(losses.values())
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in accuracies.items():
            self.log(f"train/{name}_acc", value, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        x = batch['source']
        batch_size = x.size(0)
        cond = batch.get('aspect_cond', None)
        padding_mask = batch.get('padding_mask', None)
        label = batch.get('target', None)
        cond_label = batch.get('aspect_target', None)
        logits, hidden_emb, losses, accuracies = self(x, cond, padding_mask=padding_mask, label=label, cond_label=cond_label, feature_only=False)
        output = {
            "cond_target": cond_label.detach().float().cpu(),
            "aspect_id": cond.argmax(-1).detach().long().cpu(),
        }
        if 'cond_loss' in losses:
            attention_mask = self._padding_to_attention_mask(padding_mask)
            cond_pred = self.model.get_cond_prediction(hidden_emb[-1], attention_mask)
            output["cond_pred"] = cond_pred.detach().float().cpu()
            losses['cond_loss'] *= self.cond_loss_weight
        if 'melspec_2048' in losses:
            losses['melspec_2048'] *= self.rvq_loss_weight
        for name, value in losses.items():
            self.log(f"valid/{name}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        for name, value in accuracies.items():
            self.log(f"valid/{name}_acc", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.model.cond or self.target_geometry is None:
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
        probing_modules = [self.model.linear]
        if getattr(self.model, "cond", False):
            probing_modules.append(self.model.cond_proj)

        probing_params = [param for module in probing_modules for param in module.parameters()]
        probing_ids = {id(param) for param in probing_params}
        encoding_params = [param for param in self.parameters() if id(param) not in probing_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": probing_params, "lr": self.hparams.lr },
                {"params": encoding_params, "lr": self.hparams.lr}
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
            end_factor=0.01,
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

            ax.scatter(class_preds[:, 0], class_preds[:, 1], color=color, alpha=0.25, s=8, label=f"Class {class_id}")
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
                alpha=0.8,
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
