import torch 
import numpy as np

from pytorch_lightning import LightningModule
from dataclasses import dataclass, field
from typing import List, Optional

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
    cond_loss_weight: Optional[float] = field(default=512.0)
    flash_attention: Optional[bool] = field(default=True)


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
        self.cond_loss_weight = config.cond_loss_weight 
        self.flash_attention = config.flash_attention

    def forward(self, x, *args, padding_mask=None, label=None, cond_label=None):
        attention_mask = None if padding_mask is None else ~padding_mask.bool()
        return self.model(
            x, *args, attention_mask=attention_mask, label=label, cond_label=cond_label
        )

    def extract_features(self, x, *args, padding_mask=None, ret_layer=True, **kwargs):
        was_training = self.model.training

        #TODO  Have cond inputted from outside instead of hardcoding here
        cond = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).to(x.device, dtype=x.dtype)
        cond = cond.unsqueeze(0).expand(x.size(0), -1)

        self.model.eval()
        dtype = torch.bfloat16 if self.flash_attention else torch.float32
        try:
            with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=dtype):
                attention_mask = None if padding_mask is None else ~padding_mask.bool()
                logits, hidden_emb = self.model(x, cond, *args, attention_mask=attention_mask)
        finally:
            self.model.train(was_training)
        if ret_layer:
            return torch.stack([hidden for hidden in hidden_emb[1:]], dim=1)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['source']
        cond = batch.get('aspect_cond', None)
        padding_mask = batch.get('padding_mask', None)
        label = batch.get('target', None)
        cond_label = batch.get('aspect_target', None)
        logits, hidden_emb, losses, accuracies = self(x, cond, padding_mask=padding_mask, label=label, cond_label=cond_label)
        if 'cond_loss' in losses:
            losses['cond_loss'] *= self.cond_loss_weight
        loss = sum(losses.values())
        for name, value in losses.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True)
        for name, value in accuracies.items():
            self.log(f"train/{name}_acc", value, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
