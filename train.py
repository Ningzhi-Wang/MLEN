import torch
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional
from dataclasses import asdict, fields
from muq import MuQ
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from mlen.modules import MLEN, MLENConfig
from mlen.datasets import MLENDatamodule


def get_logger(save_dir: str, run_name: Optional[str] = None) -> WandbLogger:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    return WandbLogger(
        project="MLEN-OneHot-RVQ",
        name=run_name or save_path.name,
        save_dir=str(save_path),
        log_model=False,
    )


def cast_dataclass(obj, target_cls):
    values = asdict(obj)
    target_fields = {f.name for f in fields(target_cls)}
    return target_cls(**{
        k: v for k, v in values.items()
        if k in target_fields
    })

def load_muq_weights(muq, mlen):
    src = muq.model.state_dict()      # no "model." prefix
    dst = mlen.model.state_dict()
    loadable = {}
    skipped = []

    for k, v in src.items():
        is_replaced_adaln = (
            k.startswith("conformer.layers.")
            and ".self_attn_layer_norm." in k
        )

        if is_replaced_adaln:
            if k.endswith(".weight"):
                k = k.removesuffix(".weight") + ".gamma"
            elif k.endswith(".bias"):
                k = k.removesuffix(".bias") + ".beta"
            else:
                skipped.append(k)
                continue

        if k == "conformer.pos_conv_embed.conv.parametrizations.weight.original0":
            k = "conformer.pos_conv_embed.conv.weight_g"
        elif k == "conformer.pos_conv_embed.conv.parametrizations.weight.original1":
            k = "conformer.pos_conv_embed.conv.weight_v"

        if k in dst and dst[k].shape == v.shape:
            loadable[k] = v
        else:
            raise ValueError(f"Unexpected key {k} with shape {v.shape} not found in destination model.")

    missing, unexpected = mlen.model.load_state_dict(loadable, strict=False)
    return mlen, missing, unexpected, skipped

def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    config = muq.config
    config = cast_dataclass(config, MLENConfig)
    config.cond_target_dim = 512
    config.cond_dim = 8
    config.flash_attention = True
    mlen = MLEN(config,lr=1e-4, weight_decay=0.01)
    mlen, missing, unexpected, skipped = load_muq_weights(muq, mlen)
    print(f"Loaded weights with {len(missing)} missing keys, {len(unexpected)} unexpected keys, and {len(skipped)} skipped keys.")
    datamodule = MLENDatamodule(
        root_path="acw713/mert_audios/wav",
        train_meta_path=str(project_root / "track_metadata.pt"),
        train_label_paths=[
            str(project_root / "fma4all_melrvq/melrvq_label.codebook0.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook1.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook2.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook3.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook4.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook5.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook6.txt"),
            str(project_root / "fma4all_melrvq/melrvq_label.codebook7.txt"),
        ],
        batch_size=8,
        pad_audio=True,
        fixed_sample_size=True,
        warn_on_label_length_mismatch=False,
    )
    datamodule.setup('fit')
    root_path = Path("/tmp/nw003/mlen/test/")
    wandb_logger = get_logger(str(root_path / "logs"), f"test_run")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(root_path / "checkpoints"),
        filename="epoch-{epoch:04d}",
        save_top_k=-1,          # keep all checkpoints
        every_n_train_steps=1000,       
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        default_root_dir=str(root_path),
        max_epochs=1000,
        accelerator="cuda",
        precision='bf16-mixed',
        devices=1,
        accumulate_grad_batches=15,
        log_every_n_steps=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(mlen, datamodule)


if __name__ == "__main__":
    main()