import os
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, default=None)
    parser.add_argument("--ckpt", dest="ckpt", type=str, default=None)
    parser.add_argument("-m", "--meta_path", dest="meta_path", type=str, default=None)
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None)
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--val_split_fraction", dest="val_split_fraction", type=float, default=0.0)
    parser.add_argument("--val_split_seed", dest="val_split_seed", type=int, default=1234)
    args = parser.parse_args()
 
    script_dir = Path(__file__).resolve().parent
    meta_path = Path(args.meta_path) or script_dir / "meta"
    data_path = Path(args.data_path) or script_dir / "data"
    log_path = Path(args.log_dir) or script_dir / "logs"

    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    config = muq.config
    config = cast_dataclass(config, MLENConfig)
    config.cond_target_dim = 512
    config.cond_dim = 8
    config.flash_attention = False
    mlen = MLEN(config,lr=1e-5, weight_decay=0.01)
    mlen, missing, unexpected, skipped = load_muq_weights(muq, mlen)
    print(f"Loaded weights with {len(missing)} missing keys, {len(unexpected)} unexpected keys, and {len(skipped)} skipped keys.")
    datamodule = MLENDatamodule(
        root_path=data_path,
        train_meta_path=str(meta_path / "track_metadata.pt"),
        train_label_paths=[
            str(meta_path / "melrvq_label.codebook0.txt"),
            str(meta_path / "melrvq_label.codebook1.txt"),
            str(meta_path / "melrvq_label.codebook2.txt"),
            str(meta_path / "melrvq_label.codebook3.txt"),
            str(meta_path / "melrvq_label.codebook4.txt"),
            str(meta_path / "melrvq_label.codebook5.txt"),
            str(meta_path / "melrvq_label.codebook6.txt"),
            str(meta_path / "melrvq_label.codebook7.txt"),
        ],
        batch_size=8,
        pad_audio=True,
        fixed_sample_size=True,
        warn_on_label_length_mismatch=False,
        val_split_fraction=args.val_split_fraction,
        val_split_seed=args.val_split_seed,
    )
    datamodule.setup('fit')
    wandb_logger = get_logger(str(log_path / "logs"), f"test_run")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(log_path / "checkpoints"),
        filename="epoch-{epoch:04d}",
        save_top_k=-1,          # keep all checkpoints
        every_n_train_steps=1000,       
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        default_root_dir=str(log_path),
        max_epochs=1000,
        accelerator="cuda",
        precision='bf16-mixed',
        devices='auto',
        accumulate_grad_batches=15,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(mlen, datamodule)


if __name__ == "__main__":
    main()
