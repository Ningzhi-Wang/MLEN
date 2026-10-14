import os
import argparse

import torch
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional
from dataclasses import asdict, fields
from muq import MuQ
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

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


def load_muq_weights(muq, mlen, load_loss_head=True):
    """Load MuQ weights, remapping only keys replaced by conditional MLEN."""
    src = muq.model.state_dict()      # no "model." prefix
    dst = mlen.model.state_dict()
    loadable = {}
    skipped = []

    for k, v in src.items():
        if not load_loss_head and k in {"linear.weight", "linear.bias"}:
            skipped.append(k)
            continue

        is_replaced_adaln = (
            k.startswith("conformer.layers.")
            and ".self_attn_layer_norm." in k
        )

        # With cond_dim=None MLEN retains MuQ's standard LayerNorm, so the
        # original key is already present and must not be remapped.
        if k not in dst and is_replaced_adaln:
            if k.endswith(".weight"):
                k = k.removesuffix(".weight") + ".gamma"
            elif k.endswith(".bias"):
                k = k.removesuffix(".bias") + ".beta"
            else:
                skipped.append(k)
                continue

        # CondConformer uses the legacy weight_g/weight_v representation,
        # whereas the unconditioned MuQ Conformer keeps parametrizations.*
        if (
            k not in dst
            and k == "conformer.pos_conv_embed.conv.parametrizations.weight.original0"
        ):
            k = "conformer.pos_conv_embed.conv.weight_g"
        elif (
            k not in dst
            and k == "conformer.pos_conv_embed.conv.parametrizations.weight.original1"
        ):
            k = "conformer.pos_conv_embed.conv.weight_v"

        if k in dst and dst[k].shape == v.shape:
            loadable[k] = v
        # else:
        #     raise ValueError(f"Unexpected key {k} with shape {v.shape} not found in destination model.")

    missing, unexpected = mlen.model.load_state_dict(loadable, strict=False)
    return mlen, missing, unexpected, skipped


def freeze_unused_parameters(mlen):
    for name, p in mlen.named_parameters():
        if name == "model.cls_token":
            p.requires_grad = mlen.model.cond_pooling == "cls"
        elif name.startswith("model.rvq."):
            p.requires_grad = False
    return mlen


def load_whitening_statistics(metadata_path):
    """Read the two fixed loss tensors without retaining the full metadata."""
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    try:
        statistics = metadata["target_statistics"]["aspect_whitening"]
        means = torch.as_tensor(statistics["means"], dtype=torch.float32)
        matrices = torch.as_tensor(
            statistics["whitening_matrices"], dtype=torch.float32
        )
    except KeyError as error:
        raise ValueError(
            f"{metadata_path} is missing target_statistics/aspect_whitening "
            "means or whitening_matrices; run the whitening setup cell first"
        ) from error
    return means, matrices


def load_label_weights(metadata_path):
    """Per-label cond-loss weights from the label-weight setup cell, or None if absent."""
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    record = metadata.get("target_statistics", {}).get("label_weights")
    if record is None:
        return None
    weights = torch.as_tensor(record["weights"], dtype=torch.float32)
    print(
        f"Loaded label weights from {metadata_path}: alpha={record.get('alpha')} "
        f"clip=[{record.get('clip_min')}, {record.get('clip_max')}] "
        f"range=[{weights.min():.3f}, {weights.max():.3f}]"
    )
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, default=None)
    parser.add_argument("--ckpt", dest="ckpt", type=str, default=None)
    parser.add_argument("-m", "--meta_path", dest="meta_path", type=str, default=None)
    parser.add_argument("-d", "--data_path", dest="data_path", type=str, default=None)
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default=None)
    parser.add_argument("--val_split_fraction", dest="val_split_fraction", type=float, default=0.0)
    parser.add_argument("--val_split_seed", dest="val_split_seed", type=int, default=1234)
    # Optional frontend memory layout; see TRAINING_ACCELERATION_REPORT.md.
    parser.add_argument(
        "--frontend-channels-last",
        action="store_true",
        help="Use channels-last weights and inputs for the conditional MLEN Conv2d frontend.",
    )
    parser.add_argument(
        "--cond-pooling",
        choices=("mean", "cls"),
        default="mean",
        help="Pooling used for the conditional prediction head.",
    )
    args = parser.parse_args()

    os.environ.setdefault("MASTER_PORT", "29501")
    train_name = os.environ.get("TRAIN_NAME", "test-run")
 
    script_dir = Path(__file__).resolve().parent
    meta_path = Path(args.meta_path) or script_dir / "meta"
    data_path = Path(args.data_path) or script_dir / "data"
    log_path = Path(args.log_dir) or script_dir / "logs"

    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    config = muq.config
    config = cast_dataclass(config, MLENConfig)
    config.cond_target_dim = 512
    config.cond_loss_weight = 1.
    config.rvq_loss_weight = 0.
    config.cond_dim = 8 
    config.cond_pooling = 'mean'
    # config.cond_loss = "whitened_cosine_similarity_loss"
    config.cond_loss = "whitened_or_rvq_loss"
    # Probability of replacing a training sample's aspect by the zero
    # (unconditional) condition; validation never drops. 
    aspect_dropout = 0.1
    # Weights inside the cond loss. For whitened_or_rvq_loss, e.g.
    # {"whitening_weight": 1.0, "rvq_weight": 1.0, "rvq_aspects": (-1,)};
    # with that loss also set rvq_loss_weight = 0 so RVQ is not counted twice.
    config.cond_loss_kwargs = {
        "whitening_weight": 1,
        "rvq_weight": 1,
    }
    config.flash_attention = True
    config.use_padding_mask = False
    config.target_geometry_path = str(
        meta_path / "target_geometry.pt"
    )

    # Only whitened_* cond losses require whitening statistics.
    target_statistics_path = meta_path / "track_metadata_with_centroids.pt"

    (
        config.aspect_whitening_means,
        config.aspect_whitening_matrices,
    ) = load_whitening_statistics(target_statistics_path)

    print(f"Loaded whitening statistics from {target_statistics_path}")
    # Density-balanced per-label weights; set to None for the unweighted cond loss.
    config.cond_label_weights = load_label_weights(target_statistics_path)

    #### For centroid loss.
    # metadata = torch.load(
    #     meta_path / "track_metadata_with_centroids.pt",
    #     map_location="cpu",
    # )

    # statistics = metadata["target_statistics"]

    # config.aspect_centroids = torch.as_tensor(
    #     statistics["aspect_centroids"]
    # ).float()

    # config.centered_loss_weight = 0.5
    # del metadata

    mlen = MLEN(config, lr=4e-5, head_lr=3e-4, weight_decay=0.01, warmup_steps=1000)
    mlen, missing, unexpected, skipped = load_muq_weights(
        muq,
        mlen,
        load_loss_head=False,
    )
    # Convert once after loading weights, before optimizer/DDP construction.
    if args.frontend_channels_last:
        mlen.model.enable_frontend_channels_last()
    print(f"Frontend channels-last: {args.frontend_channels_last}")
    mlen = freeze_unused_parameters(mlen)
    print(f"Loaded weights with {len(missing)} missing keys, {len(unexpected)} unexpected keys, and {len(skipped)} skipped keys.")
    datamodule = MLENDatamodule(
        root_path=data_path,
        train_meta_path=str(meta_path / "track_metadata.pt"),
        train_label_paths=[
            str(meta_path / f"latentrvq_label.codebook{i}.txt") for i in range(8)
        ],
        batch_size=8,
        min_sample_size=24_000 * 29,
        max_sample_size=24_000 * 30,
        pad_audio=True,
        fixed_sample_size=True,
        warn_on_label_length_mismatch=False,
        val_split_fraction=args.val_split_fraction,
        val_split_seed=args.val_split_seed,
        train_aspect_dropout=aspect_dropout,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # datamodule.setup('fit')
    wandb_logger = get_logger(str(log_path / "logs"), train_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(log_path / "checkpoints"),
        filename="epoch={epoch:04d}-step={step:08d}",
        auto_insert_metric_name=False,
        save_top_k=-1,          # keep all checkpoints
        every_n_train_steps=2000,       
        save_on_train_epoch_end=True,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        default_root_dir=str(log_path),
        max_epochs=50,
        accelerator="cuda",
        precision='bf16-mixed',
        devices='auto',
        # strategy=DDPStrategy(find_unused_parameters=True),
        strategy='auto',
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=10.0,
        benchmark=True,
        val_check_interval=1000,
    )
    trainer.fit(mlen, datamodule)

    trainer.save_checkpoint(
        str(log_path / "checkpoints" / "final.ckpt"),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
