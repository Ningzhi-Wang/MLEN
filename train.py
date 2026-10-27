import os
import re
import argparse
from time import perf_counter

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
from mlen.datasets.target_store import load_target_statistics, resolve_training_metadata



def get_logger(save_dir: str, run_name: Optional[str] = None) -> WandbLogger:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    name = run_name or save_path.name
    return WandbLogger(
        project="MLEN-OneHot-RVQ",
        name=name,
        # A run id derived from the run name lets a resumed job continue the
        # same W&B run instead of opening a new one on every restart.
        id=re.sub(r"[^A-Za-z0-9_.-]", "-", name),
        resume="allow",
        save_dir=str(save_path),
        log_model=False,
    )


def describe_checkpoint(path):
    """Return (epoch, global_step) of a Lightning checkpoint.

    Reads the file through mmap so a 4 GB checkpoint costs no RAM, and fails
    here with a clear message if the file is truncated or unreadable instead
    of deep inside trainer.fit().
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as error:
        raise RuntimeError(f"Resume checkpoint {path} is unreadable: {error}") from error
    epoch, step = checkpoint.get("epoch"), checkpoint.get("global_step")
    del checkpoint
    if step is None:
        raise RuntimeError(f"Resume checkpoint {path} has no global_step; not a Trainer checkpoint.")
    return epoch, step


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
    statistics = load_target_statistics(metadata_path, "aspect_whitening")
    try:
        if statistics is None:
            raise KeyError("aspect_whitening")
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
    record = load_target_statistics(metadata_path, "label_weights")
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
    parser.add_argument("--targets-path", type=Path, default=None,
                        help="Migrated target directory or manifest; defaults to META_PATH/targets when present")
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
        choices=("mean", "cls", "attention"),
        default="mean",
        help=(
            "Pooling used for the conditional prediction head; 'attention' pools "
            "once per aspect and is required for the 'multi' and 'all' cond policies."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("MASTER_PORT", "29501")
    train_name = os.environ.get("TRAIN_NAME", "test-run")
    if args.ckpt:
        epoch, step = describe_checkpoint(args.ckpt)
        print(f"Resuming from {args.ckpt} (epoch={epoch}, global_step={step})")
 
    script_dir = Path(__file__).resolve().parent
    meta_path = Path(args.meta_path) if args.meta_path else script_dir / "meta"
    data_path = Path(args.data_path) if args.data_path else script_dir / "data"
    log_path = Path(args.log_dir) if args.log_dir else script_dir / "logs"
    train_meta_path, target_statistics_path = resolve_training_metadata(meta_path, args.targets_path)
    print(f"Training targets selected: {train_meta_path}", flush=True)

    model_started = perf_counter()
    print("Loading pretrained MuQ model...", flush=True)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    print(f"Loaded pretrained MuQ model in {perf_counter() - model_started:.2f} s", flush=True)
    config = muq.config
    config = cast_dataclass(config, MLENConfig)
    config.cond_target_dim = 512
    config.cond_loss_weight = 1.
    config.rvq_loss_weight = 0.
    config.cond_dim = 8 
    config.cond_pooling = args.cond_pooling
    # config.cond_loss = "whitened_cosine_similarity_loss"
    config.cond_loss = "whitened_or_rvq_loss"
    # Which of a track's available aspects condition it in training: "single"
    # (one random aspect, as before), "multi" (random subset), "all". Multi-
    # aspect conds need cond_pooling "attention" to get one prediction per aspect.
    config.cond_policy = "single"
    # Probability that a training track gets the all-zero (unconditional) cond;
    # pair with cond_loss = "whitened_or_rvq_loss" and rvq_loss_weight = 0.
    config.cond_drop_prob = 0.0
    # Validation cond: a fixed function of the track id, never dropped.
    config.cond_val_policy = "single"
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
        train_meta_path=str(train_meta_path),
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
    # ckpt_path=None starts from scratch; otherwise Lightning restores the
    # weights, optimizer, LR schedule, epoch and global step from the file.
    # weights_only=False: our own checkpoints pickle MLENConfig in
    # hyper_parameters, which torch>=2.6's default strict unpickler rejects.
    trainer.fit(mlen, datamodule, ckpt_path=args.ckpt, weights_only=False)

    trainer.save_checkpoint(
        str(log_path / "checkpoints" / "final.ckpt"),
        weights_only=False,
    )


if __name__ == "__main__":
    main()
