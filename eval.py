import argparse
from dataclasses import asdict, fields
from itertools import product
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
os.environ["DATASET_LOCATION"] = "S3"
os.environ["AWS_ACCESS_KEY"] = "TNHXO1K0MW436CRWLN6M"
os.environ["AWS_SECRET_KEY"] = "pj7XzTiEAJ8WO2NKLW2CHqu2AxhFlKABfhbi0ZWM"

import boto3
import torch, librosa
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from torch.nn import functional as F
from torch.serialization import safe_globals
from muq import MuQMuLan, MuQ
from mlen.modules import MLEN, MLENConfig
from mlen.models import MLENModel
from mlen.datasets import AudioTextRVQDataset, MLENDatamodule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from musiceval.datasets import LatentModule
from musiceval import tasks

TASK_PATH = {
    "MTT": "c4dm-datasets/MagnaTagATune/",
    "GTZAN": "c4dm-datasets/gtzan/",
    "GTZANBeatTracking": "c4dm-datasets/gtzan/",
    "EMO": "c4dm-datasets/emomuisc/",
    "Giantsteps": "c4dm-datasets/Giantsteps/",
    "NSynthPitch": "c4dm-datasets/NSynth/",
    "NSynthTimbre": "c4dm-datasets/NSynth/",
    "VocalSetSinger": "c4dm-datasets/VocalSet/",
    "VocalSetTechnique": "c4dm-datasets/VocalSet/",
    "Harmonixset": "c4dm-datasets/Harmonixset/",
}

CONDS = torch.eye(8).float()
TASK_CONDS_IDX = {
    "MTT": 0,
    "GTZAN": 0,
    "EMO": 7,
    "Giantsteps": 1,
    "NSynthPitch": 1,
    "NSynthTimbre": 6,
    "VocalSetSinger": 3,
    "VocalSetTechnique": 3,
}

def get_logger(save_dir: str, run_name: Optional[str] = None) -> WandbLogger:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    return WandbLogger(
        project="pretrain_MERT_RVQ-VAE_CQT",
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
        # else:
        #     raise ValueError(f"Unexpected key {k} with shape {v.shape} not found in destination model.")

    missing, unexpected = mlen.model.load_state_dict(loadable, strict=False)
    return mlen, missing, unexpected, skipped


def get_pretrained_mlen():
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    config = muq.config
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    config = muq.config
    config = cast_dataclass(config, MLENConfig)
    config.cond_target_dim = 512
    config.cond_dim = 8
    config.flash_attention = True
    mlen = MLEN(config,lr=1e-4, weight_decay=0.01)
    mlen, missing, unexpected, skipped = load_muq_weights(muq, mlen)
    return mlen


def load_mlen_checkpoint(ckpt):
    with safe_globals([MLENConfig]):
        return MLEN.load_from_checkpoint(ckpt, strict=True)


def evaluate(config, ckpt):
    task_names = config["task_names"]
    conditional = config.get("conditional", False)

    # Download checkpoint from S3 if not present locally
    save_dir  = config['trainer']['path']
    if isinstance(ckpt, str) and not os.path.isfile(ckpt):
        os.makedirs(save_dir, exist_ok=True)
        aws_key = os.environ.get("AWS_ACCESS_KEY_C4DM02")
        aws_secret = os.environ.get("AWS_SECRET_KEY_C4DM02")
        s3 = boto3.client(
            "s3",
            endpoint_url="https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
        )
        s3.download_file(
            Bucket="c4dm-02",
            Key=ckpt,
            Filename=os.path.join(save_dir, "checkpoint.ckpt"),
        )
        ckpt = os.path.join(save_dir, "checkpoint.ckpt")
        

    # Perform hyperparameter search as in MERT
    lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    # lrs = [5e-3]
    batch_size = [64]
    drop_outs = [0.2]
    l2s = [0]
    layers = list(range(config['model']['input_channel']))
    layers.append("all")

    train_sizes = [-1]

    settings = list(product(lrs, batch_size, drop_outs, l2s, layers))
    encoder = get_pretrained_mlen() if ckpt is None else load_mlen_checkpoint(ckpt)
    encoder.eval()

    config["trainer"].pop("epochs", None)
    root_path = config["trainer"].pop("path", None)
    assert root_path is not None, "Root path must be provided for evaluation"

    for task_name in task_names:
        for train_size in train_sizes:
            results = {}
            task_config = config.copy()
            task = eval(f"tasks.{task_name}")(TASK_PATH[task_name], encoder, **task_config["task"])
            if conditional:
                task.set_conditional(CONDS[TASK_CONDS_IDX[task_name]])
            model_param = task_config["model"]
            trainer_param = task_config["trainer"]
            trainer_param["default_root_dir"] = root_path
            trainer_param["precision"] = "bf16-mixed"
            data_param = task_config["data"]
            data_param["trainset_size"] = train_size
            # datamodule = LatentModule(**data_param)
            datamodule = task.dataset(**data_param)
            datamodule.setup("fit")
            datamodule.setup("test")
            for setting in settings:
                print(f"evaluating {task_name} with setting: {setting}")
                log_path = Path(root_path)
                wandb_logger = get_logger(str(log_path / f"logs/{task_name}"), f"{task_name}-probing")
                model_param["lr"] = setting[0]
                model_param["dropout"] = setting[2]
                model_param["l2"] = setting[3]
                model_param["layer"] = setting[4]
                model_param["feature_loaded"] = datamodule.train_feature_loaded
                model = task.prober(**model_param)
                early_stop = EarlyStopping(
                    monitor=f"{task_name}_val_loss",
                    patience=20,
                    mode="max",
                    verbose=True,
                )
                checkpoint_callback = ModelCheckpoint(
                    dirpath=log_path / task_name,
                    filename=f"{task_name}-best-checkpoint-{'-'.join(map(str, setting))}",
                    save_top_k=1,
                    monitor=f"{task_name}_val_loss",
                    mode="max",
                    save_weights_only=True,
                )
                trainer_param["callbacks"] = [
                    early_stop,
                    checkpoint_callback,
                ]
                trainer_param["logger"] = False
                # trainer_param["logger"] = wandb_logger
                trainer = pl.Trainer(**trainer_param)
                trainer.fit(
                    model=model,
                    train_dataloaders=datamodule.train_dataloader(),
                    val_dataloaders=datamodule.val_dataloader(),
                )
                print(f"Loading best model from {checkpoint_callback.best_model_path}")
                best_model = type(model).load_from_checkpoint(
                    checkpoint_callback.best_model_path,
                    task=task,
                    feature_loaded=datamodule.test_feature_loaded,
                    strict=False,
                )
                best_model.eval()
                metric = trainer.test(
                    model=best_model,
                    dataloaders=datamodule.test_dataloader(),
                )[0]
                results[
                    f"lr:{setting[0]}-batch:{setting[1]}-drop:{setting[2]}-l2:{setting[3]}-layer:{setting[4]}"
                ] = metric

            if os.getenv("DATASET_LOCATION", "LOCAL") == "S3":
                aws_key = os.environ.get("AWS_ACCESS_KEY_C4DM02")
                aws_secret = os.environ.get("AWS_SECRET_KEY_C4DM02")
                s3 = boto3.client(
                    "s3",
                    endpoint_url="https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
                    aws_access_key_id=aws_key,
                    aws_secret_access_key=aws_secret,
                )
                s3_path = config["s3_path"]
                s3_key = f"{s3_path}/{task_name}_probing_results_size{train_size}.json"
                result_b = json.dumps(results, indent=4).encode("utf-8")
                s3.put_object(Bucket="c4dm-02", Key=s3_key, Body=result_b)
            else:
                with open(
                    Path(root_path)
                    / f"{task_name}_probing_results_size{train_size}.json",
                    "w",
                ) as f:
                    json.dump(results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", type=str, default="config.yml")
    parser.add_argument("--ckpt", dest="ckpt", type=str, default=None)
    args = parser.parse_args()
    config_file = args.config
    pl.seed_everything(1234, workers=True)
    torch.manual_seed(1234)
    np.random.seed(1234)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    evaluate(config, args.ckpt)


if __name__ == "__main__":
    main()


