import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch import nn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from look2hear.datas import VideoPretrainDataModule
from look2hear.models import VideoEncoder


def info(msg: str) -> None:
    print(f"[video-pretrain] {msg}")


def safe_get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_logger(config: Dict[str, Any], exp_name: str):
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, exp_name), exist_ok=True)

    use_swanlab = safe_get(config, "logging.use_swanlab", False)
    if use_swanlab:
        try:
            from swanlab.integration.pytorch_lightning import SwanLabLogger

            swan_cfg = safe_get(config, "logging.swanlab", {}) or {}
            info("Using SwanLabLogger")
            return SwanLabLogger(
                experiment_name=exp_name,
                save_dir=os.path.join(logger_dir, exp_name),
                project=swan_cfg.get("project", "dolphin"),
                workspace=swan_cfg.get("workspace", ""),
                offline=swan_cfg.get("offline", False),
            )
        except Exception as ex:
            info(f"SwanLab unavailable, fallback to TensorBoard. reason: {ex}")

    info("Using TensorBoardLogger")
    return TensorBoardLogger(logger_dir, name=exp_name)


class VideoEncoderPretrainSystem(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Dict[str, Any] | None = None,
        train_loader=None,
        val_loader=None,
        config: Dict[str, Any] | None = None,
    ):
        super().__init__()
        self.video_encoder = model
        self.optimizer_cfg = dict(optimizer_cfg)
        self.scheduler_cfg = scheduler_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = {} if config is None else config
        self.default_monitor = "val/loss"

        self.recon_criterion = nn.MSELoss()
        self.distill_criterion = nn.MSELoss()
        self.distill_cost = safe_get(self.config, "training.distill_cost", 0.0)
        self.commitment_cost = safe_get(self.config, "training.commitment_cost", 0.0)

        encoded_video_path = safe_get(self.config, "encoded_video.encoded_video", None)
        self.encoded_video = None
        if encoded_video_path:
            info("loading encoded video targets")
            self.encoded_video = torch.load(str(encoded_video_path), map_location="cpu")
            info("encoded video targets loaded")

    def forward(self, x):
        output = self.video_encoder.reconstruct(x)
        if isinstance(output, (tuple, list)):
            x_hat = output[0]
            distill_out = output[1] if len(output) > 1 else None
            commitment_loss = output[2] if len(output) > 2 else x.new_zeros(())
            return x_hat, distill_out, commitment_loss
        return output, None, x.new_zeros(())

    def _get_mouth_embedding(self, mouth_paths, ref_tensor):
        if self.encoded_video is None:
            return None

        embeddings = []
        for mouth_path in mouth_paths:
            key = os.path.abspath(str(mouth_path))
            if key not in self.encoded_video:
                raise KeyError(f"Missing encoded_video features for: {key}")
            embedding = self.encoded_video[key]
            if not torch.is_tensor(embedding):
                embedding = torch.as_tensor(embedding)
            embeddings.append(embedding)

        return torch.stack(embeddings).to(device=ref_tensor.device, dtype=ref_tensor.dtype)

    def step(self, batch):
        x, mouth = batch
        x = x.float()
        mouth_emb = self._get_mouth_embedding(mouth, x)

        x_hat, distill_out, commitment_loss = self(x)
        recon_loss = self.recon_criterion(x_hat, x)
        distill_loss = x.new_zeros(())
        if not torch.is_tensor(commitment_loss):
            commitment_loss = x.new_tensor(commitment_loss)
        else:
            commitment_loss = commitment_loss.to(device=x.device, dtype=x.dtype)
        if commitment_loss.ndim > 0:
            commitment_loss = commitment_loss.mean()

        if mouth_emb is not None and distill_out is not None:
            distill_loss = self.distill_criterion(distill_out, mouth_emb)

        total_loss = recon_loss + self.commitment_cost * commitment_loss + self.distill_cost * distill_loss
        return total_loss, recon_loss, commitment_loss, distill_loss

    def training_step(self, batch, batch_idx):
        loss, recon_loss, commitment_loss, distill_loss = self.step(batch)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("train/commitment_loss", commitment_loss, on_epoch=True, sync_dist=True)
        self.log("train/distill_loss", distill_loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, commitment_loss, distill_loss = self.step(batch)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/recon_loss", recon_loss, on_epoch=True, sync_dist=True)
        self.log("val/commitment_loss", commitment_loss, on_epoch=True, sync_dist=True)
        self.log("val/distill_loss", distill_loss, on_epoch=True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.optimizer_cfg.pop("optim_name", "adam").lower()
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer for pretrain: {optimizer_name}")

        optimizer = optimizer_cls(self.parameters(), **self.optimizer_cfg)
        if not self.scheduler_cfg or not self.scheduler_cfg.get("sche_name"):
            return optimizer

        scheduler_name = self.scheduler_cfg["sche_name"]
        scheduler_kwargs = dict(self.scheduler_cfg.get("sche_config", {}))
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
            optimizer=optimizer,
            **scheduler_kwargs,
        )
        if scheduler_name == "ReduceLROnPlateau":
            return [optimizer], [{"scheduler": scheduler, "monitor": self.default_monitor}]
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main(
    config: Dict[str, Any],
    accelerator_override: str | None = None,
    devices_override: list[int] | None = None,
    bf16: bool = False,
) -> None:
    requested_accelerator = accelerator_override or safe_get(config, "training.accelerator", "auto")
    if requested_accelerator not in {"auto", "cpu", "cuda", "tpu"}:
        raise ValueError(
            "training.accelerator must be one of: auto, cpu, cuda, tpu; "
            f"got {requested_accelerator!r}."
        )
    if requested_accelerator == "auto":
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        accelerator = requested_accelerator
    if accelerator == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but CUDA is not available.")
    if accelerator == "tpu":
        from pytorch_lightning.accelerators import XLAAccelerator

        if not XLAAccelerator.is_available():
            raise RuntimeError(
                "TPU training requires a TPU runtime with a compatible torch_xla installation. "
                "Install PyTorch/XLA for this TPU VM using the official instructions: "
                "https://docs.pytorch.org/xla/"
            )

    precision = "bf16-mixed" if bf16 else safe_get(config, "training.precision", "32-true")
    if accelerator == "tpu" and precision == "16-mixed":
        raise ValueError("FP16 mixed precision is not supported for TPU runs; use BF16 instead.")
    if accelerator == "cpu" and precision != "32-true":
        raise RuntimeError("Mixed precision was requested, but the CPU backend is selected.")

    if devices_override is not None:
        devices = devices_override
    elif accelerator == "tpu":
        devices = safe_get(config, "training.tpu_devices", 8)
    elif accelerator == "cuda":
        devices = safe_get(config, "training.gpus", 1)
    else:
        devices = 1
    if accelerator == "tpu":
        if isinstance(devices, (list, tuple)):
            if len(devices) != 1:
                raise ValueError("For TPU, pass a single --devices value representing the core count (1-8).")
            devices = devices[0]
        if not isinstance(devices, int) or not 1 <= devices <= 8:
            raise ValueError("TPU device count must be an integer from 1 to 8 for a TPU v5e-8.")

    data_cfg = dict(config["datamodule"]["data_config"])
    if accelerator == "tpu":
        data_cfg["pin_memory"] = False
        config["datamodule"]["data_config"]["pin_memory"] = False
    model_cfg = config["audionet"]["audionet_config"]

    info("instantiating datamodule <VideoPretrainDataModule>")
    datamodule = VideoPretrainDataModule(**data_cfg)
    datamodule.setup()
    train_loader, val_loader = datamodule.make_loader

    info("instantiating model <VideoEncoder>")
    model = VideoEncoder(**model_cfg)

    exp_name = safe_get(
        config,
        "exp.exp_name",
        f"videoencoder-pretrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config.setdefault("main_args", {})
    config["main_args"]["exp_dir"] = exp_dir
    config["runtime"] = {
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
    }
    with open(os.path.join(exp_dir, "conf.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    system = VideoEncoderPretrainSystem(
        model=model,
        optimizer_cfg=config["optimizer"],
        scheduler_cfg=config.get("scheduler"),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    checkpoint = ModelCheckpoint(
        dirpath=exp_dir,
        filename="{epoch}",
        monitor="val/loss",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks = [checkpoint]

    early_stop_cfg = safe_get(config, "training.early_stop", None)
    if early_stop_cfg:
        callbacks.append(EarlyStopping(**early_stop_cfg))

    trainer_kwargs = dict(
        max_epochs=safe_get(config, "training.epochs", 100),
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        limit_train_batches=safe_get(config, "training.limit_train_batches", 1.0),
        gradient_clip_val=safe_get(config, "training.gradient_clip_val", 5.0),
        logger=build_logger(config, exp_name),
        sync_batchnorm=(
            safe_get(config, "training.sync_batchnorm", True)
            if accelerator != "tpu"
            else False
        ),
    )
    world_size = len(devices) if isinstance(devices, (list, tuple)) else int(devices)
    if accelerator == "tpu":
        from pytorch_lightning.strategies.xla import XLAStrategy

        trainer_kwargs["strategy"] = XLAStrategy()
    elif accelerator == "cuda" and world_size > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(system)
    info("finished pretraining")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w", encoding="utf-8") as f:
        json.dump(best_k, f, indent=2, ensure_ascii=False)

    save_obj = {
        "state_dict": system.video_encoder.state_dict(),
        "model_name": system.video_encoder.__class__.__name__,
        "model_config": model_cfg,
    }
    torch.save(save_obj, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="configs/videoencoder_pretrain.yml",
        help="Path to video encoder pretraining config yaml.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "tpu"],
        default=None,
        help="Training device backend. TPU uses PyTorch/XLA (8 cores by default).",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="GPU device IDs, or a single TPU core count from 1 to 8.",
    )
    parser.add_argument("--bf16", action="store_true", help="Use mixed BF16 precision.")
    args = parser.parse_args()

    with open(args.conf_dir, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    main(config, accelerator_override=args.accelerator, devices_override=args.devices, bf16=args.bf16)
