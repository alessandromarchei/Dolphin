import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import look2hear.datas
import look2hear.models
import look2hear.system
from look2hear.system import make_optimizer


torch.cuda.empty_cache()


def info(msg: str) -> None:
    print(f"[train] {msg}")


def safe_get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_logger(config: Dict[str, Any], exp_name: str):
    logging_cfg = safe_get(config, "logging", {}) or {}

    # ------------------------------------------------------------------
    # Weights & Biases
    # ------------------------------------------------------------------
    if logging_cfg.get("use_wandb", False):
        wandb_cfg = logging_cfg.get("wandb", {}) or {}

        project = wandb_cfg.get("project", "dolphin")
        entity = wandb_cfg.get("entity", None)
        save_dir = wandb_cfg.get(
            "save_dir",
            os.path.join(os.getcwd(), "Experiments", "wandb"),
        )

        os.makedirs(save_dir, exist_ok=True)

        info(
            f"Using WandbLogger "
            f"(project={project}, entity={entity}, name={exp_name})"
        )

        return WandbLogger(
            project=project,
            entity=entity,
            name=exp_name,
            save_dir=save_dir,
            log_model=False,
        )
    # ------------------------------------------------------------------
    # TensorBoard fallback
    # ------------------------------------------------------------------
    logger_dir = os.path.join(
        os.getcwd(),
        "Experiments",
        "tensorboard_logs",
    )
    os.makedirs(logger_dir, exist_ok=True)

    info("Using TensorBoardLogger")

    return TensorBoardLogger(
        logger_dir,
        name=exp_name,
    )

def main(config: Dict[str, Any]) -> None:
    if "datamodule" not in config:
        raise KeyError(
            "Config missing `datamodule`. Please add datamodule.data_name and "
            "datamodule.data_config."
        )

    data_name = config["datamodule"]["data_name"]
    data_cfg = config["datamodule"]["data_config"]
    info(f"Instantiating datamodule <{data_name}>")
    datamodule = getattr(look2hear.datas, data_name)(**data_cfg)
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    audionet_name = config["audionet"]["audionet_name"]
    audionet_cfg = config["audionet"]["audionet_config"]
    info(f"Instantiating AudioNet <{audionet_name}>")
    model_cfg = dict(audionet_cfg)
    model_cfg.setdefault("sample_rate", data_cfg["sample_rate"])
    model = getattr(look2hear.models, audionet_name)(**model_cfg)

    info(f"Instantiating Optimizer <{config['optimizer']['optim_name']}>")
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    scheduler_name = safe_get(config, "scheduler.sche_name", None)
    if scheduler_name:
        info(f"Instantiating Scheduler <{scheduler_name}>")
        if scheduler_name != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer=optimizer,
                **config["scheduler"]["sche_config"],
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, scheduler_name)(
                    optimizer,
                    len(train_loader) // data_cfg["batch_size"],
                    64,
                ),
                "interval": "step",
            }

    exp_name = safe_get(
        config,
        "exp.exp_name",
        f"dolphin-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    exp_dir = os.path.join(os.getcwd(), "Experiments", "checkpoint", exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config.setdefault("main_args", {})
    config["main_args"]["exp_dir"] = exp_dir
    with open(os.path.join(exp_dir, "conf.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    system_name = safe_get(config, "training.system", "AudioVisualLightningModuleAE")
    info(f"Instantiating System <{system_name}>")
    system = getattr(look2hear.system, system_name)(
        audio_model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    callbacks = []
    checkpoint = ModelCheckpoint(
        dirpath=exp_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    early_stop_cfg = safe_get(config, "training.early_stop", None)
    if early_stop_cfg:
        info("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**early_stop_cfg))

    gpus = safe_get(config, "training.gpus", 1) if torch.cuda.is_available() else None
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_kwargs = dict(
        max_epochs=safe_get(config, "training.epochs", 100),
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=accelerator,
        limit_train_batches=safe_get(config, "training.limit_train_batches", 1.0),
        gradient_clip_val=safe_get(config, "training.gradient_clip_val", 5.0),
        logger=build_logger(config, exp_name),
        sync_batchnorm=safe_get(config, "training.sync_batchnorm", True),
    )
    if torch.cuda.is_available():
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(system)
    info("Finished Training")

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w", encoding="utf-8") as f:
        json.dump(best_k, f, indent=2, ensure_ascii=False)

    if checkpoint.best_model_path:
        state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
        system.load_state_dict(state_dict=state_dict["state_dict"])
        system.cpu()

        if hasattr(system.audio_model, "serialize"):
            to_save = system.audio_model.serialize()
        else:
            to_save = {
                "state_dict": system.audio_model.state_dict(),
                "model_name": system.audio_model.__class__.__name__,
                "model_config": audionet_cfg,
            }
        torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    else:
        info("No best_model_path produced by checkpoint callback.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf_dir",
        default="configs/dolphin.yml",
        help="Path to training config yaml.",
    )
    args = parser.parse_args()

    with open(args.conf_dir, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    main(config)
