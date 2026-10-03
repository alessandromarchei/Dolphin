import argparse
import json
import os
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

# Persistent torch.compile cache. Must be configured before importing torch.
COMPILE_CACHE = Path.home() / ".cache" / "torchinductor"
COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(COMPILE_CACHE))
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, OnExceptionCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import look2hear.datas
import look2hear.models
import look2hear.system
from look2hear.system import make_optimizer


def info(msg=""): print(f"[train] {msg}")


def safe_get(cfg: dict, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur: return default
        cur = cur[key]
    return cur


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f: return yaml.safe_load(f)


def save_yaml(data: dict, path: Path):
    with path.open("w", encoding="utf-8") as f: yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def create_unique_experiment(root: Path, requested: str):
    root.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        name = requested if i == 0 else f"{requested}_{i}"
        path = root / name
        if not path.exists():
            path.mkdir(parents=True)
            return name, path
        i += 1


def resolve_resume(value: str):
    """Accept either an experiment directory or a Lightning .ckpt."""
    p = Path(value).expanduser().resolve()
    if not p.exists(): raise FileNotFoundError(f"Resume target does not exist: {p}")

    if p.is_dir():
        exp_dir = p
        ckpt_dir = exp_dir / "checkpoints"
        candidates = [ckpt_dir / "last.ckpt", ckpt_dir / "exception.ckpt"]
        candidates += sorted(ckpt_dir.glob("periodic-*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
        ckpt = next((x for x in candidates if x.exists()), None)
        if ckpt is None: raise FileNotFoundError(f"No resumable checkpoint found in {ckpt_dir}")
    else:
        if p.suffix != ".ckpt": raise ValueError("--resume must point to an experiment directory or a .ckpt file")
        ckpt = p
        exp_dir = p.parent.parent if p.parent.name == "checkpoints" else None
        if exp_dir is None or not (exp_dir / "conf.yml").exists():
            raise ValueError(f"Cannot infer experiment directory from checkpoint: {p}")

    return exp_dir.name, exp_dir, ckpt


def build_logger(config: dict, exp_name: str, exp_dir: Path, resume: bool):
    cfg = safe_get(config, "logging", {}) or {}
    if cfg.get("use_wandb", False):
        wc = cfg.get("wandb", {}) or {}
        wandb_dir = exp_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        id_file = exp_dir / "wandb_run_id.txt"

        if resume:
            run_id = id_file.read_text().strip() if id_file.exists() else None
            if run_id is None: info("WARNING: wandb_run_id.txt missing; W&B will create a new run.")
        else:
            run_id = uuid.uuid4().hex[:8]
            id_file.write_text(run_id, encoding="utf-8")

        project, entity, offline = wc.get("project", "dolphin"), wc.get("entity"), bool(wc.get("offline", False))
        info(f"Logger: W&B | project={project} | name={exp_name} | id={run_id or 'new'}")
        return WandbLogger(project=project, entity=entity, name=exp_name, save_dir=str(wandb_dir),
                           offline=offline, log_model=False, id=run_id, resume="allow" if run_id else None)

    tb_dir = exp_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    info(f"Logger: TensorBoard | dir={tb_dir}")
    return TensorBoardLogger(save_dir=str(tb_dir), name="", version="")


def configure_precision(tf32: bool):
    if not torch.cuda.is_available(): return
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    info(f"TF32: {'enabled' if tf32 else 'disabled'}")


def unwrap_compiled(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def export_model_from_checkpoint(checkpoint_path: str, system, output_path: Path, model_cfg: dict):
    info(f"Exporting {checkpoint_path} -> {output_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    system.load_state_dict(ckpt["state_dict"], strict=True)
    system.cpu()
    model = unwrap_compiled(system.audio_model)
    if hasattr(model, "serialize"):
        payload = model.serialize()
    else:
        payload = {"state_dict": model.state_dict(), "model_name": model.__class__.__name__, "model_config": model_cfg}
    torch.save(payload, output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Dolphin training")
    p.add_argument("--conf_dir", default="configs/dolphin.yml", help="YAML config used for a new run.")
    p.add_argument("--resume", default=None, help="Resume from experiment directory or Lightning .ckpt.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None, help="Default: 1 for new runs; saved value for resume.")
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--bf16", action="store_true")
    prec.add_argument("--fp16", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default=None, choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    p.add_argument("--compile-dynamic", action="store_true")
    p.add_argument("--compile-fullgraph", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--devices", type=int, nargs="+", default=None)
    p.add_argument(
        "--visual-input",
        choices=["mouth_frames", "avhubert_embeddings"],
        default=None,
        help="Visual input source. Defaults to the YAML visual.input_type setting.",
    )
    p.add_argument(
        "--visual-embeddings-dir",
        default=None,
        help="Directory of per-clip AV-HuBERT .npy embeddings; supplying it selects AV-HuBERT mode by default.",
    )
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000, help="Periodic checkpoint interval in training batches. 0 disables.")
    return p.parse_args()


def main(args):
    # Experiment/config resolution. Resume uses the experiment's saved config as the source of truth.
    if args.resume:
        exp_name, exp_dir, resume_ckpt = resolve_resume(args.resume)
        saved_conf = exp_dir / "conf.yml"
        config = load_yaml(saved_conf)
        info(f"RESUME | experiment={exp_name} | checkpoint={resume_ckpt}")
    else:
        config = load_yaml(Path(args.conf_dir))
        requested = safe_get(config, "exp.exp_name", f"dolphin-{datetime.now():%Y%m%d-%H%M%S}")
        exp_name, exp_dir = create_unique_experiment(Path.cwd() / "Experiments", requested)
        resume_ckpt = None
        info(f"NEW RUN | requested={requested} | resolved={exp_name}")

    config = deepcopy(config)
    visual_cfg = config.setdefault("visual", {})
    if args.visual_input is not None:
        visual_cfg["input_type"] = args.visual_input
    elif args.visual_embeddings_dir is not None:
        visual_cfg["input_type"] = "avhubert_embeddings"
    if args.visual_embeddings_dir is not None:
        visual_cfg["embeddings_dir"] = args.visual_embeddings_dir
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("exp", {})["exp_name"] = exp_name
    config.setdefault("main_args", {})["exp_dir"] = str(exp_dir)

    # On resume, runtime defaults come from the saved config unless explicitly overridden on CLI.
    old_rt = config.get("runtime", {}) if args.resume else {}
    batch_size = args.batch_size if args.batch_size is not None else safe_get(config, "datamodule.data_config.batch_size")
    grad_accum = args.grad_accum if args.grad_accum is not None else int(old_rt.get("gradient_accumulation", 1))
    use_compile = args.compile or bool(old_rt.get("torch_compile", False))
    compile_mode = args.compile_mode or old_rt.get("compile_mode") or "default"
    compile_dynamic = args.compile_dynamic or bool(old_rt.get("compile_dynamic", False))
    compile_fullgraph = args.compile_fullgraph or bool(old_rt.get("compile_fullgraph", False))

    if args.bf16: precision = "bf16-mixed"
    elif args.fp16: precision = "16-mixed"
    elif args.resume: precision = old_rt.get("precision", "32-true")
    else: precision = "32-true"

    tf32 = args.tf32 or bool(old_rt.get("tf32", False))
    devices_override = args.devices if args.devices is not None else old_rt.get("devices") if args.resume else None

    if batch_size is not None: config["datamodule"]["data_config"]["batch_size"] = int(batch_size)
    if args.epochs is not None: config["training"]["epochs"] = args.epochs
    if grad_accum < 1: raise ValueError("--grad-accum must be >= 1")

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    configure_precision(tf32)

    if torch.cuda.is_available():
        accelerator = "cuda"
        devices = devices_override if devices_override is not None else safe_get(config, "training.gpus", [0])
    else:
        if precision != "32-true": raise RuntimeError("Mixed precision requested but CUDA is unavailable.")
        accelerator, devices = "cpu", 1

    world_size = len(devices) if isinstance(devices, (list, tuple)) else int(devices)

    # Data.
    data_name = config["datamodule"]["data_name"]
    data_cfg = dict(config["datamodule"]["data_config"])
    visual_input_type = visual_cfg.get("input_type", "mouth_frames")
    visual_embedding_dim = int(visual_cfg.get("embedding_dim", 1024))
    data_cfg.update(
        visual_input_type=visual_input_type,
        visual_embeddings_dir=visual_cfg.get("embeddings_dir"),
        visual_embedding_dim=visual_embedding_dim,
    )
    info(f"Data: {data_name}")
    datamodule = getattr(look2hear.datas, data_name)(**data_cfg)
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader
    physical_batch = int(data_cfg["batch_size"])
    effective_batch = physical_batch * grad_accum * world_size

    # Model.
    audionet_name = config["audionet"]["audionet_name"]
    audionet_cfg = config["audionet"]["audionet_config"]
    model_cfg = dict(audionet_cfg)
    model_cfg.setdefault("sample_rate", data_cfg["sample_rate"])
    model_cfg.update(
        visual_input_type=visual_input_type,
        visual_embedding_dim=visual_embedding_dim,
    )
    info(f"Model: {audionet_name}")
    model = getattr(look2hear.models, audionet_name)(**model_cfg)

    if use_compile:
        info(f"torch.compile: mode={compile_mode}, dynamic={compile_dynamic}, fullgraph={compile_fullgraph}")
        model = torch.compile(model, mode=compile_mode, dynamic=compile_dynamic, fullgraph=compile_fullgraph)
    else:
        info("torch.compile: disabled")

    # Optimizer/scheduler are constructed normally; Lightning replaces their state from ckpt_path on resume.
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])
    scheduler = None
    scheduler_name = safe_get(config, "scheduler.sche_name")
    if scheduler_name:
        if scheduler_name != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer=optimizer, **config["scheduler"]["sche_config"])
        else:
            steps_per_epoch = max(1, len(train_loader) // grad_accum)
            scheduler = {"scheduler": getattr(look2hear.system.schedulers, scheduler_name)(optimizer, steps_per_epoch, 64), "interval": "step"}

    system_name = safe_get(config, "training.system", "AudioVisualLightningModuleAE")
    system = getattr(look2hear.system, system_name)(audio_model=model, optimizer=optimizer, train_loader=train_loader,
                                                   val_loader=val_loader, test_loader=test_loader,
                                                   scheduler=scheduler, config=config)

    logger = build_logger(config, exp_name, exp_dir, resume=bool(args.resume))

    # Persist the exact runtime settings so a later --resume can reconstruct them.
    config["runtime"] = {
        "experiment_name": exp_name, "experiment_dir": str(exp_dir), "precision": precision, "tf32": tf32,
        "torch_compile": use_compile, "compile_mode": compile_mode if use_compile else None,
        "compile_dynamic": compile_dynamic if use_compile else False,
        "compile_fullgraph": compile_fullgraph if use_compile else False,
        "physical_batch_size": physical_batch, "gradient_accumulation": grad_accum,
        "world_size": world_size, "effective_batch_size": effective_batch,
        "accelerator": accelerator, "devices": devices,
    }
    save_yaml(config, exp_dir / "conf.yml")

    # Checkpoints:
    # - best.ckpt: best validation loss
    # - last.ckpt: latest epoch/checkpoint state
    # - periodic-*.ckpt: latest periodic safety checkpoint (only one kept)
    # - exception.ckpt: written on an exception/interrupt and removed after a clean finish
    best_cb = ModelCheckpoint(dirpath=str(checkpoint_dir), filename="best", monitor="val_loss/dataloader_idx_0",
                              mode="min", save_top_k=1, save_last=True, verbose=True, auto_insert_metric_name=False)
    callbacks = [best_cb, OnExceptionCheckpoint(dirpath=str(checkpoint_dir), filename="exception")]

    if args.save_every > 0:
        periodic_cb = ModelCheckpoint(dirpath=str(checkpoint_dir), filename="periodic-{step:09d}",
                                      every_n_train_steps=args.save_every, save_top_k=1,
                                      save_on_train_epoch_end=False, auto_insert_metric_name=False)
        callbacks.append(periodic_cb)

    early_stop_cfg = safe_get(config, "training.early_stop")
    if early_stop_cfg: callbacks.append(EarlyStopping(**early_stop_cfg))

    trainer_kwargs = dict(
        max_epochs=safe_get(config, "training.epochs", 100), callbacks=callbacks, default_root_dir=str(exp_dir),
        accelerator=accelerator, devices=devices, precision=precision, accumulate_grad_batches=grad_accum,
        gradient_clip_val=safe_get(config, "training.gradient_clip_val", 5.0),
        limit_train_batches=safe_get(config, "training.limit_train_batches", 1.0), logger=logger,
        sync_batchnorm=safe_get(config, "training.sync_batchnorm", True) if world_size > 1 else False,
        log_every_n_steps=args.log_every, num_sanity_val_steps=0,
    )
    if torch.cuda.is_available() and world_size > 1:
        trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    info("=" * 78)
    info(f"Experiment        : {exp_name}")
    info(f"Directory         : {exp_dir}")
    info(f"Resume            : {resume_ckpt or 'no'}")
    info(f"Model             : {audionet_name}")
    info(f"Device            : {accelerator} {devices}")
    info(f"Precision / TF32  : {precision} / {tf32}")
    info(f"torch.compile     : {use_compile} ({compile_mode if use_compile else '-'})")
    info(f"Batch             : {physical_batch} x accum {grad_accum} x GPUs {world_size} = {effective_batch}")
    info(f"Epochs            : {safe_get(config, 'training.epochs', 100)}")
    info(f"Safety checkpoint : every {args.save_every} train batches" if args.save_every else "Safety checkpoint : disabled")
    info(f"Compile cache     : {COMPILE_CACHE}")
    info("=" * 78)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(system, ckpt_path=str(resume_ckpt) if resume_ckpt else None)

    # Clean completion: record and export best/last.
    best_ckpt, last_ckpt = best_cb.best_model_path, best_cb.last_model_path
    checkpoint_info = {
        "best_model_path": best_ckpt,
        "best_model_score": best_cb.best_model_score.item() if best_cb.best_model_score is not None else None,
        "last_model_path": last_ckpt,
    }
    with (exp_dir / "checkpoints.json").open("w", encoding="utf-8") as f: json.dump(checkpoint_info, f, indent=2)

    if best_ckpt: export_model_from_checkpoint(best_ckpt, system, exp_dir / "best_model.pth", audionet_cfg)
    if last_ckpt: export_model_from_checkpoint(last_ckpt, system, exp_dir / "last_model.pth", audionet_cfg)

    info(f"Finished. Experiment: {exp_dir}")
    info(f"Best: {best_ckpt or 'n/a'}")
    info(f"Last: {last_ckpt or 'n/a'}")


if __name__ == "__main__":
    main(parse_args())
