###
# Author: Kai Li
# Date: 2025-09-28
# LastEditors: Kai Li
# LastEditTime: 2026-04-13
###

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import warnings

import torch
import torchaudio
import yaml
import numpy as np

warnings.filterwarnings("ignore")

import look2hear.models
import look2hear.datas
from look2hear.metrics import ALLMetricsTracker

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text


class MetricsColumn(TextColumn):
    """A progress column that displays live evaluation metrics."""

    def __init__(self):
        super().__init__("")
        self._metrics = {}

    def update(self, metrics: dict):
        self._metrics = metrics

    def render(self, task) -> Text:
        if not self._metrics:
            return Text("")
        parts = [f"{k}: {v:.2f}" for k, v in self._metrics.items()]
        return Text(" | ".join(parts), style="cyan")


def main(config):
    # ---- paths ----
    exp_dir = os.path.join(
        os.getcwd(),
        "Experiments",
        "checkpoint",
        config["exp"]["exp_name"],
    )
    results_dir = os.path.join(exp_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- device ----
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- model ----
    audionet_name = config["audionet"]["audionet_name"]

    if config.get("use_hf_model", False):
        hf_model_id = config["hf_model_id"]
        print(f"Loading model from HuggingFace Hub: {hf_model_id}")
        model = getattr(look2hear.models, audionet_name).from_pretrained(
            hf_model_id, strict=False,
        )
    else:
        model_path = os.path.join(exp_dir, "best_model.pth")
        audionet_cfg = dict(config["audionet"]["audionet_config"])
        audionet_cfg["is_train"] = False

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model = getattr(look2hear.models, audionet_name)(
            sample_rate=config["datamodule"]["data_config"]["sample_rate"],
            **audionet_cfg,
        )
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # ---- data ----
    datamodule = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"],
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    # ---- metrics ----
    metrics = ALLMetricsTracker(
        save_file=os.path.join(results_dir, "metrics.csv"),
    )

    # ---- progress bar ----
    metrics_column = MetricsColumn()
    progress = Progress(
        TextColumn("[bold blue]Evaluating", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        metrics_column,
    )

    # ---- evaluation loop ----
    n_src = config["datamodule"]["data_config"].get("n_src", 1)
    sample_rate = config["datamodule"]["data_config"]["sample_rate"]

    with torch.no_grad(), progress:
        for idx in progress.track(range(len(test_set))):
            if n_src == 1:
                mix, source, mouth, key = test_set[idx]
                mix = mix.to(device)
                source = source.to(device)
                mouth = mouth.unsqueeze(0).unsqueeze(0).to(device)

                est_source = model(mix.unsqueeze(0), mouth)
                est_source = est_source.squeeze(0)

                metrics(
                    mix=mix,
                    clean=source.unsqueeze(0),
                    estimate=est_source,
                    key=key,
                )
            else:
                mix, sources, mouths, key = test_set[idx]
                mix = mix.to(device)
                sources = sources.to(device)
                mouths = mouths.unsqueeze(0).to(device)

                est_sources = model(mix.unsqueeze(0), mouths)
                est_sources = est_sources.squeeze(0)

                metrics(
                    mix=mix,
                    clean=sources.unsqueeze(0),
                    estimate=est_sources,
                    key=key,
                )

            if idx % 50 == 0:
                metrics_column.update(metrics.get_mean())

    metrics.final()

    # ---- print summary ----
    mean_metrics = metrics.get_mean()
    std_metrics = metrics.get_std()
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k in mean_metrics:
        print(f"  {k:>10s}: {mean_metrics[k]:.4f} +/- {std_metrics[k]:.4f}")
    print("=" * 60)
    print(f"Results saved to {results_dir}/metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Dolphin AVSS model")
    parser.add_argument(
        "--conf_dir",
        default="Experiments/checkpoint/LRS2-final-Dolphin-1gpu-batch4/conf.yml",
        help="Path to the experiment config (conf.yml saved during training).",
    )
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help="Load model from HuggingFace Hub instead of local checkpoint.",
    )
    parser.add_argument(
        "--hf_model_id",
        default="JusperLee/Dolphin",
        help="HuggingFace model ID (default: JusperLee/Dolphin).",
    )
    parser.add_argument(
        "--test_dir",
        default=None,
        help="Override test set directory from config.",
    )
    args = parser.parse_args()

    with open(args.conf_dir, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.use_hf:
        config["use_hf_model"] = True
        config["hf_model_id"] = args.hf_model_id

    if args.test_dir:
        config["datamodule"]["data_config"]["test_dir"] = args.test_dir

    main(config)
