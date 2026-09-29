import os
import sys
import importlib.util
import torch
import numpy as np
import argparse
from argparse import Namespace
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
AV_HUBERT_ROOT = os.path.join(REPO_ROOT, "av_hubert")
AVHUBERT_USER_DIR = os.path.join(AV_HUBERT_ROOT, "avhubert")
FAIRSEQ_ROOT = os.path.join(AV_HUBERT_ROOT, "fairseq")
AVHUBERT_UTILS_PATH = os.path.join(AVHUBERT_USER_DIR, "utils.py")


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(AV_HUBERT_ROOT)
add_path(FAIRSEQ_ROOT)


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}' from '{module_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


avhubert_utils = load_module_from_path("avhubert_local_utils", AVHUBERT_UTILS_PATH)
from fairseq import checkpoint_utils, utils as fairseq_utils
from fairseq.data.dictionary import Dictionary

try:
    from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
except Exception:
    SentencepieceBPE = None

try:
    import sentencepiece as spm
except Exception:
    spm = None


def patch_torch_load_for_legacy_checkpoints():
    original_torch_load = torch.load

    if getattr(original_torch_load, "_dolphin_legacy_patch", False):
        return

    def wrapped_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    wrapped_torch_load._dolphin_legacy_patch = True
    torch.load = wrapped_torch_load


def register_checkpoint_safe_globals():
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    safe_globals = [Dictionary]
    if SentencepieceBPE is not None:
        safe_globals.append(SentencepieceBPE)
    if spm is not None and hasattr(spm, "SentencePieceProcessor"):
        safe_globals.append(spm.SentencePieceProcessor)

    add_safe_globals(safe_globals)


def merge_shard_files(output_dir, merged_output_name, skip_filenames=None):
    merged = {}
    skip_filenames = set(skip_filenames or [])

    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith(".pth"):
            continue
        if filename in skip_filenames:
            continue

        part_path = os.path.join(output_dir, filename)
        print(f"Loading {part_path} ...")
        data = torch.load(part_path, map_location="cpu")

        for key in data:
            if key in merged:
                print("have duplicated key:", key)
        merged.update(data)

    merged_output_path = os.path.join(output_dir, merged_output_name)
    torch.save(merged, merged_output_path)
    print(len(merged.items()))
    print(f"Saved merged data to {merged_output_path}")


def load_avhubert_model(ckpt_path, user_dir):
    # Register AV-HuBERT with fairseq once.
    if "avhubert" not in sys.modules:
        fairseq_utils.import_user_module(
            Namespace(user_dir=user_dir)
        )

    patch_torch_load_for_legacy_checkpoints()
    register_checkpoint_safe_globals()

    print(f"Loading AV-HuBERT checkpoint: {ckpt_path}")

    models, saved_cfg, task = (
        checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
    )

    model = models[0]

    if hasattr(model, "decoder"):
        model = model.encoder.w2v_model

    model = model.cuda().eval()

    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop(
            (
                task.cfg.image_crop_size,
                task.cfg.image_crop_size,
            )
        ),
        avhubert_utils.Normalize(
            task.cfg.image_mean,
            task.cfg.image_std,
        ),
    ])

    print("AV-HuBERT loaded.")

    return model, transform



def extract_visual_feature(
    video_path,
    model,
    transform,
):
    frames = torch.tensor(
        np.load(video_path)["data"],
        dtype=torch.float32,
    )

    frames = transform(frames)

    frames = (
        frames
        .unsqueeze(0)
        .unsqueeze(0)
        .cuda(non_blocking=True)
    )

    with torch.inference_mode():
        feature, _ = model.extract_finetune(
            source={
                "video": frames,
                "audio": None,
            },
            padding_mask=None,
            output_layer=None,
        )

    return feature.squeeze(0)

# def extract_visual_feature(video_path, ckpt_path, user_dir, is_finetune_ckpt=False):
#     # Import AV-HuBERT tasks/models only once via fairseq's user_dir hook.
#     if "avhubert" not in sys.modules:
#         fairseq_utils.import_user_module(Namespace(user_dir=user_dir))
#     patch_torch_load_for_legacy_checkpoints()
#     register_checkpoint_safe_globals()
#     models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

#     transform = avhubert_utils.Compose([
#         avhubert_utils.Normalize(0.0, 255.0),
#         avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
#         avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std)
#     ])

#     frames = torch.tensor(np.load(video_path)["data"], dtype=torch.float32)
#     frames = transform(frames)
#     frames = frames.unsqueeze(0).unsqueeze(0).cuda()

#     model = models[0]
#     if hasattr(models[0], "decoder"):
#         model = models[0].encoder.w2v_model
#     model.cuda().eval()

#     with torch.no_grad():
#         feature, _ = model.extract_finetune(
#             source={"video": frames, "audio": None},
#             padding_mask=None,
#             output_layer=None,
#         )
#         feature = feature.squeeze(0)
#     return feature


class VideoDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return self.file_list[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract AV-HuBERT mouth features with distributed inference."
    )
    parser.add_argument(
        "--in_dirs",
        nargs="+",
        default=[
            "/gpfs-flash/hulab/public_datasets/audio_datasets/lrs2/mouths",
            "/gpfs/hulab/public_datasets/audio_datasets/lrs3_rebuild/mouths",
            "/gpfs/hulab/public_datasets/audio_datasets/vox2/mouths",
        ],
        help="Input directories containing mouth npz files.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save extracted feature shards.",
    )
    parser.add_argument(
        "--output_prefix",
        default="videodata_large_vox",
        help="Prefix of saved shard files.",
    )
    parser.add_argument(
        "--ckpt_path",
        default="/gpfs/hulab/gaokejun/av_hubert/pretrained_zoo/large_vox_433h.pt",
        help="Path to the AV-HuBERT checkpoint.",
    )
    parser.add_argument(
        "--user_dir",
        default=AVHUBERT_USER_DIR,
        help="Path to the AV-HuBERT user_dir.",
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        help="Distributed backend.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--merged_output_name",
        default="videodata_large_3datasets.pth",
        help="Filename of the merged output saved in output_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed inference.
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Input mouth files.
    results = {}

    file_list = []
    for in_dir in args.in_dirs:
        file_list += [os.path.join(in_dir, f) for f in os.listdir(in_dir)]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_prefix}_{rank}.pth")

    # Distributed sampling.
    dataset = VideoDataset(file_list)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )


    model, transform = load_avhubert_model(
        args.ckpt_path,
        args.user_dir,
    )


    for video_paths in tqdm(
        dataloader,
        disable=(rank != 0),
    ):
        for video_path in video_paths:
            feature = extract_visual_feature(
                video_path,
                model,
                transform,
            )

            results[
                os.path.abspath(video_path)
            ] = feature.cpu()

    torch.save(results, output_path)
    dist.barrier()

    if rank == 0:
        merge_shard_files(
            output_dir=args.output_dir,
            merged_output_name=args.merged_output_name,
            skip_filenames={args.merged_output_name},
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
