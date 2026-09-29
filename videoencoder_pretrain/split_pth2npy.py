#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PyTorch .pth dictionary containing "
            "AV-HuBERT features into individual .npy files."
        )
    )

    parser.add_argument(
        "input_pth",
        type=Path,
        help="Input .pth file containing {path: tensor} entries.",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output directory for .npy files. "
            "Default: directory containing input .pth."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_pth = args.input_pth.expanduser().resolve()

    if not input_pth.is_file():
        raise FileNotFoundError(
            f"Input file not found: {input_pth}"
        )

    if args.output_dir is None:
        output_dir = input_pth.parent
    else:
        output_dir = args.output_dir.expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("PTH -> NPY CONVERTER")
    print("=" * 80)
    print(f"Input:       {input_pth}")
    print(f"Output dir:  {output_dir}")
    print(f"Overwrite:   {args.overwrite}")
    print()

    print("Loading .pth file...")

    data = torch.load(
        input_pth,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict inside .pth, got {type(data)}"
        )

    print(f"Entries:     {len(data):,}")
    print()

    written = 0
    skipped = 0

    for source_path, feature in tqdm(
        data.items(),
        total=len(data),
        desc="Converting",
        unit="file",
    ):
        source_path = Path(source_path)

        # Example:
        #
        # 6164033984458952747_00051.npz
        #
        # becomes:
        #
        # 6164033984458952747_00051.npy

        output_path = (
            output_dir
            / f"{source_path.stem}.npy"
        )

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        if isinstance(feature, torch.Tensor):
            array = (
                feature
                .detach()
                .cpu()
                .numpy()
            )
        elif isinstance(feature, np.ndarray):
            array = feature
        else:
            array = np.asarray(feature)

        np.save(
            output_path,
            array,
            allow_pickle=False,
        )

        written += 1

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Total:       {len(data):,}")
    print(f"Written:     {written:,}")
    print(f"Skipped:     {skipped:,}")
    print(f"Output:      {output_dir}")


if __name__ == "__main__":
    main()