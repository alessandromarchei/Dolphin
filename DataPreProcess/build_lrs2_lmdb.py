#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

import lmdb


def extract_utterance_ids(wav_name: str):
    """
    Example:
    6261660309214448793_00028_0.81423_5933122369233932002_00068_-0.81423.wav

    returns:
        6261660309214448793_00028
        5933122369233932002_00068
    """
    stem = Path(wav_name).stem
    parts = stem.split("_")

    if len(parts) < 6:
        raise ValueError(f"Unexpected LRS2 mixture filename: {wav_name}")

    utt1 = f"{parts[0]}_{parts[1]}"
    utt2 = f"{parts[3]}_{parts[4]}"

    return utt1, utt2


def put_file(txn, key: str, path: Path):
    with path.open("rb") as f:
        data = f.read()

    if not txn.put(key.encode("utf-8"), data, overwrite=False):
        raise RuntimeError(f"Duplicate LMDB key: {key}")

    return len(data)


def build_split(root: Path, output_root: Path, split: str):
    audio_root = root / "wav16k" / "min" / split
    mouth_root = root / "mouths"

    output_path = output_root / f"{split}.lmdb"

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Remove it before rebuilding."
        )

    mix_files = sorted((audio_root / "mix").glob("*.wav"))
    s1_files = sorted((audio_root / "s1").glob("*.wav"))
    s2_files = sorted((audio_root / "s2").glob("*.wav"))

    print(f"\n[{split}]")
    print(f"mix: {len(mix_files)}")
    print(f"s1 : {len(s1_files)}")
    print(f"s2 : {len(s2_files)}")

    if not (len(mix_files) == len(s1_files) == len(s2_files)):
        raise RuntimeError(
            f"{split}: mix/s1/s2 counts differ: "
            f"{len(mix_files)}, {len(s1_files)}, {len(s2_files)}"
        )

    mix_by_name = {p.name: p for p in mix_files}
    s1_by_name = {p.name: p for p in s1_files}
    s2_by_name = {p.name: p for p in s2_files}

    names = sorted(mix_by_name)

    if set(names) != set(s1_by_name) or set(names) != set(s2_by_name):
        raise RuntimeError(f"{split}: filenames differ between mix/s1/s2")

    # Determine which mouth files this split actually needs.
    required_mouths = set()

    for name in names:
        utt1, utt2 = extract_utterance_ids(name)
        required_mouths.add(utt1)
        required_mouths.add(utt2)

    print(f"unique mouth utterances required: {len(required_mouths)}")

    missing = [
        utt for utt in required_mouths
        if not (mouth_root / f"{utt}.npz").is_file()
    ]

    if missing:
        print(f"ERROR: {len(missing)} missing mouth files.")
        for utt in missing[:20]:
            print("  ", utt)
        raise RuntimeError("Missing mouth files; aborting.")

    # Estimate required database size from actual files.
    audio_bytes = sum(
        p.stat().st_size
        for folder in ("mix", "s1", "s2")
        for p in (audio_root / folder).glob("*.wav")
    )

    mouth_bytes = sum(
        (mouth_root / f"{utt}.npz").stat().st_size
        for utt in required_mouths
    )

    actual_bytes = audio_bytes + mouth_bytes

    # Plenty of headroom. map_size is virtual address reservation,
    # NOT RAM allocation.
    map_size = max(
        1 << 30,
        int(actual_bytes * 1.5) + (256 << 20)
    )

    print(f"raw payload: {actual_bytes / 2**30:.2f} GiB")
    print(f"LMDB map_size: {map_size / 2**30:.2f} GiB")

    output_path.mkdir(parents=True)

    env = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=False,
    )

    entries = 0
    bytes_written = 0

    # Commit periodically instead of one gigantic transaction.
    txn = env.begin(write=True)

    try:
        for i, name in enumerate(names):
            for kind, mapping in (
                ("mix", mix_by_name),
                ("s1", s1_by_name),
                ("s2", s2_by_name),
            ):
                key = f"audio/{kind}/{name}"
                bytes_written += put_file(txn, key, mapping[name])
                entries += 1

            if (i + 1) % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(
                    f"\r[{split}] audio {i+1}/{len(names)}",
                    end="",
                    flush=True,
                )

        txn.commit()
        txn = None
        print()

        txn = env.begin(write=True)

        for i, utt in enumerate(sorted(required_mouths)):
            path = mouth_root / f"{utt}.npz"
            key = f"mouth/{utt}.npz"

            bytes_written += put_file(txn, key, path)
            entries += 1

            if (i + 1) % 500 == 0:
                txn.commit()
                txn = env.begin(write=True)
                print(
                    f"\r[{split}] mouths {i+1}/{len(required_mouths)}",
                    end="",
                    flush=True,
                )

        metadata = {
            "split": split,
            "num_mixtures": len(names),
            "num_mouths": len(required_mouths),
            "num_entries": entries,
        }

        txn.put(
            b"__metadata__",
            json.dumps(metadata).encode("utf-8"),
        )

        txn.commit()
        txn = None

        env.sync()

    finally:
        if txn is not None:
            txn.abort()
        env.close()

    print()
    print(f"[{split}] DONE")
    print(f"entries: {entries}")
    print(f"payload: {bytes_written / 2**30:.2f} GiB")
    print(f"output : {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./lrs2_lmdb"),
    )

    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()

    print("Input :", root)
    print("Output:", output)

    output.mkdir(parents=True, exist_ok=True)

    for split in ("tr", "cv", "tt"):
        build_split(root, output, split)


if __name__ == "__main__":
    main()