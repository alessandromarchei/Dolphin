import argparse
import io
import json
from pathlib import Path

import lmdb
import soundfile as sf


def get_utterance_id(filename, speaker):
    parts = Path(filename).stem.split("_")

    if speaker == "s1":
        return f"{parts[0]}_{parts[1]}"
    elif speaker == "s2":
        return f"{parts[3]}_{parts[4]}"
    else:
        raise ValueError(speaker)


def process_split(lmdb_path, output_dir, split):
    lmdb_path = Path(lmdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        subdir=True,
    )

    # Get all audio keys from the LMDB.
    names = {
        "mix": [],
        "s1": [],
        "s2": [],
    }

    with env.begin() as txn:
        cursor = txn.cursor()

        for key, _ in cursor:
            key = key.decode("utf-8")

            for kind in ("mix", "s1", "s2"):
                prefix = f"audio/{kind}/"

                if key.startswith(prefix):
                    names[kind].append(key[len(prefix):])
                    break

    for kind in names:
        names[kind].sort()

    print(
        f"[{split}] "
        f"mix={len(names['mix'])}, "
        f"s1={len(names['s1'])}, "
        f"s2={len(names['s2'])}"
    )

    # They must correspond exactly.
    if not (
        names["mix"] == names["s1"] == names["s2"]
    ):
        raise RuntimeError(
            f"{split}: mix/s1/s2 filenames do not correspond"
        )

    infos = {
        "mix": [],
        "s1": [],
        "s2": [],
    }

    with env.begin() as txn:

        for i, filename in enumerate(names["mix"]):

            # --------------------------------------------------
            # MIX
            # --------------------------------------------------

            mix_bytes = txn.get(
                f"audio/mix/{filename}".encode()
            )

            if mix_bytes is None:
                raise KeyError(filename)

            with sf.SoundFile(io.BytesIO(mix_bytes)) as f:
                num_frames = len(f)

            # Virtual path.
            mix_path = f"/lmdb/{split}/mix/{filename}"

            infos["mix"].append([
                mix_path,
                num_frames
            ])

            # --------------------------------------------------
            # S1 / S2
            # --------------------------------------------------

            for speaker in ("s1", "s2"):

                audio_bytes = txn.get(
                    f"audio/{speaker}/{filename}".encode()
                )

                if audio_bytes is None:
                    raise KeyError(
                        f"audio/{speaker}/{filename}"
                    )

                with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                    src_frames = len(f)

                utt = get_utterance_id(filename, speaker)

                mouth_key = f"mouth/{utt}.npz"

                if txn.get(mouth_key.encode()) is None:
                    raise KeyError(
                        f"Missing {mouth_key}"
                    )

                # Again: these are identifiers, not real files.
                audio_path = (
                    f"/lmdb/{split}/{speaker}/{filename}"
                )

                mouth_path = (
                    f"/lmdb/{split}/mouths/{utt}.npz"
                )

                infos[speaker].append([
                    audio_path,
                    mouth_path,
                    src_frames
                ])

            if (i + 1) % 1000 == 0:
                print(
                    f"\r[{split}] {i+1}/{len(names['mix'])}",
                    end="",
                    flush=True
                )

    print()

    for kind in ("mix", "s1", "s2"):
        output_file = output_dir / f"{kind}.json"

        with output_file.open("w") as f:
            json.dump(infos[kind], f)

        print(
            f"  {output_file}: "
            f"{len(infos[kind])} entries"
        )

    env.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lmdb-root",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    for split in ("tr", "cv", "tt"):
        process_split(
            args.lmdb_root / f"{split}.lmdb",
            args.out_dir / split,
            split
        )


if __name__ == "__main__":
    main()