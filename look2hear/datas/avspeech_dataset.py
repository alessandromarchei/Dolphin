import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from .transform import get_preprocessing_pipelines

import io
import lmdb


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


class AVSpeechDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        is_train: bool = True,
        visual_input_type: str = "mouth_frames",
        visual_embeddings_dir: str = None,
        visual_embedding_dim: int = 1024,
        lmdb_path: str = None,
    ):
        super().__init__()
        if json_dir is None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError(f"{n_src} is not in [1, 2]")
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.is_train = is_train
        if visual_input_type not in {"mouth_frames", "avhubert_embeddings"}:
            raise ValueError(
                "visual_input_type must be 'mouth_frames' or 'avhubert_embeddings', "
                f"got {visual_input_type!r}"
            )
        self.visual_input_type = visual_input_type
        self.visual_embeddings_dir = (
            Path(visual_embeddings_dir).expanduser() if visual_embeddings_dir else None
        )
        self.visual_embedding_dim = visual_embedding_dim
        if self.visual_input_type == "avhubert_embeddings" and not self.visual_embeddings_dir:
            raise ValueError(
                "visual_embeddings_dir is required when visual_input_type='avhubert_embeddings'"
            )
        if self.visual_input_type == "avhubert_embeddings":
            if self.visual_embedding_dim < 1:
                raise ValueError("visual_embedding_dim must be a positive integer")
            if not self.visual_embeddings_dir.is_dir():
                raise FileNotFoundError(
                    f"AV-HuBERT embeddings directory not found: {self.visual_embeddings_dir}"
                )
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[
            "train" if is_train else "val"
        ]
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            self.fps_len = int(segment * 25)
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix.json")
        sources_json = [os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]]

        with open(mix_json, "r", encoding="utf-8") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r", encoding="utf-8") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt += 1
                        drop_len += mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt += 1
                        drop_len += mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        self.mix.append(mix_infos[i])
                        self.sources.append([src_inf[i] for src_inf in sources_infos])
            else:
                for i in range(len(mix_infos)):
                    self.mix.append(mix_infos[i])
                    self.sources.append([sources_infos[0][i], sources_infos[1][i]])
            print(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = orig_len

            self.lmdb_path = str(lmdb_path) if lmdb_path else None

            self._lmdb_env = None

    def __len__(self):
        return self.length

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            if self.lmdb_path is None:
                raise RuntimeError("LMDB path is not configured")

            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                subdir=True,
            )

        return self._lmdb_env


    def _lmdb_get(self, key: str) -> bytes:
        env = self._get_lmdb_env()

        with env.begin(write=False) as txn:
            value = txn.get(key.encode("utf-8"))

        if value is None:
            raise KeyError(
                f"LMDB key not found: {key} "
                f"(database={self.lmdb_path})"
            )

        return value


    @staticmethod
    def _utterance_from_source_path(path: str) -> str:
        """
        source filename example:

        6261660309214448793_00028_0.81423_
        5933122369233932002_00068_-0.81423.wav

        Directory tells us whether this is s1 or s2.
        """

        p = Path(path)

        stem_parts = p.stem.split("_")

        if p.parent.name == "s1":
            return f"{stem_parts[0]}_{stem_parts[1]}"

        if p.parent.name == "s2":
            return f"{stem_parts[3]}_{stem_parts[4]}"

        raise ValueError(f"Expected s1/s2 source path, got {path}")


    def _load_audio_lmdb(self, path: str):
        p = Path(path)

        kind = p.parent.name

        if kind not in {"mix", "s1", "s2"}:
            raise ValueError(f"Unknown audio type from path: {path}")

        key = f"audio/{kind}/{p.name}"

        wav_bytes = self._lmdb_get(key)

        audio, sample_rate = sf.read(
            io.BytesIO(wav_bytes),
            dtype="float32",
        )

        return audio, sample_rate


    def _load_mouth_lmdb(self, source_path: str):
        utt = self._utterance_from_source_path(source_path)

        key = f"mouth/{utt}.npz"

        npz_bytes = self._lmdb_get(key)

        with np.load(io.BytesIO(npz_bytes)) as npz:
            data = npz["data"]

        return data


    def _load_visual_input(self, source_json):
        if self.visual_input_type == "avhubert_embeddings":
            source_stem = Path(source_json[1]).stem
            embedding_path = Path(self.visual_embeddings_dir) / f"{source_stem}.npy"

            if not embedding_path.is_file():
                raise FileNotFoundError(
                    f"Precomputed AV-HuBERT embedding not found for {source_json[1]}: "
                    f"{embedding_path}"
                )

            visual_input = np.load(embedding_path)

            if visual_input.ndim != 2:
                raise ValueError(
                    f"Expected AV-HuBERT embeddings with shape "
                    f"(frames, {self.visual_embedding_dim}), "
                    f"got {visual_input.shape} in {embedding_path}"
                )

            if visual_input.shape[1] != self.visual_embedding_dim:
                raise ValueError(
                    f"Expected AV-HuBERT embedding dimension "
                    f"{self.visual_embedding_dim}, "
                    f"got {visual_input.shape[1]} in {embedding_path}"
                )

            if visual_input.shape[0] == 0:
                raise ValueError(
                    f"AV-HuBERT embedding contains no frames: {embedding_path}"
                )

            if self.fps_len is not None:
                visual_input = visual_input[: self.fps_len]

            return np.asarray(visual_input, dtype=np.float32)

        # mouth_frames mode: load the NPZ bytes from LMDB
        mouth = self._load_mouth_lmdb(source_json[0])

        # Same original Dolphin preprocessing
        return self.lipreading_preprocessing_func(mouth)

    def __getitem__(self, idx: int):
        eps = 1e-8
        if self.is_train and self.n_src == 1:
            rand_start = 0
            stop = None if self.test else rand_start + self.seg_len

            while True:
                s1_json = random.choice(self.sources)
                spk = s1_json[0].split("/")[-2]
                spk_id = ["s1", "s2"].index(spk)
                s1_split = s1_json[0].split("/")[-1].split("_")
                s1_name = [f"{s1_split[0]}_{s1_split[1]}", f"{s1_split[3]}_{s1_split[4]}"][spk_id]

                s2_json = random.choice(self.sources)
                spk = s2_json[0].split("/")[-2]
                spk_id = ["s1", "s2"].index(spk)
                s2_split = s2_json[0].split("/")[-1].split("_")
                s2_name = [f"{s2_split[0]}_{s2_split[1]}", f"{s2_split[3]}_{s2_split[4]}"][spk_id]

                if s1_name != s2_name:
                    break

            s1, _ = self._load_audio_lmdb(s1_json[0])
            s2, _ = self._load_audio_lmdb(s2_json[0])

            s1 = s1[rand_start:stop]
            s2 = s2[rand_start:stop]

            s1_mouth = self._load_visual_input(s1_json)[: self.fps_len]
            s2_mouth = self._load_visual_input(s2_json)[: self.fps_len]

            sources_json = [s1_json, s2_json]
            mouths = [s1_mouth, s2_mouth]
            sources = [torch.from_numpy(s1), torch.from_numpy(s2)]
            select_idx = np.random.randint(0, 2)
            mixture = torch.from_numpy(s1) + torch.from_numpy(s2)
            source = sources[select_idx]
            source_mouth = torch.from_numpy(np.asarray(mouths[select_idx], dtype=np.float32))

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=eps, std=m_std)
                source = normalize_tensor_wav(source, eps=eps, std=m_std)
            return mixture, source, source_mouth, sources_json[select_idx][0].split("/")[-1]

        if self.n_src == 1:
            rand_start = 0
            stop = None if self.test else rand_start + self.seg_len

            mix_source, _ = self._load_audio_lmdb(self.mix[idx][0])
            source, _ = self._load_audio_lmdb(self.sources[idx][0])

            mix_source = mix_source[rand_start:stop]
            source = source[rand_start:stop]
            source_mouth = self._load_visual_input(self.sources[idx])[: self.fps_len]

            source_mouth = torch.from_numpy(np.asarray(source_mouth, dtype=np.float32))
            source = torch.from_numpy(source)
            mixture = torch.from_numpy(mix_source)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=eps, std=m_std)
                source = normalize_tensor_wav(source, eps=eps, std=m_std)
            return mixture, source, source_mouth, self.mix[idx][0].split("/")[-1]

        rand_start = 0
        stop = None if self.test else rand_start + self.seg_len

        mix_source, _ = self._load_audio_lmdb(self.mix[idx][0])
        mix_source = mix_source[rand_start:stop]

        sources = []

        for src in self.sources[idx]:
            source, _ = self._load_audio_lmdb(src[0])
            source = source[rand_start:stop]
            sources.append(source)

        sources_mouths = torch.stack(
            [
                torch.from_numpy(
                    np.asarray(self._load_visual_input(src), dtype=np.float32)
                )
                for src in self.sources[idx]
            ]
        )[: self.fps_len]
        sources = torch.stack([torch.from_numpy(source) for source in sources])
        mixture = torch.from_numpy(mix_source)

        if self.normalize_audio:
            m_std = mixture.std(-1, keepdim=True)
            mixture = normalize_tensor_wav(mixture, eps=eps, std=m_std)
            sources = normalize_tensor_wav(sources, eps=eps, std=m_std)

        return mixture, sources, sources_mouths, self.mix[idx][0].split("/")[-1]


class AVSpeechDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        visual_input_type: str = "mouth_frames",
        visual_embeddings_dir: str = None,
        visual_embedding_dim: int = 1024,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        train_lmdb: str = None,
        valid_lmdb: str = None,
        test_lmdb: str = None,
    ) -> None:
        super().__init__()
        if train_dir is None or valid_dir is None or test_dir is None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError(f"{n_src} is not in [1, 2]")

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.n_src = n_src
        self.sample_rate = sample_rate
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.visual_input_type = visual_input_type
        self.visual_embeddings_dir = visual_embeddings_dir
        self.visual_embedding_dim = visual_embedding_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None
        self.train_lmdb = train_lmdb
        self.valid_lmdb = valid_lmdb
        self.test_lmdb = test_lmdb


    def setup(self) -> None:
        self.data_train = AVSpeechDataset(
            json_dir=self.train_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=True,
            visual_input_type=self.visual_input_type,
            visual_embeddings_dir=self.visual_embeddings_dir,
            visual_embedding_dim=self.visual_embedding_dim,
            lmdb_path=self.train_lmdb,
        )
        self.data_val = AVSpeechDataset(
            json_dir=self.valid_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=False,
            visual_input_type=self.visual_input_type,
            visual_embeddings_dir=self.visual_embeddings_dir,
            visual_embedding_dim=self.visual_embedding_dim,
            lmdb_path=self.valid_lmdb,
        )
        self.data_test = AVSpeechDataset(
            json_dir=self.test_dir,
            n_src=self.n_src,
            sample_rate=self.sample_rate,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            is_train=False,
            visual_input_type=self.visual_input_type,
            visual_embeddings_dir=self.visual_embeddings_dir,
            visual_embedding_dim=self.visual_embedding_dim,
            lmdb_path=self.test_lmdb,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test
