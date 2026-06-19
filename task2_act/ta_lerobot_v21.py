from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from task2_act.constants import ACTION_DIM, ACTION_NAMES, IMAGE_SIZE, STATE_DIM, STATE_NAMES

OBS_IMAGE = "observation.images.image"
OBS_WRIST_IMAGE = "observation.images.wrist_image"
OBS_STATE = "observation.state"
ACTION = "action"


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    length: int
    scene: str | None = None


class TASplitDataset(Dataset):
    """Reader for the TA-provided CALVIN LeRobot v2.1 split folders."""

    def __init__(
        self,
        root: str | Path,
        *,
        chunk_size: int,
        max_frames: int | None = None,
        seed: int = 42,
        image_size: tuple[int, int] = IMAGE_SIZE,
        cache_size: int = 8,
    ) -> None:
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.cache_size = cache_size
        self._cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

        if not (self.root / "meta" / "info.json").exists():
            raise FileNotFoundError(f"Missing TA LeRobot split metadata: {self.root / 'meta' / 'info.json'}")
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.episodes = self._load_episodes()
        self.selected_episodes = self._select_episodes(self.episodes, max_frames, seed)
        self.samples = [
            (episode.episode_index, frame_idx)
            for episode in self.selected_episodes
            for frame_idx in range(episode.length)
        ]

    @property
    def fps(self) -> int:
        return int(self.info.get("fps", 10))

    def _load_episodes(self) -> list[EpisodeMeta]:
        episodes_path = self.root / "meta" / "episodes.jsonl"
        episodes: list[EpisodeMeta] = []
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                episodes.append(
                    EpisodeMeta(
                        episode_index=int(item["episode_index"]),
                        length=int(item["length"]),
                        scene=item.get("scene"),
                    )
                )
        return episodes

    @staticmethod
    def _select_episodes(episodes: list[EpisodeMeta], max_frames: int | None, seed: int) -> list[EpisodeMeta]:
        if max_frames is None or max_frames <= 0:
            return sorted(episodes, key=lambda item: item.episode_index)

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(episodes))
        chosen: list[EpisodeMeta] = []
        frames = 0
        for idx in order:
            episode = episodes[int(idx)]
            chosen.append(episode)
            frames += episode.length
            if frames >= max_frames:
                break
        return sorted(chosen, key=lambda item: item.episode_index)

    def _episode_path(self, episode_index: int) -> Path:
        chunk_size = int(self.info.get("chunks_size", 1000))
        chunk = episode_index // chunk_size
        data_path = self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
        return self.root / data_path

    def _load_episode(self, episode_index: int) -> pd.DataFrame:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        path = self._episode_path(episode_index)
        df = pd.read_parquet(path)
        self._cache[episode_index] = df
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return df

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, frame_idx = self.samples[index]
        df = self._load_episode(episode_index)
        row = df.iloc[frame_idx]

        actions = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
        action_is_pad = np.ones((self.chunk_size,), dtype=bool)
        end = min(len(df), frame_idx + self.chunk_size)
        valid = end - frame_idx
        if valid > 0:
            actions[:valid] = np.stack(df.iloc[frame_idx:end]["actions"].to_numpy()).astype(np.float32)
            action_is_pad[:valid] = False

        return {
            OBS_IMAGE: image_to_tensor(row["image"], self.image_size),
            OBS_WRIST_IMAGE: image_to_tensor(row["wrist_image"], self.image_size),
            OBS_STATE: torch.from_numpy(np.asarray(row["state"], dtype=np.float32).copy()),
            ACTION: torch.from_numpy(actions),
            "action_is_pad": torch.from_numpy(action_is_pad),
        }

    def iter_selected_episode_frames(self):
        for episode in self.selected_episodes:
            df = self._load_episode(episode.episode_index)
            yield df


def image_to_tensor(value: object, image_size: tuple[int, int]) -> torch.Tensor:
    if isinstance(value, dict) and "bytes" in value:
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
    elif isinstance(value, np.ndarray):
        image = Image.fromarray(value.astype(np.uint8)).convert("RGB")
    else:
        raise TypeError(f"Unsupported image cell type: {type(value)}")

    if image.size != (image_size[1], image_size[0]):
        image = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = (np.asarray(image, dtype=np.float32) / 255.0).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_ta_dataset(
    split_roots: list[str | Path],
    *,
    chunk_size: int,
    target_frames: int | None,
    frames_per_split: int | None,
    seed: int,
) -> Dataset:
    datasets: list[TASplitDataset] = []
    for idx, root in enumerate(split_roots):
        max_frames = frames_per_split if frames_per_split is not None else target_frames
        datasets.append(TASplitDataset(root, chunk_size=chunk_size, max_frames=max_frames, seed=seed + idx))
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def iter_ta_splits(dataset: Dataset) -> list[TASplitDataset]:
    if isinstance(dataset, TASplitDataset):
        return [dataset]
    if isinstance(dataset, ConcatDataset):
        return [item for item in dataset.datasets if isinstance(item, TASplitDataset)]
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def compute_ta_stats(dataset: Dataset) -> dict[str, dict[str, np.ndarray]]:
    state_sum = np.zeros((STATE_DIM,), dtype=np.float64)
    state_sumsq = np.zeros((STATE_DIM,), dtype=np.float64)
    action_sum = np.zeros((ACTION_DIM,), dtype=np.float64)
    action_sumsq = np.zeros((ACTION_DIM,), dtype=np.float64)
    count = 0

    for split in iter_ta_splits(dataset):
        selected = {episode.episode_index for episode in split.selected_episodes}
        for episode in split.selected_episodes:
            if episode.episode_index not in selected:
                continue
            df = split._load_episode(episode.episode_index)
            states = np.stack(df["state"].to_numpy()).astype(np.float64)
            actions = np.stack(df["actions"].to_numpy()).astype(np.float64)
            state_sum += states.sum(axis=0)
            state_sumsq += np.square(states).sum(axis=0)
            action_sum += actions.sum(axis=0)
            action_sumsq += np.square(actions).sum(axis=0)
            count += len(df)

    if count == 0:
        raise ValueError("Cannot compute statistics from an empty TA dataset")

    state_mean = state_sum / count
    action_mean = action_sum / count
    state_std = np.sqrt(np.maximum(state_sumsq / count - np.square(state_mean), 1e-12))
    action_std = np.sqrt(np.maximum(action_sumsq / count - np.square(action_mean), 1e-12))

    return {
        OBS_STATE: {"mean": state_mean.astype(np.float32), "std": state_std.astype(np.float32)},
        ACTION: {"mean": action_mean.astype(np.float32), "std": action_std.astype(np.float32)},
    }


def ta_policy_features():
    from lerobot.configs.types import FeatureType, PolicyFeature

    input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])),
        OBS_WRIST_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE[0], IMAGE_SIZE[1])),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }
    return input_features, output_features


def stats_summary(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    return {
        key: {name: value.tolist() for name, value in sub.items()}
        for key, sub in stats.items()
    }


def axis_names() -> dict[str, list[str]]:
    return {OBS_STATE: STATE_NAMES, ACTION: ACTION_NAMES}
