from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from task2_act.constants import (
    ACTION_DIM,
    ACTION_KEY,
    GRIPPER_IMAGE_KEY,
    IMAGE_SIZE,
    STATE_DIM,
    STATE_KEY,
    STATIC_IMAGE_KEY,
    TASK_DESCRIPTION,
)


@dataclass(frozen=True)
class SequenceRange:
    start: int
    end: int
    scene: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class NamingPattern:
    prefix: str
    digits: int
    suffix: str


@dataclass
class CalvinSplit:
    root: Path
    naming: NamingPattern

    @classmethod
    def open(cls, root: Path | str) -> "CalvinSplit":
        split_root = Path(root)
        if not split_root.is_dir():
            raise FileNotFoundError(f"CALVIN split directory not found: {split_root}")
        return cls(root=split_root, naming=discover_naming_pattern(split_root))

    def frame_path(self, frame_index: int) -> Path:
        name = f"{self.naming.prefix}{frame_index:0{self.naming.digits}d}{self.naming.suffix}"
        return self.root / name

    def load_frame(self, frame_index: int) -> dict[str, np.ndarray]:
        path = self.frame_path(frame_index)
        if not path.exists():
            raise FileNotFoundError(f"Missing CALVIN frame: {path}")
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def resolve_split_dir(calvin_root: Path | str, split: str) -> Path:
    root = Path(calvin_root)
    direct = root / split
    if direct.is_dir():
        return direct
    if root.name == split and root.is_dir():
        return root
    raise FileNotFoundError(f"Could not find split '{split}' under {root}")


def discover_naming_pattern(split_dir: Path) -> NamingPattern:
    for path in sorted(split_dir.glob("*.npz")):
        match = re.match(r"^(.*?)(\d+)(\.npz)$", path.name)
        if match:
            return NamingPattern(prefix=match.group(1), digits=len(match.group(2)), suffix=match.group(3))
    raise FileNotFoundError(f"No episode_*.npz files found in {split_dir}")


def load_sequence_ranges(split_dir: Path, scene_ranges: dict[str, list[tuple[int, int]]] | None = None) -> list[SequenceRange]:
    ep_file = split_dir / "ep_start_end_ids.npy"
    if not ep_file.exists():
        raise FileNotFoundError(f"Missing ep_start_end_ids.npy in {split_dir}")

    ep_ranges = np.load(ep_file, allow_pickle=False)
    sequences: list[SequenceRange] = []
    for start, end in ep_ranges:
        start_i = int(start)
        end_i = int(end)
        scene = assign_scene(start_i, end_i, scene_ranges or {})
        sequences.append(SequenceRange(start=start_i, end=end_i, scene=scene))
    return sequences


def infer_equal_scene_ranges(sequences: list[SequenceRange], scene_order: list[str]) -> dict[str, list[tuple[int, int]]]:
    if not sequences:
        raise ValueError("Cannot infer scene ranges from an empty sequence list")
    start = min(seq.start for seq in sequences)
    end = max(seq.end for seq in sequences)
    total = end - start + 1
    block = total // len(scene_order)
    ranges: dict[str, list[tuple[int, int]]] = {}
    cursor = start
    for idx, scene in enumerate(scene_order):
        block_end = end if idx == len(scene_order) - 1 else cursor + block - 1
        ranges[scene] = [(cursor, block_end)]
        cursor = block_end + 1
    return ranges


def load_scene_ranges(split_dir: Path) -> dict[str, list[tuple[int, int]]]:
    scene_file = split_dir / "scene_info.npy"
    if not scene_file.exists():
        return {}

    raw = np.load(scene_file, allow_pickle=True)
    item = raw.item() if getattr(raw, "shape", None) == () else raw
    ranges: dict[str, list[tuple[int, int]]] = {}

    if isinstance(item, dict):
        for key, value in item.items():
            scene = infer_scene_letter(str(key))
            if scene is None:
                continue
            ranges.setdefault(scene, []).extend(extract_intervals(value))
    else:
        raise ValueError(f"Unsupported scene_info.npy format in {scene_file}")

    return {scene: sorted(items) for scene, items in ranges.items()}


def infer_scene_letter(text: str) -> str | None:
    patterns = [
        r"calvin[_-]?scene[_-]?([ABCD])",
        r"scene[_-]?([ABCD])",
        r"^([ABCD])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def extract_intervals(value: object) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        intervals: list[tuple[int, int]] = []
        for nested in value.values():
            intervals.extend(extract_intervals(nested))
        return intervals

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1 and len(arr) == 2 and all(np.isscalar(x) for x in arr):
        return [(int(arr[0]), int(arr[1]))]
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [(int(start), int(end)) for start, end in arr]

    intervals = []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in value:
            intervals.extend(extract_intervals(item))
    return intervals


def assign_scene(start: int, end: int, scene_ranges: dict[str, list[tuple[int, int]]]) -> str | None:
    best_scene = None
    best_overlap = 0
    for scene, ranges in scene_ranges.items():
        overlap = sum(max(0, min(end, r_end) - max(start, r_start) + 1) for r_start, r_end in ranges)
        if overlap > best_overlap:
            best_scene = scene
            best_overlap = overlap
    return best_scene


def select_sequences(
    sequences: list[SequenceRange],
    scenes: set[str] | None,
    target_frames: int,
    seed: int,
) -> list[SequenceRange]:
    if scenes:
        pool = [seq for seq in sequences if seq.scene in scenes]
    else:
        pool = list(sequences)

    if not pool:
        scene_text = ",".join(sorted(scenes or [])) or "all"
        raise ValueError(f"No CALVIN sequences available for scene set: {scene_text}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    chosen: list[SequenceRange] = []
    frame_count = 0
    for idx in order:
        seq = pool[int(idx)]
        chosen.append(seq)
        frame_count += seq.length
        if target_frames > 0 and frame_count >= target_frames:
            break
    return sorted(chosen, key=lambda item: (item.start, item.end))


def resize_rgb(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {image.shape}")
    if image.shape[0] == size[0] and image.shape[1] == size[1]:
        return image
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((size[1], size[0]), Image.BILINEAR), dtype=np.uint8)


def frame_to_lerobot(frame: dict[str, np.ndarray], image_size: tuple[int, int] = IMAGE_SIZE) -> dict:
    required = ["rgb_static", "rgb_gripper", "robot_obs", "rel_actions"]
    missing = [key for key in required if key not in frame]
    if missing:
        raise KeyError(f"CALVIN frame missing keys: {missing}")

    state = np.asarray(frame["robot_obs"], dtype=np.float32).reshape(-1)
    action = np.asarray(frame["rel_actions"], dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"robot_obs must have shape ({STATE_DIM},), got {state.shape}")
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"rel_actions must have shape ({ACTION_DIM},), got {action.shape}")

    return {
        STATIC_IMAGE_KEY: resize_rgb(frame["rgb_static"], image_size),
        GRIPPER_IMAGE_KEY: resize_rgb(frame["rgb_gripper"], image_size),
        STATE_KEY: state,
        ACTION_KEY: action,
        "task": TASK_DESCRIPTION,
    }


def write_manifest(path: Path, sequences: list[SequenceRange], extra: dict) -> None:
    payload = {
        "extra": extra,
        "num_sequences": len(sequences),
        "num_frames": sum(seq.length for seq in sequences),
        "sequences": [asdict(seq) for seq in sequences],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
