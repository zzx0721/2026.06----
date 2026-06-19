from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def write_split(root: Path, scenes: list[str], frames_per_scene: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    ep_ranges = []
    scene_info: dict[str, list[tuple[int, int]]] = {}
    frame_idx = 0

    for scene in scenes:
        scene_start = frame_idx
        frames_left = frames_per_scene
        while frames_left > 0:
            length = min(6, frames_left)
            start = frame_idx
            end = frame_idx + length - 1
            ep_ranges.append((start, end))
            for idx in range(start, end + 1):
                np.savez_compressed(
                    root / f"episode_{idx:07d}.npz",
                    rgb_static=rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8),
                    rgb_gripper=rng.integers(0, 255, size=(84, 84, 3), dtype=np.uint8),
                    robot_obs=rng.normal(size=(15,)).astype(np.float32),
                    rel_actions=rng.normal(size=(7,)).astype(np.float32),
                    scene_obs=rng.normal(size=(24,)).astype(np.float32),
                )
            frame_idx += length
            frames_left -= length
        scene_info[f"calvin_scene_{scene}"] = [(scene_start, frame_idx - 1)]

    np.save(root / "ep_start_end_ids.npy", np.asarray(ep_ranges, dtype=np.int64))
    np.save(root / "scene_info.npy", scene_info, allow_pickle=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny CALVIN-like dataset for local tests.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames-per-scene", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root already has files: {args.output_root}")
    write_split(args.output_root / "training", ["A", "B", "C"], args.frames_per_scene, args.seed)
    write_split(args.output_root / "validation", ["D"], args.frames_per_scene, args.seed + 1)


if __name__ == "__main__":
    main()
