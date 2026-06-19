from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from task2_act.calvin_io import (
    CalvinSplit,
    SequenceRange,
    frame_to_lerobot,
    infer_equal_scene_ranges,
    load_scene_ranges,
    load_sequence_ranges,
    resolve_split_dir,
    select_sequences,
    write_manifest,
)
from task2_act.constants import FPS, IMAGE_SIZE, SEED, lerobot_features


def build_dataset(
    *,
    split_dir: Path,
    output_root: Path,
    repo_id: str,
    sequences: list[SequenceRange],
    fps: int,
    image_size: tuple[int, int],
    use_videos: bool,
) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output dataset directory already has files: {output_root}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    calvin_split = CalvinSplit.open(split_dir)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        robot_type="franka_panda_calvin",
        fps=fps,
        features=lerobot_features(image_size),
        use_videos=use_videos,
        vcodec="h264",
        image_writer_threads=4,
    )

    for seq in tqdm(sequences, desc=f"convert {repo_id}"):
        for frame_idx in range(seq.start, seq.end + 1):
            dataset.add_frame(frame_to_lerobot(calvin_split.load_frame(frame_idx), image_size))
        dataset.save_episode()
    dataset.finalize()


def plan_training_sequences(
    training_dir: Path,
    train_frames: int,
    seed: int,
) -> tuple[list[SequenceRange], list[SequenceRange]]:
    scene_ranges = load_scene_ranges(training_dir)
    if not scene_ranges:
        base_sequences = load_sequence_ranges(training_dir, {})
        scene_ranges = infer_equal_scene_ranges(base_sequences, ["A", "B", "C"])
    sequences = load_sequence_ranges(training_dir, scene_ranges)
    b_sequences = select_sequences(sequences, {"B"}, train_frames, seed)

    per_scene = train_frames // 3
    abc_sequences: list[SequenceRange] = []
    for offset, scene in enumerate(["A", "B", "C"]):
        abc_sequences.extend(select_sequences(sequences, {scene}, per_scene, seed + offset))
    abc_sequences = sorted(abc_sequences, key=lambda item: (item.start, item.end))
    return b_sequences, abc_sequences


def plan_eval_sequences(validation_dir: Path, eval_frames: int, seed: int) -> list[SequenceRange]:
    scene_ranges = load_scene_ranges(validation_dir)
    sequences = load_sequence_ranges(validation_dir, scene_ranges)
    scenes = {"D"} if any(seq.scene == "D" for seq in sequences) else None
    return select_sequences(sequences, scenes, eval_frames, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CALVIN task_ABC_D into LeRobot datasets for ACT.")
    parser.add_argument("--calvin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-frames", type=int, default=120_000)
    parser.add_argument("--eval-frames", type=int, default=40_000)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--no-videos", action="store_true", help="Store image files instead of encoded videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = resolve_split_dir(args.calvin_root, "training")
    validation_dir = resolve_split_dir(args.calvin_root, "validation")

    b_sequences, abc_sequences = plan_training_sequences(training_dir, args.train_frames, args.seed)
    d_sequences = plan_eval_sequences(validation_dir, args.eval_frames, args.seed)

    specs = [
        ("local/calvin_b_120k", "calvin_b_120k", training_dir, b_sequences),
        ("local/calvin_abc_120k", "calvin_abc_120k", training_dir, abc_sequences),
        ("local/calvin_d_eval_40k", "calvin_d_eval_40k", validation_dir, d_sequences),
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for repo_id, folder_name, split_dir, sequences in specs:
        dataset_root = args.output_root / folder_name
        build_dataset(
            split_dir=split_dir,
            output_root=dataset_root,
            repo_id=repo_id,
            sequences=sequences,
            fps=args.fps,
            image_size=tuple(args.image_size),
            use_videos=not args.no_videos,
        )
        write_manifest(
            dataset_root / "selection_manifest.json",
            sequences,
            {
                "repo_id": repo_id,
                "split_dir": str(split_dir),
                "fps": args.fps,
                "image_size": list(args.image_size),
            },
        )


if __name__ == "__main__":
    main()
