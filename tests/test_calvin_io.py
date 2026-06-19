from __future__ import annotations

from pathlib import Path

from task2_act.calvin_io import (
    CalvinSplit,
    frame_to_lerobot,
    load_scene_ranges,
    load_sequence_ranges,
    select_sequences,
)
from task2_act.make_mock_calvin import write_split


def test_scene_ranges_and_sequence_selection(tmp_path: Path) -> None:
    split_dir = tmp_path / "training"
    write_split(split_dir, ["A", "B", "C"], frames_per_scene=12, seed=1)

    scene_ranges = load_scene_ranges(split_dir)
    sequences = load_sequence_ranges(split_dir, scene_ranges)
    selected = select_sequences(sequences, {"B"}, target_frames=8, seed=42)

    assert scene_ranges["B"] == [(12, 23)]
    assert selected
    assert all(seq.scene == "B" for seq in selected)
    assert sum(seq.length for seq in selected) >= 8


def test_frame_conversion_shapes(tmp_path: Path) -> None:
    split_dir = tmp_path / "validation"
    write_split(split_dir, ["D"], frames_per_scene=6, seed=2)

    calvin = CalvinSplit.open(split_dir)
    frame = frame_to_lerobot(calvin.load_frame(0), image_size=(200, 200))

    assert frame["observation.images.static"].shape == (200, 200, 3)
    assert frame["observation.images.gripper"].shape == (200, 200, 3)
    assert frame["observation.state"].shape == (15,)
    assert frame["action"].shape == (7,)
    assert frame["task"] == "calvin play data"
