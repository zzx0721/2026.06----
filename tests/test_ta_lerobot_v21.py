from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from task2_act.ta_lerobot_v21 import ACTION, OBS_IMAGE, OBS_STATE, OBS_WRIST_IMAGE, TASplitDataset, compute_ta_stats


def png_cell(size: tuple[int, int]) -> dict[str, bytes]:
    image = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    handle = BytesIO()
    image.save(handle, format="PNG")
    return {"bytes": handle.getvalue()}


def write_ta_split(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "total_episodes": 1,
        "fps": 10,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 3, "scene": "B"}) + "\n",
        encoding="utf-8",
    )
    df = pd.DataFrame(
        {
            "image": [png_cell((200, 200)) for _ in range(3)],
            "wrist_image": [png_cell((84, 84)) for _ in range(3)],
            "state": [np.ones((15,), dtype=np.float32) * idx for idx in range(3)],
            "actions": [np.ones((7,), dtype=np.float32) * idx for idx in range(3)],
        }
    )
    df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")


def test_ta_split_dataset_outputs_act_batch_keys(tmp_path: Path) -> None:
    write_ta_split(tmp_path)
    dataset = TASplitDataset(tmp_path, chunk_size=5, max_frames=3)
    item = dataset[1]
    stats = compute_ta_stats(dataset)

    assert item[OBS_IMAGE].shape == (3, 200, 200)
    assert item[OBS_WRIST_IMAGE].shape == (3, 200, 200)
    assert item[OBS_STATE].shape == (15,)
    assert item[ACTION].shape == (5, 7)
    assert item["action_is_pad"].tolist() == [False, False, True, True, True]
    assert stats[OBS_STATE]["mean"].shape == (15,)
    assert stats[ACTION]["mean"].shape == (7,)
