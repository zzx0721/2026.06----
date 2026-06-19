from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from task2_act.constants import ACTION_NAMES


def action_l1_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected arrays shaped batch x horizon x action_dim, got {pred.shape}")

    diff = np.abs(pred - target)
    if mask is None:
        mask = np.ones(pred.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != pred.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match batch/horizon {pred.shape[:2]}")

    valid = diff[mask]
    if valid.size == 0:
        raise ValueError("No valid action entries after applying mask")

    per_dim = []
    for dim, name in enumerate(ACTION_NAMES[: pred.shape[-1]]):
        per_dim.append({"name": name, "l1": float(diff[..., dim][mask].mean())})

    per_horizon = []
    for horizon_idx in range(pred.shape[1]):
        h_mask = mask[:, horizon_idx]
        if h_mask.any():
            value = float(diff[:, horizon_idx, :][h_mask].mean())
        else:
            value = float("nan")
        per_horizon.append({"horizon": horizon_idx + 1, "l1": value})

    return {
        "overall_l1": float(valid.mean()),
        "per_dim_l1": per_dim,
        "per_horizon_l1": per_horizon,
        "num_valid_action_vectors": int(mask.sum()),
    }


def save_metrics(output_dir: Path, metrics_by_model: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_by_model, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "overall_l1", "num_valid_action_vectors"])
        for model_name, metrics in metrics_by_model.items():
            writer.writerow([model_name, metrics["overall_l1"], metrics["num_valid_action_vectors"]])

    with (output_dir / "per_dim_l1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "action_dim", "l1"])
        for model_name, metrics in metrics_by_model.items():
            for item in metrics["per_dim_l1"]:
                writer.writerow([model_name, item["name"], item["l1"]])

    with (output_dir / "per_horizon_l1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "horizon", "l1"])
        for model_name, metrics in metrics_by_model.items():
            for item in metrics["per_horizon_l1"]:
                writer.writerow([model_name, item["horizon"], item["l1"]])
