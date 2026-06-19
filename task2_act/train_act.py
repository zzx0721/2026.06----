from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def make_delta_timestamps(indices: list[int] | None, fps: int) -> list[float]:
    if indices is None:
        return [0.0]
    return [idx / fps for idx in indices]


def save_checkpoint(step: int, output_dir: Path, policy: Any, preprocessor: Any, postprocessor: Any) -> Path:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    preprocessor.save_pretrained(checkpoint_dir)
    postprocessor.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LeRobot ACT on a converted CALVIN dataset.")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--ta-split-root", type=Path, action="append", default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--frames-per-split", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already has files: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.ta_split_root:
        from task2_act.ta_lerobot_v21 import build_ta_dataset, compute_ta_stats, stats_summary, ta_policy_features

        input_features, output_features = ta_policy_features()
        dataset = build_ta_dataset(
            args.ta_split_root,
            chunk_size=args.chunk_size,
            target_frames=args.target_frames,
            frames_per_split=args.frames_per_split,
            seed=args.seed,
        )
        dataset_stats = compute_ta_stats(dataset)
        dataset_fps = 10
        (args.output_dir / "dataset_stats.json").write_text(
            json.dumps(stats_summary(dataset_stats), indent=2),
            encoding="utf-8",
        )
    else:
        if args.repo_id is None or args.dataset_root is None:
            raise ValueError("Use --ta-split-root for TA data or provide both --repo-id and --dataset-root.")

        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.feature_utils import dataset_to_policy_features
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        metadata = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
        features = dataset_to_policy_features(metadata.features)
        output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key, ft in features.items() if key not in output_features}
        dataset_stats = metadata.stats
        dataset_fps = metadata.fps

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        normalization_mapping={
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_lr_backbone=args.lr,
        device=str(device),
    )
    policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)
    policy.train()
    policy.to(device)

    if not args.ta_split_root:
        delta_timestamps = {"action": make_delta_timestamps(cfg.action_delta_indices, dataset_fps)}
        delta_timestamps.update(
            {key: make_delta_timestamps(cfg.observation_delta_indices, dataset_fps) for key in cfg.image_features}
        )
        dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())

    run = None
    if args.wandb_project:
        import wandb

        wandb_config = {key: str(value) for key, value in vars(args).items()}
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            mode=args.wandb_mode,
            config=wandb_config,
        )

    (args.output_dir / "train_args.json").write_text(
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    step = 0
    train_label = args.repo_id or "+".join(path.name for path in (args.ta_split_root or []))
    progress = tqdm(total=args.steps, desc=f"train {train_label}")
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "loss", "l1_loss", "kld_loss"])
    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)
            if step % args.log_freq == 0:
                values = {"train/loss": float(loss.item()), **{f"train/{k}": float(v) for k, v in loss_dict.items()}}
                progress.set_postfix(loss=f"{loss.item():.4f}")
                metrics_writer.writerow(
                    [
                        step,
                        float(loss.item()),
                        float(loss_dict.get("l1_loss", float("nan"))),
                        float(loss_dict.get("kld_loss", float("nan"))),
                    ]
                )
                metrics_file.flush()
                if run:
                    run.log(values, step=step)
            if step % args.save_freq == 0:
                save_checkpoint(step, args.output_dir, policy, preprocessor, postprocessor)
            if step >= args.steps:
                break

    save_checkpoint(step, args.output_dir, policy, preprocessor, postprocessor)
    metrics_file.close()
    progress.close()
    if run:
        run.finish()


if __name__ == "__main__":
    main()
