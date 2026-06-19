from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from task2_act.metrics import action_l1_metrics, save_metrics
from task2_act.plots import plot_metrics


def parse_model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model argument must look like name=/path/to/checkpoint")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Model name is empty")
    return name, Path(path)


def make_delta_timestamps(indices: list[int] | None, fps: int) -> list[float]:
    if indices is None:
        return [0.0]
    return [idx / fps for idx in indices]


def evaluate_one_model(args: argparse.Namespace, model_name: str, checkpoint: Path) -> dict:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))

    if args.ta_eval_root:
        from task2_act.ta_lerobot_v21 import TASplitDataset

        dataset = TASplitDataset(
            args.ta_eval_root,
            chunk_size=policy.config.chunk_size,
            max_frames=args.target_frames,
            seed=args.seed,
        )
    else:
        if args.eval_repo_id is None or args.eval_root is None:
            raise ValueError("Use --ta-eval-root for TA data or provide both --eval-repo-id and --eval-root.")

        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        metadata = LeRobotDatasetMetadata(args.eval_repo_id, root=args.eval_root)
        delta_timestamps = {"action": make_delta_timestamps(policy.config.action_delta_indices, metadata.fps)}
        delta_timestamps.update(
            {
                key: make_delta_timestamps(policy.config.observation_delta_indices, metadata.fps)
                for key in policy.config.image_features
            }
        )
        dataset = LeRobotDataset(args.eval_repo_id, root=args.eval_root, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    preds = []
    targets = []
    masks = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"eval {model_name}")):
            target = batch["action"].detach().cpu()
            mask = ~batch.get("action_is_pad", torch.zeros(target.shape[:2], dtype=torch.bool)).detach().cpu()
            processed = preprocessor(batch)
            pred_norm = policy.predict_action_chunk(processed)
            pred = postprocessor.process_action(pred_norm).detach().cpu()

            preds.append(pred.numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
            if args.max_batches and batch_idx + 1 >= args.max_batches:
                break

    return action_l1_metrics(np.concatenate(preds), np.concatenate(targets), np.concatenate(masks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ACT checkpoints on CALVIN D with Action L1.")
    parser.add_argument("--eval-repo-id", default=None)
    parser.add_argument("--eval-root", type=Path, default=None)
    parser.add_argument("--ta-eval-root", type=Path, default=None)
    parser.add_argument("--target-frames", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already has files: {args.output_dir}")
    metrics_by_model = {
        model_name: evaluate_one_model(args, model_name, checkpoint) for model_name, checkpoint in args.model
    }
    save_metrics(args.output_dir, metrics_by_model)
    plot_metrics(args.output_dir, metrics_by_model)


if __name__ == "__main__":
    main()
