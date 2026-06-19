from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def read_episodes(path: Path) -> list[dict]:
    episodes = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            episodes.append(json.loads(line))
    return episodes


def select_episode_indices(episodes: list[dict], target_frames: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    chosen: list[int] = []
    frames = 0
    for idx in order:
        episode = episodes[int(idx)]
        chosen.append(int(episode["episode_index"]))
        frames += int(episode["length"])
        if frames >= target_frames:
            break
    return sorted(chosen)


def episode_pattern(split: str, episode_index: int, chunk_size: int) -> str:
    chunk = episode_index // chunk_size
    return f"{split}/data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def local_file_path(output_root: Path, filename: str) -> Path:
    return output_root / Path(filename)


def iter_exception_chain(exc: Exception):
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


def should_retry_download(exc: Exception) -> bool:
    markers = [
        "Consistency check failed",
        "Read timed out",
        "Network is unreachable",
        "client has been closed",
        "Temporary failure",
        "Please check your connection",
    ]
    retry_types = {
        "LocalEntryNotFoundError",
        "ReadTimeout",
        "ReadTimeoutError",
        "TimeoutError",
        "ConnectTimeout",
        "ConnectionError",
    }
    for item in iter_exception_chain(exc):
        if item.__class__.__name__ in retry_types:
            return True
        text = str(item)
        if any(marker in text for marker in markers):
            return True
    return False


def download_file(repo_id: str, revision: str, filename: str, output_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "filename": filename,
        "local_dir": output_root,
        "endpoint": os.environ.get("HF_ENDPOINT"),
    }
    for attempt in range(1, 6):
        force_download = attempt > 1
        try:
            return Path(hf_hub_download(**kwargs, force_download=force_download))
        except Exception as exc:
            if not should_retry_download(exc) or attempt == 5:
                raise
            wait_seconds = attempt * 5
            print(
                f"Retry download after error on attempt {attempt}/5: {filename} ({exc}). Sleep {wait_seconds}s.",
                flush=True,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Unreachable retry state for {filename}")


def download_many(repo_id: str, revision: str, filenames: list[str], output_root: Path) -> None:
    pending: list[str] = []
    seen = set()
    for filename in filenames:
        if filename in seen:
            continue
        seen.add(filename)
        if local_file_path(output_root, filename).exists():
            continue
        pending.append(filename)

    print(f"Need to download {len(pending)} files, skip {len(seen) - len(pending)} existing files.", flush=True)
    for idx, filename in enumerate(pending, start=1):
        print(f"[{idx}/{len(pending)}] {filename}", flush=True)
        download_file(repo_id, revision, filename, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the TA-provided CALVIN LeRobot split dataset.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="xiaoma26/calvin-lerobot")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--train-frames", type=int, default=120_000)
    parser.add_argument("--frames-per-split", type=int, default=40_000)
    parser.add_argument("--eval-frames", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "full":
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            local_dir=args.output_root,
            allow_patterns=["splitA/**", "splitB/**", "splitC/**", "splitD/**", ".gitattributes"],
        )
        (args.output_root / "download_manifest.json").write_text(
            json.dumps({"mode": "full", "repo_id": args.repo_id}, indent=2),
            encoding="utf-8",
        )
        return

    meta_patterns = [
        ".gitattributes",
        "splitA/meta/info.json",
        "splitA/meta/episodes.jsonl",
        "splitB/meta/info.json",
        "splitB/meta/episodes.jsonl",
        "splitC/meta/info.json",
        "splitC/meta/episodes.jsonl",
        "splitD/meta/info.json",
        "splitD/meta/episodes.jsonl",
    ]
    download_many(args.repo_id, args.revision, meta_patterns, args.output_root)

    selected: dict[str, set[int]] = {split: set() for split in ["splitA", "splitB", "splitC", "splitD"]}
    selected["splitB"].update(
        select_episode_indices(read_episodes(args.output_root / "splitB/meta/episodes.jsonl"), args.train_frames, args.seed)
    )
    for offset, split in enumerate(["splitA", "splitB", "splitC"]):
        selected[split].update(
            select_episode_indices(
                read_episodes(args.output_root / f"{split}/meta/episodes.jsonl"),
                args.frames_per_split,
                args.seed + offset,
            )
        )
    selected["splitD"].update(
        select_episode_indices(read_episodes(args.output_root / "splitD/meta/episodes.jsonl"), args.eval_frames, args.seed)
    )

    allow_patterns = list(meta_patterns)
    manifest = {"mode": "sampled", "repo_id": args.repo_id, "splits": {}}
    for split, episode_indices in selected.items():
        info = json.loads((args.output_root / split / "meta/info.json").read_text(encoding="utf-8"))
        chunk_size = int(info.get("chunks_size", 1000))
        allow_patterns.extend(episode_pattern(split, idx, chunk_size) for idx in sorted(episode_indices))
        manifest["splits"][split] = {
            "episodes": len(episode_indices),
            "episode_indices": sorted(episode_indices),
        }

    download_many(args.repo_id, args.revision, allow_patterns, args.output_root)
    (args.output_root / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
