#!/usr/bin/env python
"""Create an episode-level train/validation split for a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-val-episodes", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {args.val_ratio}")

    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    episodes = list(range(int(meta.total_episodes)))
    rng = random.Random(args.seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)

    val_count = max(args.min_val_episodes, round(len(episodes) * args.val_ratio))
    val_count = min(val_count, len(episodes) - 1)
    val_episodes = sorted(shuffled[:val_count])
    train_episodes = sorted(shuffled[val_count:])

    overlap = sorted(set(train_episodes) & set(val_episodes))
    if overlap:
        raise RuntimeError(f"Episode split overlap detected: {overlap}")

    split = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "total_episodes": len(episodes),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")
    print(f"[split] wrote {args.output}")
    print(f"[split] train episodes: {len(train_episodes)}")
    print(f"[split] val episodes: {len(val_episodes)}")
    print("[split] overlap: 0")


if __name__ == "__main__":
    main()
