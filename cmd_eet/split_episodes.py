#!/usr/bin/env python
"""Create episode-level train/validation splits for LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def _read_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open() as f:
        return json.load(f)


def _read_episodes(dataset_root: Path) -> pd.DataFrame:
    files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        info = _read_info(dataset_root)
        return pd.DataFrame(
            {
                "episode_index": list(range(int(info["total_episodes"]))),
                "length": [None] * int(info["total_episodes"]),
            }
        )
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _as_shell_list(values: list[int]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _split_random(episodes: list[int], k_folds: int, fold: int, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)
    fold_sizes = [len(shuffled) // k_folds] * k_folds
    for i in range(len(shuffled) % k_folds):
        fold_sizes[i] += 1
    start = sum(fold_sizes[:fold])
    end = start + fold_sizes[fold]
    val = sorted(shuffled[start:end])
    train = sorted(ep for ep in episodes if ep not in set(val))
    return train, val


def _split_tail(episodes: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    n_val = max(1, round(len(episodes) * val_fraction))
    ordered = sorted(episodes)
    return ordered[:-n_val], ordered[-n_val:]


def _split_by_length(df: pd.DataFrame, val_fraction: float) -> tuple[list[int], list[int]]:
    if "length" not in df.columns or df["length"].isna().all():
        raise ValueError("length split needs meta/episodes parquet with a non-empty 'length' column")
    n_val = max(1, round(len(df) * val_fraction))
    ordered = df.sort_values("length")["episode_index"].astype(int).tolist()
    val = sorted(ordered[-n_val:])
    train = sorted(ep for ep in ordered if ep not in set(val))
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["random", "tail", "length"], default="random")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shell", action="store_true", help="Print shell exports for the selected fold.")
    args = parser.parse_args()

    if args.k_folds < 2:
        raise ValueError("--k-folds must be at least 2")
    if not 0 <= args.fold < args.k_folds:
        raise ValueError("--fold must be in [0, k_folds)")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    df = _read_episodes(args.dataset_root)
    episodes = sorted(df["episode_index"].astype(int).tolist())
    if len(episodes) < 2:
        raise ValueError("Need at least two episodes for a train/validation split")

    if args.mode == "random":
        train, val = _split_random(episodes, args.k_folds, args.fold, args.seed)
    elif args.mode == "tail":
        train, val = _split_tail(episodes, args.val_fraction)
    else:
        train, val = _split_by_length(df, args.val_fraction)

    payload = {
        "dataset_root": str(args.dataset_root),
        "mode": args.mode,
        "k_folds": args.k_folds,
        "fold": args.fold,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "total_episodes": len(episodes),
        "train_episodes": train,
        "val_episodes": val,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    if args.shell:
        print(f"CV_TRAIN_EPISODES='{_as_shell_list(train)}'")
        print(f"CV_VAL_EPISODES='{_as_shell_list(val)}'")
        print(f"CV_SPLIT_FILE='{args.output}'")
        print(f"CV_TRAIN_COUNT='{len(train)}'")
        print(f"CV_VAL_COUNT='{len(val)}'")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
