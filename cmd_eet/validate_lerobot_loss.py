#!/usr/bin/env python
"""Evaluate supervised behavior-cloning loss on selected LeRobot episodes."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION


def _parse_episodes(raw: str | None) -> list[int] | None:
    if raw is None or raw == "":
        return None
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, list):
        raise ValueError("--episodes must be a Python list like '[0,1,2]'")
    return [int(v) for v in parsed]


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episodes", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means evaluate the full split.")
    parser.add_argument("--step", type=int, default=None, help="Training step associated with this evaluation.")
    parser.add_argument("--split", default="validation", help="Name of the evaluated split for logging.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    episodes = _parse_episodes(args.episodes)

    cfg = PreTrainedConfig.from_pretrained(
        args.policy_checkpoint,
        cli_overrides=[f"--device={args.device}"],
        local_files_only=True,
    )
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_checkpoint, config=cfg, local_files_only=True)
    # ACT's VAE loss only returns latent statistics in training mode. We still
    # wrap evaluation in torch.inference_mode(), so weights are not updated.
    if cfg.type == "act":
        policy.train()
    else:
        policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.policy_checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": torch.device(args.device).type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=torch.device(args.device).type == "cuda",
    )

    total_loss = 0.0
    total_batches = 0
    metric_sums: dict[str, float] = {}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", unit="batch"):
            batch = preprocessor(batch)
            loss, metrics = policy.forward(batch)
            metrics = metrics or {}
            total_loss += _to_float(loss)
            total_batches += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + _to_float(value)
            if args.max_batches and total_batches >= args.max_batches:
                break

    if total_batches == 0:
        raise RuntimeError("No validation batches were produced.")

    result = {
        "policy_type": cfg.type,
        "policy_checkpoint": str(args.policy_checkpoint),
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "episodes": episodes,
        "num_episodes": dataset.num_episodes,
        "num_frames": dataset.num_frames,
        "step": args.step,
        "split": args.split,
        "num_batches": total_batches,
        "loss": total_loss / total_batches,
        "metrics": {key: value / total_batches for key, value in metric_sums.items()},
    }
    line = json.dumps(result, sort_keys=True)
    print(line)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
