#!/usr/bin/env python
"""Compute imitation validation loss for a SmolVLA checkpoint on val episodes."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from scripts.train_smolvla_temporal_debug import make_temporal_video_dataset

warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module="torchvision.io._video_deprecation_warning",
)


def _pretrained_dir(path: Path) -> Path:
    if (path / "pretrained_model").is_dir():
        return path / "pretrained_model"
    return path


def _existing_variant(path: Path) -> Path | None:
    """Return an existing path with common run-name separators swapped."""

    path_text = str(path)
    variants = []
    if "_" in path_text:
        variants.append(Path(path_text.replace("_", "-")))
    if "-" in path_text:
        variants.append(Path(path_text.replace("-", "_")))

    for variant in variants:
        if variant.exists():
            return variant
    return None


def _missing_path_message(label: str, path: Path) -> str:
    message = f"{label} does not exist: {path}"
    variant = _existing_variant(path)
    if variant is not None:
        message += f"\n  Did you mean: {variant}"
        return message

    if not path.parent.exists() and path.parent.parent.exists():
        normalized_parent_name = path.parent.name.replace("_", "").replace("-", "")
        for sibling in path.parent.parent.iterdir():
            normalized_sibling_name = sibling.name.replace("_", "").replace("-", "")
            if sibling.is_dir() and normalized_sibling_name == normalized_parent_name:
                sibling_path = sibling / path.name
                if sibling_path.exists():
                    message += f"\n  Did you mean: {sibling_path}"
                else:
                    message += f"\n  Similar directory exists but this file is missing: {sibling}"
                break

    return message


def _validate_input_paths(args: argparse.Namespace) -> None:
    errors = []

    if not args.split.is_file():
        errors.append(
            _missing_path_message("Validation split", args.split)
            + "\n  Create it first, for example:\n"
            + "    uv run python scripts/split_lerobot_episodes.py \\\n"
            + f"      --dataset-repo-id {args.dataset_repo_id} \\\n"
            + f"      --dataset-root {args.dataset_root} \\\n"
            + f"      --output {args.split}"
        )

    if not _pretrained_dir(args.checkpoint).is_dir():
        errors.append(_missing_path_message("Checkpoint", args.checkpoint))

    if not args.dataset_root.is_dir():
        errors.append(_missing_path_message("Dataset root", args.dataset_root))

    if errors:
        raise FileNotFoundError("\n\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means validate all val batches")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        _validate_input_paths(args)
    except FileNotFoundError as exc:
        parser.exit(2, f"error: {exc}\n")

    split = json.loads(args.split.read_text())
    val_episodes = [int(ep) for ep in split["val_episodes"]]
    if not val_episodes:
        raise ValueError("Validation split is empty")

    checkpoint = _pretrained_dir(args.checkpoint)
    if not checkpoint.is_dir():
        raise NotADirectoryError(checkpoint)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(checkpoint, local_files_only=True)
    policy.to(device)
    policy.eval()

    cfg_policy = policy.config
    if not isinstance(cfg_policy, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(cfg_policy).__name__}")

    cfg = SimpleNamespace(
        policy=cfg_policy,
        dataset=SimpleNamespace(
            repo_id=args.dataset_repo_id,
            root=str(args.dataset_root),
            episodes=val_episodes,
            revision=None,
            video_backend=args.video_backend,
            use_imagenet_stats=True,
        ),
        tolerance_s=0.0001,
    )
    dataset = make_temporal_video_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    total_to_run = min(len(loader), args.max_batches) if args.max_batches > 0 else len(loader)
    progress = tqdm(loader, total=total_to_run, desc="Validation", unit="batch")
    with torch.inference_mode():
        for batch in progress:
            batch_size = len(next(iter(batch.values())))
            batch = preprocessor(batch)
            loss, output = policy.forward(batch)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            total_batches += 1
            progress.set_postfix(loss=f"{total_loss / total_examples:.6f}")
            if args.max_batches > 0 and total_batches >= args.max_batches:
                break
    progress.close()

    if total_batches == 0 or total_examples == 0:
        raise RuntimeError("Validation produced zero batches")

    result = {
        "checkpoint": str(args.checkpoint),
        "pretrained_model": str(checkpoint),
        "split": str(args.split),
        "val_episodes": len(val_episodes),
        "val_batches": total_batches,
        "val_examples": total_examples,
        "validation_loss": total_loss / total_examples,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
