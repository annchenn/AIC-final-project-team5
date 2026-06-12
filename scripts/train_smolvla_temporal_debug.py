#!/usr/bin/env python
"""Train SmolVLA with runtime temporal video frames as camera inputs.

This is方案B: keep the original LeRobot v3 video dataset unchanged and build
front/wrist temporal image keys inside __getitem__ at training time.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import json
import os

import torch

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.datasets.factory import IMAGENET_STATS
from lerobot.datasets.lerobot_dataset import LeRobotDataset


BASE_IMAGE_KEYS = ("front", "wrist")
OFFSETS = (0, -1, -2, -3)
EXPECTED_TEMPORAL_IMAGE_KEYS = {
    "observation.images.front_t0",
    "observation.images.front_t-1",
    "observation.images.front_t-2",
    "observation.images.front_t-3",
    "observation.images.wrist_t0",
    "observation.images.wrist_t-1",
    "observation.images.wrist_t-2",
    "observation.images.wrist_t-3",
}


_orig_make_dataset = lerobot_train.make_dataset
_orig_update_policy = lerobot_train.update_policy
_printed_debug_batch = False


def _temporal_key(camera: str, offset: int) -> str:
    return f"observation.images.{camera}_t{offset}"


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    return type(value).__name__


class TemporalVideoMetadata:
    """Metadata view that exposes temporal frames as separate image cameras."""

    def __init__(self, base_meta, use_imagenet_stats: bool):
        self._base_meta = base_meta
        self.info = deepcopy(base_meta.info)
        self.info["features"] = self._build_features(base_meta.features)
        self.info["video_path"] = None
        self.stats = self._build_stats(base_meta.stats, use_imagenet_stats)

    def __getattr__(self, name: str):
        return getattr(self._base_meta, name)

    @staticmethod
    def _build_features(base_features: dict[str, dict]) -> dict[str, dict]:
        features: dict[str, dict] = {}
        for key, feature in base_features.items():
            if key.startswith("observation.images."):
                continue
            features[key] = deepcopy(feature)

        for camera in BASE_IMAGE_KEYS:
            source_key = f"observation.images.{camera}"
            source_feature = deepcopy(base_features[source_key])
            source_feature["dtype"] = "image"
            source_feature.pop("video_info", None)
            source_feature.pop("info", None)
            for offset in OFFSETS:
                features[_temporal_key(camera, offset)] = deepcopy(source_feature)
        return features

    @staticmethod
    def _build_stats(base_stats: dict | None, use_imagenet_stats: bool) -> dict | None:
        if base_stats is None:
            return None

        stats = {key: value for key, value in deepcopy(base_stats).items() if not key.startswith("observation.images.")}
        for camera in BASE_IMAGE_KEYS:
            source_key = f"observation.images.{camera}"
            for offset in OFFSETS:
                target_key = _temporal_key(camera, offset)
                if use_imagenet_stats:
                    stats[target_key] = {
                        stats_type: torch.tensor(value, dtype=torch.float32)
                        for stats_type, value in IMAGENET_STATS.items()
                    }
                elif source_key in base_stats:
                    stats[target_key] = deepcopy(base_stats[source_key])
        return stats

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return []

    @property
    def camera_keys(self) -> list[str]:
        return self.image_keys

    @property
    def names(self) -> dict[str, list | dict]:
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict[str, tuple]:
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}


class TemporalVideoDataset:
    """Runtime wrapper over the original LeRobot video dataset.

    Base LeRobotDataset decodes each original camera into a 4-frame stack ordered
    [t-3, t-2, t-1, t0]. This wrapper exposes that stack as eight separate image
    feature keys so SmolVLA can treat temporal history as multiple cameras.
    """

    def __init__(self, base_dataset: LeRobotDataset, use_imagenet_stats: bool):
        self.base_dataset = base_dataset
        self.meta = TemporalVideoMetadata(base_dataset.meta, use_imagenet_stats)
        self.episodes = base_dataset.episodes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base_dataset[idx]

        for camera in BASE_IMAGE_KEYS:
            source_key = f"observation.images.{camera}"
            frames = item.pop(source_key)
            if frames.shape[0] != len(OFFSETS):
                raise RuntimeError(f"Expected {len(OFFSETS)} temporal frames for {source_key}, got {tuple(frames.shape)}")

            # LeRobot returns frames in delta timestamp order [-3, -2, -1, 0].
            stack_index_by_offset = {-3: 0, -2: 1, -1: 2, 0: 3}
            for offset in OFFSETS:
                item[_temporal_key(camera, offset)] = frames[stack_index_by_offset[offset]]

            item.pop(f"{source_key}_is_pad", None)

        return item

    @property
    def num_frames(self) -> int:
        return self.base_dataset.num_frames

    @property
    def num_episodes(self) -> int:
        return self.base_dataset.num_episodes

    @property
    def fps(self) -> int:
        return self.base_dataset.fps

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self):
        return self.base_dataset.hf_features


def _episodes_from_split_env() -> list[int] | None:
    split_path = os.environ.get("TRAIN_EPISODES_JSON")
    if split_path:
        data = json.loads(Path(split_path).read_text())
        return [int(ep) for ep in data["train_episodes"]]

    episodes = os.environ.get("TRAIN_EPISODES")
    if episodes:
        return [int(ep) for ep in episodes.split(",") if ep.strip()]
    return None


def make_temporal_video_dataset(cfg):
    if cfg.dataset.episodes is None:
        split_episodes = _episodes_from_split_env()
        if split_episodes is not None:
            cfg.dataset.episodes = split_episodes
            print(f"[temporal-smolvla] training on {len(split_episodes)} train episodes from split")
    else:
        print(f"[temporal-smolvla] using explicit episode list with {len(cfg.dataset.episodes)} episodes")

    fps = cfg.policy.fps if getattr(cfg.policy, "fps", None) else 30
    delta_timestamps: dict[str, list[float]] = {
        "observation.state": [0.0],
        "observation.images.front": [-3 / fps, -2 / fps, -1 / fps, 0.0],
        "observation.images.wrist": [-3 / fps, -2 / fps, -1 / fps, 0.0],
    }

    if cfg.policy.action_delta_indices is not None:
        delta_timestamps["action"] = [i / fps for i in cfg.policy.action_delta_indices]
    if cfg.policy.reward_delta_indices is not None:
        delta_timestamps["next.reward"] = [i / fps for i in cfg.policy.reward_delta_indices]

    video_backend = cfg.dataset.video_backend or "pyav"
    if video_backend != "pyav":
        print(f"[temporal-smolvla] forcing dataset.video_backend='pyav' for AV1 video decoding; got {video_backend!r}")
        video_backend = "pyav"

    base_dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
        video_backend=video_backend,
        tolerance_s=cfg.tolerance_s,
        download_videos=False,
    )
    dataset = TemporalVideoDataset(base_dataset, use_imagenet_stats=cfg.dataset.use_imagenet_stats)
    print("[temporal-smolvla] using runtime temporal video dataset")
    print(f"[temporal-smolvla] repo_id={cfg.dataset.repo_id}, root={cfg.dataset.root}, video_backend={video_backend}")
    print(f"[temporal-smolvla] temporal image keys={sorted(EXPECTED_TEMPORAL_IMAGE_KEYS)}")
    return dataset


def update_policy_with_temporal_debug(*args, **kwargs):
    global _printed_debug_batch

    batch = args[2] if len(args) >= 3 else kwargs["batch"]
    if not _printed_debug_batch:
        image_keys = sorted(key for key in batch if key.startswith("observation.images."))
        missing = sorted(EXPECTED_TEMPORAL_IMAGE_KEYS - set(image_keys))
        extra = sorted(set(image_keys) - EXPECTED_TEMPORAL_IMAGE_KEYS)

        print("[temporal-smolvla] forward batch temporal image keys:")
        for key in image_keys:
            print(f"  {key}: {_shape(batch[key])}")

        if missing or extra:
            raise RuntimeError(f"Temporal image key mismatch before policy.forward: missing={missing}, extra={extra}")
        _printed_debug_batch = True

    return _orig_update_policy(*args, **kwargs)


def main() -> None:
    lerobot_train.make_dataset = make_temporal_video_dataset
    lerobot_train.update_policy = update_policy_with_temporal_debug
    lerobot_train.main()


if __name__ == "__main__":
    main()
