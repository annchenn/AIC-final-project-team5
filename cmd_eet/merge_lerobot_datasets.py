#!/usr/bin/env python3
"""Merge small local LeRobot datasets by rewriting episode/global indices.

This is intentionally local/offline and does not touch source datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SOURCES = [
    "aic-finalproject-dataset-clear-fsm",
    "aic-finalproject-dataset-debug-grasp",
    "aic-finalproject-dataset-fsm-tuned-debug",
    "aic-finalproject-dataset-no-basket-fsm",
    "aic-finalproject-dataset-recovery-fsm-aug",
    "aic-finalproject-dataset-recovery-fsm-clear",
    "aic-finalproject-dataset-recovery-fsm-strict-nobasket",
    "aic-finalproject-dataset-stage-v3-debug",
]


def read_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text())


def _read_parquet_checked(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Bad parquet file: {path}: {exc}") from exc


def load_data(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        frames.append(_read_parquet_checked(path))
    if not frames:
        raise ValueError(f"No data parquet files in {root}")
    return pd.concat(frames, ignore_index=True)


def load_episodes(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        frames.append(_read_parquet_checked(path))
    if not frames:
        raise ValueError(f"No episode parquet files in {root}")
    return pd.concat(frames, ignore_index=True)


def list_video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, spec in info.get("features", {}).items() if spec.get("dtype") == "video"]


def stat_from_series(series: pd.Series) -> dict[str, list[Any]]:
    # Values are scalar columns here: timestamp/frame_index/episode_index/index/task_index.
    return {
        "min": [series.min().item() if hasattr(series.min(), "item") else series.min()],
        "max": [series.max().item() if hasattr(series.max(), "item") else series.max()],
        "mean": [float(series.mean())],
        "std": [float(series.std(ddof=0))],
        "count": [int(series.count())],
        "q01": [series.quantile(0.01).item() if hasattr(series.quantile(0.01), "item") else series.quantile(0.01)],
        "q10": [series.quantile(0.10).item() if hasattr(series.quantile(0.10), "item") else series.quantile(0.10)],
        "q50": [series.quantile(0.50).item() if hasattr(series.quantile(0.50), "item") else series.quantile(0.50)],
        "q90": [series.quantile(0.90).item() if hasattr(series.quantile(0.90), "item") else series.quantile(0.90)],
        "q99": [series.quantile(0.99).item() if hasattr(series.quantile(0.99), "item") else series.quantile(0.99)],
    }


def vector_stat_from_column(df: pd.DataFrame, column: str) -> dict[str, list[Any]]:
    vals = pd.DataFrame(df[column].tolist())
    return {
        "min": vals.min(axis=0).tolist(),
        "max": vals.max(axis=0).tolist(),
        "mean": vals.mean(axis=0).tolist(),
        "std": vals.std(axis=0, ddof=0).tolist(),
        "count": [int(len(vals))],
        "q01": vals.quantile(0.01, axis=0).tolist(),
        "q10": vals.quantile(0.10, axis=0).tolist(),
        "q50": vals.quantile(0.50, axis=0).tolist(),
        "q90": vals.quantile(0.90, axis=0).tolist(),
        "q99": vals.quantile(0.99, axis=0).tolist(),
    }


def weighted_image_stats(source_roots: list[Path], video_keys: list[str]) -> dict[str, Any]:
    # Keep image stats good enough for normalization: weighted mean/std plus global min/max.
    stats: dict[str, Any] = {}
    for key in video_keys:
        parts = []
        for root in source_roots:
            st = json.loads((root / "meta" / "stats.json").read_text())
            if key in st:
                parts.append(st[key])
        if not parts:
            continue
        total = sum(float(p.get("count", [1])[0]) for p in parts)
        first = parts[0]
        out = {k: first[k] for k in first.keys()}
        # These nested RGB stats are small; use per-source min/max envelopes and weighted moments.
        def flat3(v):
            return [float(v[i][0][0]) for i in range(3)]
        mins = [min(flat3(p["min"])[i] for p in parts) for i in range(3)]
        maxs = [max(flat3(p["max"])[i] for p in parts) for i in range(3)]
        means = [sum(flat3(p["mean"])[i] * float(p.get("count", [1])[0]) for p in parts) / total for i in range(3)]
        variances = []
        for i in range(3):
            accum = 0.0
            for p in parts:
                n = float(p.get("count", [1])[0])
                m = flat3(p["mean"])[i]
                sd = flat3(p["std"])[i]
                accum += n * (sd * sd + (m - means[i]) ** 2)
            variances.append(accum / total)
        def nest(v):
            return [[[float(x)]] for x in v]
        out["min"] = nest(mins)
        out["max"] = nest(maxs)
        out["mean"] = nest(means)
        out["std"] = nest([x ** 0.5 for x in variances])
        out["count"] = [int(total)]
        stats[key] = out
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/aicapstone/datasets/lerobot_cache/ann0000000")
    ap.add_argument("--output", default="aic-finalproject-dataset-debug-merged")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("sources", nargs="*", default=DEFAULT_SOURCES)
    args = ap.parse_args()

    root = Path(args.root)
    source_roots = [root / name for name in args.sources]
    missing = [p.name for p in source_roots if not (p / "meta" / "tasks.parquet").exists()]
    if missing:
        raise SystemExit(f"Missing or invalid source datasets: {missing}")

    out = root / args.output
    if out.exists():
        if not args.fresh:
            raise SystemExit(f"Output exists: {out}. Re-run with --fresh to replace it.")
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (out / "videos").mkdir(parents=True)

    first_info = read_info(source_roots[0])
    video_keys = list_video_keys(first_info)
    for key in video_keys:
        (out / "videos" / key / "chunk-000").mkdir(parents=True)

    data_parts = []
    episode_parts = []
    global_frame = 0
    global_episode = 0
    video_file_maps: dict[tuple[str, int, int, int], int] = {}
    copied_video_counts = {key: 0 for key in video_keys}

    for source_i, src in enumerate(source_roots):
        try:
            data = load_data(src).copy()
            episodes = load_episodes(src).copy()
        except Exception as exc:
            print(f"[skip] {src.name}: {exc}")
            continue

        episode_map = {old: global_episode + i for i, old in enumerate(sorted(data["episode_index"].unique()))}
        data["episode_index"] = data["episode_index"].map(episode_map).astype("int64")
        data["index"] = range(global_frame, global_frame + len(data))

        for key in video_keys:
            video_dir = src / "videos" / key
            for video in sorted(video_dir.glob("chunk-*/file-*.mp4")):
                old_chunk = int(video.parent.name.split("-")[-1])
                old_file = int(video.stem.split("-")[-1])
                new_file = copied_video_counts[key]
                copied_video_counts[key] += 1
                dst = out / "videos" / key / "chunk-000" / f"file-{new_file:03d}.mp4"
                shutil.copy2(video, dst)
                video_file_maps[(key, source_i, old_chunk, old_file)] = new_file

        episodes["episode_index"] = [global_episode + i for i in range(len(episodes))]
        episodes["data/chunk_index"] = 0
        episodes["data/file_index"] = 0
        episodes["dataset_from_index"] = [global_frame + int(x) for x in episodes["dataset_from_index"]]
        episodes["dataset_to_index"] = [global_frame + int(x) for x in episodes["dataset_to_index"]]
        for key in video_keys:
            chunk_col = f"videos/{key}/chunk_index"
            file_col = f"videos/{key}/file_index"
            if chunk_col in episodes and file_col in episodes:
                new_files = []
                for old_chunk, old_file in zip(episodes[chunk_col], episodes[file_col]):
                    new_files.append(video_file_maps[(key, source_i, int(old_chunk), int(old_file))])
                episodes[chunk_col] = 0
                episodes[file_col] = new_files
        episodes["meta/episodes/chunk_index"] = 0
        episodes["meta/episodes/file_index"] = 0

        data_parts.append(data)
        episode_parts.append(episodes)
        global_frame += len(data)
        global_episode += len(episodes)
        print(f"merged {src.name}: {len(episodes)} episodes, {len(data)} frames")

    merged_data = pd.concat(data_parts, ignore_index=True)
    merged_episodes = pd.concat(episode_parts, ignore_index=True)
    merged_data.to_parquet(out / "data" / "chunk-000" / "file-000.parquet", index=False)
    merged_episodes.to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    shutil.copy2(source_roots[0] / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")

    info = first_info.copy()
    info["total_episodes"] = int(global_episode)
    info["total_frames"] = int(global_frame)
    info["splits"] = {"train": f"0:{global_episode}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    stats = {
        "action": vector_stat_from_column(merged_data, "action"),
        "observation.state": vector_stat_from_column(merged_data, "observation.state"),
        "timestamp": stat_from_series(merged_data["timestamp"]),
        "frame_index": stat_from_series(merged_data["frame_index"]),
        "episode_index": stat_from_series(merged_data["episode_index"]),
        "index": stat_from_series(merged_data["index"]),
        "task_index": stat_from_series(merged_data["task_index"]),
    }
    stats.update(weighted_image_stats(source_roots, video_keys))
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    print(f"wrote {out}")
    print(f"total: {global_episode} episodes, {global_frame} frames")


if __name__ == "__main__":
    main()
