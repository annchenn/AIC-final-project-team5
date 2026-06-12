#!/usr/bin/env python3
"""Filter UMI object_poses.json episodes by initial distance from storage box."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BLOCK_NAMES = {"green_block", "blue_block", "red_block"}
BASKET_XY = (0.65, -0.55)
ANCHOR_XY = (0.35, 0.0)


def block_world_xy(obj: dict) -> tuple[float, float]:
    tvec = obj["tvec"]
    return ANCHOR_XY[0] + float(tvec[0]), ANCHOR_XY[1] + float(tvec[1])


def block_points(entry: dict) -> dict[str, tuple[float, float]]:
    return {
        obj["object_name"]: block_world_xy(obj)
        for obj in entry.get("objects", [])
        if obj.get("object_name") in BLOCK_NAMES
    }


def min_basket_distance(entry: dict) -> float:
    distances = []
    for x, y in block_points(entry).values():
        distances.append(math.hypot(x - BASKET_XY[0], y - BASKET_XY[1]))
    return min(distances) if distances else float("inf")


def min_pair_distance(entry: dict) -> float:
    pts = list(block_points(entry).items())
    distances = []
    for i, (_, a) in enumerate(pts):
        for _, b in pts[i + 1 :]:
            distances.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    return min(distances) if distances else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/merge/object_poses.json")
    parser.add_argument("--output", default="data/object_poses_no_basket.json")
    parser.add_argument("--min-distance", type=float, default=0.22, help="Minimum distance from any block to basket xy.")
    parser.add_argument("--min-pair-distance", type=float, default=0.12, help="Minimum xy distance between any pair of blocks.")
    parser.add_argument("--keep-non-full", action="store_true", help="Keep non-full entries unchanged in the output.")
    args = parser.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text())
    if not isinstance(data, list):
        raise SystemExit(f"{src}: expected top-level list")

    full = [entry for entry in data if isinstance(entry, dict) and entry.get("status") == "full"]
    rejected = []
    kept = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "full":
            if args.keep_non_full:
                kept.append(entry)
            continue
        basket_dist = min_basket_distance(entry)
        pair_dist = min_pair_distance(entry)
        if basket_dist < args.min_distance or pair_dist < args.min_pair_distance:
            reason = []
            if basket_dist < args.min_distance:
                reason.append(f"basket={basket_dist:.3f}")
            if pair_dist < args.min_pair_distance:
                reason.append(f"pair={pair_dist:.3f}")
            rejected.append((idx, basket_dist, pair_dist, ",".join(reason), entry.get("video_name", "")))
        else:
            kept.append(entry)

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(kept, indent=2))

    print(f"input={src}")
    print(f"output={dst}")
    print(f"full_episodes={len(full)}")
    print(f"kept_full={sum(1 for e in kept if isinstance(e, dict) and e.get('status') == 'full')}")
    print(f"rejected_full={len(rejected)}")
    print(f"min_basket_distance={args.min_distance:.3f}")
    print(f"min_pair_distance={args.min_pair_distance:.3f}")
    if rejected:
        print("closest rejected examples:")
        for idx, basket_dist, pair_dist, reason, name in sorted(rejected, key=lambda row: min(row[1], row[2]))[:10]:
            print(
                f"  index={idx} basket={basket_dist:.3f} pair={pair_dist:.3f} "
                f"reason={reason} video={name}"
            )


if __name__ == "__main__":
    main()
