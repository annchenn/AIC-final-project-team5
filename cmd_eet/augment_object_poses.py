#!/usr/bin/env python3
"""Augment UMI object_poses.json with safe xy jitter.

The task config uses fixed yaw, so this script jitters block tvec x/y and
keeps rvec unchanged. It rejects samples too close to the basket, too close to
each other, or outside a conservative workspace box.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

BLOCK_NAMES = {"green_block", "blue_block", "red_block"}
ANCHOR_XY = (0.35, 0.0)
BASKET_XY = (0.65, -0.55)


def world_xy_from_tvec(tvec):
    return ANCHOR_XY[0] + float(tvec[0]), ANCHOR_XY[1] + float(tvec[1])


def valid(entry, *, min_basket, min_pair, x_bounds, y_bounds):
    pts = []
    for obj in entry.get("objects", []):
        if obj.get("object_name") not in BLOCK_NAMES:
            continue
        x, y = world_xy_from_tvec(obj["tvec"])
        if not (x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]):
            return False
        if math.hypot(x - BASKET_XY[0], y - BASKET_XY[1]) < min_basket:
            return False
        pts.append((x, y))
    if len(pts) != len(BLOCK_NAMES):
        return False
    for i, a in enumerate(pts):
        for b in pts[i + 1:]:
            if math.hypot(a[0] - b[0], a[1] - b[1]) < min_pair:
                return False
    return True


def jitter_entry(entry, rng, jitter_xy, copy_index):
    out = json.loads(json.dumps(entry))
    out["video_name"] = f"aug{copy_index:04d}_{out.get('video_name', 'pose')}"
    for obj in out.get("objects", []):
        if obj.get("object_name") in BLOCK_NAMES:
            obj["tvec"][0] = float(obj["tvec"][0]) + rng.uniform(-jitter_xy, jitter_xy)
            obj["tvec"][1] = float(obj["tvec"][1]) + rng.uniform(-jitter_xy, jitter_xy)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/object_poses_clear.json")
    parser.add_argument("--output", default="data/object_poses_clear_aug.json")
    parser.add_argument("--copies", type=int, default=5, help="Augmented copies per source full episode.")
    parser.add_argument("--jitter-xy", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-basket-distance", type=float, default=0.22)
    parser.add_argument("--min-pair-distance", type=float, default=0.12)
    parser.add_argument("--x-bounds", type=float, nargs=2, default=(0.05, 0.58))
    parser.add_argument("--y-bounds", type=float, nargs=2, default=(-0.62, -0.10))
    parser.add_argument("--drop-original", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = Path(args.input)
    data = json.loads(src.read_text())
    full = [e for e in data if isinstance(e, dict) and e.get("status") == "full"]
    if not full:
        raise SystemExit(f"No full episodes in {src}")

    out = [] if args.drop_original else [json.loads(json.dumps(e)) for e in full]
    rejected = 0
    made = 0
    for entry in full:
        for _ in range(args.copies):
            accepted = None
            for _attempt in range(100):
                candidate = jitter_entry(entry, rng, args.jitter_xy, made)
                if valid(
                    candidate,
                    min_basket=args.min_basket_distance,
                    min_pair=args.min_pair_distance,
                    x_bounds=tuple(args.x_bounds),
                    y_bounds=tuple(args.y_bounds),
                ):
                    accepted = candidate
                    break
            if accepted is None:
                rejected += 1
                continue
            out.append(accepted)
            made += 1

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"source_full={len(full)}")
    print(f"augmented_new={made}")
    print(f"rejected_attempt_groups={rejected}")
    print(f"output_full={len(out)}")
    print(f"output={dst}")


if __name__ == "__main__":
    main()
