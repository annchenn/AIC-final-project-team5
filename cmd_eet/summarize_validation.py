#!/usr/bin/env python
"""Summarize validation JSONL emitted by validate_lerobot_loss.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_validation.py <validation_results.jsonl>")
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"No validation results found at {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No rows in {path}")

    print(f"Results: {path}")
    print("policy\tsplit\tstep\tepisodes\tframes\tloss\tmain_metric\tcheckpoint")
    for row in rows:
        metric = row["metrics"].get("l1_loss", row["loss"])
        print(
            f"{row['policy_type']}\t{row.get('split', '')}\t{row.get('step', '')}\t"
            f"{row['num_episodes']}\t{row['num_frames']}\t"
            f"{row['loss']:.6f}\t{metric:.6f}\t{row['policy_checkpoint']}"
        )


if __name__ == "__main__":
    main()
