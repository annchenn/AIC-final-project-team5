#!/usr/bin/env python
"""Patch LeRobot's EpisodeAwareSampler for local episode-subset training.

LeRobot v3.0's sampler yields absolute frame indices even when
LeRobotDataset(..., episodes=[...]) has loaded a compact subset. In that case the
DataLoader asks the compact dataset for out-of-bounds rows. This patch makes the
sampler emit relative indices for selected episodes while preserving the original
full-dataset behavior when all episodes are used.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import lerobot.datasets.sampler as sampler

PATCH_MARKER = "# patched by cmd_eet/patch_lerobot_sampler.py"

PATCHED_SOURCE = '''#!/usr/bin/env python

from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        {marker}
        When episode_indices_to_use is provided, LeRobotDataset has already
        compacted the selected episodes into a smaller HF dataset. Emit relative
        row indices for that compact dataset instead of absolute full-dataset
        frame indices.
        """
        indices = []
        selected = set(episode_indices_to_use) if episode_indices_to_use is not None else None
        compact_start = 0
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            episode_len = end_index - start_index
            if selected is None:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))
            elif episode_idx in selected:
                compact_end = compact_start + episode_len
                indices.extend(range(compact_start + drop_n_first_frames, compact_end - drop_n_last_frames))
                compact_start = compact_end

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)
'''.format(marker=PATCH_MARKER)


def main() -> None:
    path = Path(inspect.getfile(sampler))
    current = path.read_text()
    if PATCH_MARKER in current:
        print(f"LeRobot sampler already patched: {path}")
        return
    backup = path.with_suffix(path.suffix + ".cmd_eet_backup")
    if not backup.exists():
        backup.write_text(current)
    path.write_text(PATCHED_SOURCE)
    print(f"Patched LeRobot sampler: {path}")
    print(f"Backup: {backup}")


if __name__ == "__main__":
    main()
