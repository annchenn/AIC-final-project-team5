"""Evaluate trained Diffusion Policy on MovingCubeGrasp across different cube speeds.

Three speed levels per proposal:
  Slow   0.08 m/s  — slightly below training range (OOD, generalization)
  Medium 0.12 m/s  — mid training range (in-distribution)
  Fast   0.18 m/s  — training upper bound (in-distribution, hardest)

Metrics reported per proposal:
  - Capture rate    : fraction of episodes where cube was lifted above threshold
  - Placement rate  : fraction of captured episodes where cube was placed in basket
  - End-to-end rate : fraction of all episodes with final success (capture + place)

Usage (inside docker container):
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH
  python eval/moving_cube_grasp_eval.py \
      --checkpoint checkpoints/100000/pretrained_model \
      --rounds 10
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLLOUT_SCRIPT = str(Path(__file__).parent.parent / "scripts" / "rollout.py")

# Three speed levels: slow (OOD) / medium (in-dist) / fast (in-dist upper bound)
SPEEDS = [0.08, 0.12, 0.18]
SPEED_LABELS = {0.08: "Slow (0.08)", 0.12: "Medium (0.12)", 0.18: "Fast (0.18)"}

# Proposal targets
TARGET_CAPTURE   = 0.60
TARGET_PLACEMENT = 0.80


@dataclass
class EpisodeResult:
    capture_rate: float
    placement_rate: float   # success | captured; nan if capture_count == 0
    end_to_end_rate: float


def run_single(checkpoint: str, speed: float, rounds: int, action_horizon: int) -> EpisodeResult:
    """Run rollout.py for one speed level and parse the three metrics."""
    import os
    env = os.environ.copy()

    cmd = [
        sys.executable, ROLLOUT_SCRIPT,
        "--task", "HCIS-MovingCubeGrasp-SingleArm-v0",
        "--policy_type", "lerobot-diffusion",
        "--policy_checkpoint_path", checkpoint,
        "--policy_action_horizon", str(action_horizon),
        "--eval_rounds", str(rounds),
        "--episode_length_s", "30",
        "--cube_speed", str(speed),
        "--device", "cuda",
        "--enable_cameras",
    ]

    print(f"\n{'='*60}")
    print(f"[eval] Speed: {speed} m/s  |  Rounds: {rounds}  |  horizon: {action_horizon}")
    print(f"{'='*60}", flush=True)

    result_path = Path(f"eval/results/speed_{speed}.txt")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        cmd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    log = proc.stdout
    result_path.write_text(log)
    print(log)

    def _parse(pattern: str) -> float:
        m = re.search(pattern, log)
        return float(m.group(1)) if m else float("nan")

    capture_rate   = _parse(r"Final capture rate:\s*([\d.]+)")
    placement_rate = _parse(r"Final placement rate:\s*([\d.nan]+)")
    end_to_end     = _parse(r"Final end-to-end rate:\s*([\d.]+)")

    result = EpisodeResult(capture_rate, placement_rate, end_to_end)
    print(
        f"[eval] Speed {speed} m/s -> "
        f"capture={capture_rate:.1%}  placement={placement_rate:.1%}  e2e={end_to_end:.1%}"
    )
    return result


def plot_results(speeds, results, output_path="eval/speed_vs_success.png"):
    captures    = [r.capture_rate   for r in results]
    placements  = [r.placement_rate for r in results]
    end_to_ends = [r.end_to_end_rate for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, captures,    marker="o", label="Capture rate",     linewidth=2)
    ax.plot(speeds, placements,  marker="s", label="Placement rate (given capture)", linewidth=2)
    ax.plot(speeds, end_to_ends, marker="^", label="End-to-end rate",  linewidth=2)

    # Proposal target lines
    ax.axhline(TARGET_CAPTURE,   color="C0", linestyle="--", alpha=0.5,
               label=f"Target capture ≥{TARGET_CAPTURE:.0%}")
    ax.axhline(TARGET_PLACEMENT, color="C1", linestyle="--", alpha=0.5,
               label=f"Target placement ≥{TARGET_PLACEMENT:.0%}")

    ax.set_xlabel("Cube Speed (m/s)")
    ax.set_ylabel("Rate")
    ax.set_title("Moving Cube Grasp — Speed vs. Success Metrics")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xticks(speeds)
    ax.set_xticklabels([SPEED_LABELS.get(s, str(s)) for s in speeds])
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n[eval] Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/100000/pretrained_model",
                        help="Path to pretrained_model directory")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Episodes per speed level")
    parser.add_argument("--speeds", type=float, nargs="+", default=SPEEDS,
                        help="Cube speeds to evaluate (m/s)")
    parser.add_argument("--action_horizon", type=int, default=1,
                        help="policy_action_horizon passed to rollout.py")
    args = parser.parse_args()

    print(f"[eval] Checkpoint:     {args.checkpoint}")
    print(f"[eval] Speeds:         {args.speeds}")
    print(f"[eval] Rounds/speed:   {args.rounds}")
    print(f"[eval] Action horizon: {args.action_horizon}")
    print(f"[eval] Targets — capture ≥{TARGET_CAPTURE:.0%}, placement ≥{TARGET_PLACEMENT:.0%}")

    results = []
    for speed in args.speeds:
        r = run_single(args.checkpoint, speed, args.rounds, args.action_horizon)
        results.append(r)

    print("\n=== Summary ===")
    print(f"{'Speed':>12}  {'Capture':>10}  {'Placement':>12}  {'End-to-end':>12}")
    for s, r in zip(args.speeds, results):
        label = SPEED_LABELS.get(s, f"{s} m/s")
        pr = f"{r.placement_rate:.1%}" if r.placement_rate == r.placement_rate else "n/a"
        print(f"{label:>12}  {r.capture_rate:>10.1%}  {pr:>12}  {r.end_to_end_rate:>12.1%}")

    plot_results(args.speeds, results)


if __name__ == "__main__":
    main()
