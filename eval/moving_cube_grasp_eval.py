"""Evaluate trained Diffusion Policy on MovingCubeGrasp across different cube speeds.

Usage (inside docker container):
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/torch/lib:$LD_LIBRARY_PATH
  python eval/moving_cube_grasp_eval.py \
      --checkpoint checkpoints/moving_cube_run1/checkpoints/100000/pretrained_model \
      --rounds 10
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLLOUT_SCRIPT = str(Path(__file__).parent.parent / "scripts" / "rollout.py")
SPEEDS = [0.05, 0.08, 0.10, 0.12]


def run_single(checkpoint: str, speed: float, rounds: int) -> float:
    """Run rollout for one speed and return success rate."""
    import os
    env = os.environ.copy()

    cmd = [
        sys.executable, ROLLOUT_SCRIPT,
        "--task", "HCIS-MovingCubeGrasp-SingleArm-v0",
        "--policy_type", "lerobot-diffusion",
        "--policy_checkpoint_path", checkpoint,
        "--eval_rounds", str(rounds),
        "--episode_length_s", "30",
        "--cube_speed", str(speed),
        "--device", "cuda",
        "--enable_cameras",
    ]

    print(f"\n{'='*60}")
    print(f"[eval] Speed: {speed} m/s  |  Rounds: {rounds}")
    print(f"{'='*60}", flush=True)

    result_path = Path(f"eval/results/speed_{speed}.txt")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    with open(result_path, "w") as log_file:
        proc = subprocess.run(
            cmd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        log_file.write(proc.stdout)
        print(proc.stdout)

    m = re.search(r"Final success rate:\s*([\d.]+)", result_path.read_text())
    rate = float(m.group(1)) if m else 0.0
    print(f"[eval] Speed {speed} m/s -> success rate: {rate:.3f}")
    return rate


def plot_results(speeds, rates, output_path="eval/speed_vs_success.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(speeds, rates, marker="o", linewidth=2, markersize=8)
    for s, r in zip(speeds, rates):
        plt.annotate(f"{r:.0%}", (s, r), textcoords="offset points", xytext=(0, 10), ha="center")
    plt.xlabel("Cube Speed (m/s)")
    plt.ylabel("Success Rate")
    plt.title("Moving Cube Grasp ¡X Speed vs Success Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n[eval] Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/moving_cube_run1/checkpoints/100000/pretrained_model")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--speeds", type=float, nargs="+", default=SPEEDS)
    args = parser.parse_args()

    print(f"[eval] Checkpoint: {args.checkpoint}")
    print(f"[eval] Speeds: {args.speeds}")
    print(f"[eval] Rounds per speed: {args.rounds}")

    rates = []
    for speed in args.speeds:
        rate = run_single(args.checkpoint, speed, args.rounds)
        rates.append(rate)

    print("\n=== Summary ===")
    for s, r in zip(args.speeds, rates):
        print(f"  Speed {s} m/s: {r:.1%}")

    plot_results(args.speeds, rates)


if __name__ == "__main__":
    main()
