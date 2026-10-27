"""Reconstruct PushT initial geometry from the exact evaluation seeds.

Run in the WSL LeRobot environment with gym_pusht installed. Bins describe
geometry, not intrinsic task difficulty, and are fixed before reading rewards.
"""

import argparse
import csv
import math
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  # registers gym_pusht/PushT-v0


def angle_difference(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def distance_bin(distance: float) -> str:
    if distance < 100:
        return "near_lt100"
    if distance < 200:
        return "medium_100to199"
    return "far_ge200"


def angle_bin(radians: float) -> str:
    degrees = math.degrees(radians)
    if degrees < 30:
        return "aligned_lt30deg"
    if degrees < 90:
        return "moderate_30to89deg"
    return "opposed_ge90deg"


def derive(start_seed: int, n_episodes: int) -> list[dict]:
    env = gym.make("gym_pusht/PushT-v0")
    rows = []
    try:
        for seed in range(start_seed, start_seed + n_episodes):
            _, info = env.reset(seed=seed)
            block = info["block_pose"]
            goal = info["goal_pose"]
            agent = info["pos_agent"]
            block_goal_distance = math.dist(block[:2], goal[:2])
            agent_block_distance = math.dist(agent, block[:2])
            orientation_error = angle_difference(float(block[2]), float(goal[2]))
            rows.append({
                "seed": seed,
                "agent_x": float(agent[0]),
                "agent_y": float(agent[1]),
                "block_x": float(block[0]),
                "block_y": float(block[1]),
                "block_angle_rad": float(block[2]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "goal_angle_rad": float(goal[2]),
                "block_goal_distance_px": round(block_goal_distance, 6),
                "orientation_error_deg": round(math.degrees(orientation_error), 6),
                "agent_block_distance_px": round(agent_block_distance, 6),
                "distance_bin": distance_bin(block_goal_distance),
                "angle_bin": angle_bin(orientation_error),
                "agent_distance_bin": distance_bin(agent_block_distance),
            })
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.n_episodes <= 0:
        parser.error("--n-episodes must be positive")
    rows = derive(args.start_seed, args.n_episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} initial scenes -> {args.output}")


if __name__ == "__main__":
    main()
