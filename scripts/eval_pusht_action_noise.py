"""LeRobot 0.6.1 PushT evaluation with seeded Gaussian action perturbations.

The original LeRobot evaluator remains responsible for policy loading, rollout,
metrics and videos. This script only wraps the environment's step() call. Noise
is applied after policy postprocessing, in PushT action coordinates.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from lerobot.scripts import lerobot_eval


class ActionNoiseWrapper(gym.vector.VectorWrapper):
    def __init__(self, env: gym.vector.VectorEnv, std: float):
        super().__init__(env)
        if env.num_envs != 1:
            raise ValueError("action-noise protocol requires --eval.batch_size=1")
        self.std = std
        self._rng = None

    def reset(self, *, seed=None, options=None):
        if not isinstance(seed, (list, tuple)) or len(seed) != 1:
            raise ValueError("each rollout must provide one explicit episode seed")
        # Independent generator avoids changing the environment or policy RNG.
        self._rng = np.random.default_rng(int(seed[0]) + 100_000)
        return self.env.reset(seed=seed, options=options)

    def call(self, name, *args, **kwargs):
        return self.env.call(name, *args, **kwargs)

    def step(self, actions):
        if self._rng is None:
            raise RuntimeError("reset must be called before step")
        if self.std == 0:
            return self.env.step(actions)
        perturbed = np.asarray(actions) + self._rng.normal(0, self.std, np.asarray(actions).shape)
        perturbed = np.clip(perturbed, self.action_space.low, self.action_space.high)
        return self.env.step(perturbed.astype(np.asarray(actions).dtype, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--action-noise-std", type=float, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args, lerobot_args = parser.parse_known_args()
    if not math.isfinite(args.action_noise_std) or args.action_noise_std < 0:
        parser.error("--action-noise-std must be finite and nonnegative")
    if "--eval.batch_size=1" not in lerobot_args or "--eval.use_async_envs=false" not in lerobot_args:
        parser.error("use --eval.batch_size=1 --eval.use_async_envs=false")
    output_args = [arg.split("=", 1)[1] for arg in lerobot_args if arg.startswith("--output_dir=")]
    if len(output_args) != 1:
        parser.error("pass exactly one --output_dir=...")
    output_dir = Path(output_args[0])

    original_make_env = lerobot_eval.make_env
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    def make_perturbed_env(*env_args, **env_kwargs):
        if args.deterministic:
            # LeRobot eval_main enables benchmark before invoking make_env.
            torch.backends.cudnn.benchmark = False
        envs = original_make_env(*env_args, **env_kwargs)
        return {
            group: {task_id: ActionNoiseWrapper(env, args.action_noise_std) for task_id, env in tasks.items()}
            for group, tasks in envs.items()
        }

    lerobot_eval.make_env = make_perturbed_env
    sys.argv = [sys.argv[0], *lerobot_args]
    try:
        lerobot_eval.eval_main()
    finally:
        lerobot_eval.make_env = original_make_env
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "perturbation.json").write_text(json.dumps({
        "type": "gaussian_action_noise",
        "std": args.action_noise_std,
        "applied_at": "after_policy_postprocessing_before_env_step",
        "noise_seed_rule": "episode_seed + 100000",
        "batch_size": 1,
        "le_robot_version": "0.6.1",
        "deterministic_torch": args.deterministic,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
