"""Run LeRobot 0.6.1 evaluation with deterministic PyTorch settings.

This is an opt-in reproducibility probe. It preserves LeRobot's policy,
environment, processors, rollout and result format.
"""

import os

# CUDA requires this before torch initializes cuBLAS for deterministic matmul.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
from lerobot.scripts import lerobot_eval  # noqa: E402


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    original_make_env = lerobot_eval.make_env

    def make_env_with_determinism(*args, **kwargs):
        # LeRobot eval_main enables cudnn.benchmark before calling make_env.
        torch.backends.cudnn.benchmark = False
        return original_make_env(*args, **kwargs)

    lerobot_eval.make_env = make_env_with_determinism
    try:
        lerobot_eval.eval_main()
    finally:
        lerobot_eval.make_env = original_make_env


if __name__ == "__main__":
    main()
