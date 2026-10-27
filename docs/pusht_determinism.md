# PushT 评测重复性修复

LeRobot 0.6.1 的 `lerobot-eval` 会启用 `torch.backends.cudnn.benchmark=True`。在本机 RTX 4060 Laptop GPU 上，两次原生评测即使使用相同策略、参数和 seed 3000，回合最高奖励仍分别为 0.614、0.392。固定种子可复现 PushT 的初始状态抽样，但没有保证完整 GPU 推理轨迹一致。

`scripts/eval_pusht_deterministic.py` 在运行 LeRobot 原评测入口时启用 `torch.use_deterministic_algorithms(True)`、`torch.backends.cudnn.deterministic=True`，并在 LeRobot 创建环境时关掉 cuDNN benchmark；同时设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。策略加载、处理器、仿真环境、rollout、视频和 `eval_info.json` 均仍由 LeRobot 0.6.1 完成。

验证：相同 seed 3000 的两次确定性原生评测及一次确定性零噪声包装评测，回合最高奖励均为 **0.990895103392703**。这证明本机、本模型、该种子的重复运行一致；不代表所有硬件与依赖版本均确定。

确定性结果与原先非确定性结果属于不同实验协议，报告必须分开呈现。之后的动作扰动实验先用相同的确定性设置分别运行 `--action-noise-std=0` 和非零强度，再比较同种子结果。确认零噪声包装与干净评测一致后，才能将扰动差异归因于加入的噪声。
