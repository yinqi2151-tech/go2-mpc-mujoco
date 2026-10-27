"""
把 Genesis + rsl_rl 训练出的检查点导出成只依赖 numpy 的格式，供 MuJoCo 里运行。

输入：logs/<实验名>/model_<轮数>.pt 和 cfgs.pkl（Genesis go2_train.py 保存的）
输出：policy.npz（actor 网络权重）+ policy_cfg.json（关节顺序、默认角、观测缩放、PD 增益等）

需要 torch（在训练服务器上或任何装了 torch 的环境里运行一次即可）：
    python scripts/export_rsl_policy.py --ckpt policies/genesis_go2_walking/model_100.pt \
        --cfgs policies/genesis_go2_walking/cfgs.pkl --out policies/genesis_go2_walking
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfgs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    if any("normalizer" in k for k in sd):
        raise RuntimeError("检查点里有观测归一化参数，本导出脚本还没处理这种情况")

    # mlp.0 / mlp.2 / mlp.4 / mlp.6 是线性层，中间的奇数下标是激活函数
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    arrays = {"n_layers": np.array(len(lin))}
    for i, idx in enumerate(lin):
        arrays[f"w{i}"] = sd[f"mlp.{idx}.weight"].numpy().astype(np.float32)
        arrays[f"b{i}"] = sd[f"mlp.{idx}.bias"].numpy().astype(np.float32)

    with open(a.cfgs, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    activation = train_cfg.get("actor", train_cfg.get("policy", {})).get("activation", "elu")
    if activation != "elu":
        raise RuntimeError(f"激活函数是 {activation}，运行时目前只实现了 elu")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "policy.npz", **arrays)
    meta = dict(env_cfg=env_cfg, obs_cfg=obs_cfg, command_cfg=command_cfg, reward_cfg=reward_cfg,
                actor=train_cfg.get("actor"), iteration=int(ck.get("iter", -1)),
                source_checkpoint=Path(a.ckpt).name)
    (out / "policy_cfg.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    shapes = [tuple(arrays[f"w{i}"].shape) for i in range(len(lin))]
    print(f"导出完成：{out / 'policy.npz'}  层形状 {shapes}  训练轮数 {meta['iteration']}")


if __name__ == "__main__":
    main()
