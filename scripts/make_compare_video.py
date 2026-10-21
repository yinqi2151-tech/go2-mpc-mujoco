"""
生成“修正前 vs 修正后”的左右对比视频，以及对应的曲线图。

输出：
    results/video_original.mp4   原始 MPC 参考
    results/video_fix.mp4        位置参考修正后
    results/compare.mp4          左右拼接的对比视频（面试 / 汇报用这个）
    results/plot_original.png、results/plot_fix.png   每次运行的速度 / 高度 / 俯仰曲线

用法（需要 pip install imageio imageio-ffmpeg pillow）：
    python scripts/make_compare_video.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_mpc_walk_mujoco import RunParams, run, plot_run, open_video  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def main():
    os.makedirs(R, exist_ok=True)
    runs = [("original", False, "Original MPC"), ("fix", True, "With x-ref fix")]
    for name, fix, label in runs:
        print(f"正在运行并录制：{label} ……（约 1 分钟）", flush=True)
        p = RunParams(vx=0.5, duty=0.65, duration=12.0, mpc_every=4, fix_xref=fix,
                      video=os.path.join(R, f"video_{name}.mp4"), video_label=label)
        log, s = run(p)
        plot_run(log, s, os.path.join(R, f"plot_{name}.png"))
        print(f"  速度跟踪 {s['vx_track_ratio'] * 100:.1f}%，前进 {s['distance_m']:.2f} m，摔倒：{s['fell']}")

    import imageio.v2 as imageio
    a = imageio.get_reader(os.path.join(R, "video_original.mp4"))
    b = imageio.get_reader(os.path.join(R, "video_fix.mp4"))
    out = open_video(os.path.join(R, "compare.mp4"))
    for fa, fb in zip(a, b):
        gap = np.full((fa.shape[0], 8, 3), 255, dtype=np.uint8)
        out.append_data(np.hstack([fa, gap, fb]))
    out.close()
    print("对比视频已保存：results/compare.mp4")


if __name__ == "__main__":
    main()
