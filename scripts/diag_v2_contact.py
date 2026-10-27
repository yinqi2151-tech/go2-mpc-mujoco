"""
v2a / v2b 在 MuJoCo 里速度偏低的原因定位：Go2 估计转子惯量、L2 扰动、每格 30 次 × 20 s，
分别只换接触参数、只换 PD 实现与步长、全部换成 Genesis 风格，看跟踪率怎么变。
用法（仓库根目录）：python scripts/diag_v2_contact.py
"""
import os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, ".."))
os.chdir(os.path.join(_HERE, ".."))
from multiprocessing import Pool
import numpy as np
from eval_sim2sim_repeat import run_one
from mpc_go2.sim_align import GO2_ARMATURE_EST
CFG = {
  "orig":            {},
  "contact_genesis": {"contact": "genesis"},
  "pd_dt10":         {"actuator": "pd", "physics_dt": 0.01},
  "genesis_like":    {"contact": "genesis", "actuator": "pd", "physics_dt": 0.01, "torque_limits": "menagerie"},
}
def task(a):
    name, pol, mu, s = a
    extra = dict(CFG[name], armature=GO2_ARMATURE_EST, duration=20.0, policy_dir=f"policies/{pol}")
    r = run_one(("rl", mu, "L2", s, 1, extra)); r["cfg"] = name; r["pol"] = pol; return r
if __name__ == "__main__":
    tasks = [(c, p, mu, s) for p in ("genesis_go2_v2b", "genesis_go2_v2a") for c in CFG for mu in (0.4, 0.6, 1.0) for s in range(30)]
    with Pool(2) as pool: rows = pool.map(task, tasks)
    for p in ("genesis_go2_v2b", "genesis_go2_v2a"):
        for c in CFG:
            cells = []
            for mu in (0.4, 0.6, 1.0):
                rs = [r for r in rows if r["pol"] == p and r["cfg"] == c and r["friction"] == mu]
                ok = [r["vx_track_ratio"] for r in rs if not r["fell"]]
                cells.append(f"μ{mu}: 摔{sum(r['fell'] for r in rs)}/30 跟踪{np.mean(ok)*100:.1f}%")
            print(p, f"{c:16s}", " | ".join(cells), flush=True)
