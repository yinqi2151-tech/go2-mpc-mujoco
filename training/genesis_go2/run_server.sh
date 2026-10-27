#!/usr/bin/env bash
# 在训练服务器上一键运行：训练 v2a、v2b → 导出 → Genesis 参照评测 → （可选）MuJoCo 验收。
#
# 用法（在仓库根目录）：
#   bash training/genesis_go2/run_server.sh                    # 默认用 0 号卡，依次跑 v2a、v2b
#   GPU=3 bash training/genesis_go2/run_server.sh              # 指定卡号（规矩：只用一张卡）
#   VARIANTS="v2b" V2B_ITERS=1500 bash training/genesis_go2/run_server.sh
#
# 预计用时（单张 A100）：v2a 300 轮约 4 分钟，v2b 1000 轮约 15 分钟，每个版本的 Genesis 评测约 1 分钟；
# MuJoCo 验收是 CPU 计算，900 次 × 20 s 仿真，8 个进程约 5 分钟。
# 长时间占卡前记得在群里说一声。
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
VARIANTS="${VARIANTS:-v2a v2b}"
WORKERS="${WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="$GPU"

# conda 环境 genesis（驱动 535 需要 cuda-compat，见进度记录第 2 节）
if [[ "${CONDA_DEFAULT_ENV:-}" != "genesis" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate genesis
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/cuda-compat:${LD_LIBRARY_PATH:-}"

mkdir -p logs results/acceptance
echo "== GPU $GPU，版本：$VARIANTS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPU" || true

for v in $VARIANTS; do
  iters_var="$(echo "${v}_ITERS" | tr '[:lower:]' '[:upper:]')"
  iters="${!iters_var:-0}"
  echo "== [$v] 训练（max-iterations=${iters}，0 表示默认）"
  python training/genesis_go2/train_v2.py --variant "$v" --max-iterations "$iters" 2>&1 | tee "logs/train_${v}.log"

  echo "== [$v] Genesis 参照评测（armature = Genesis 默认 0.1 和 Go2 估计值）"
  mkdir -p "results/acceptance/$v"
  rm -f "results/acceptance/$v/genesis_runs.csv" "results/acceptance/$v/genesis_go2est_runs.csv"
  python scripts/eval_genesis_reference.py --backend gpu --policy-dir "policies/genesis_go2_$v" \
    --levels L2 --frictions 0.4 0.6 1.0 --n 100 --out "results/acceptance/$v/genesis_runs.csv"
  python scripts/eval_genesis_reference.py --backend gpu --policy-dir "policies/genesis_go2_$v" \
    --levels L2 --frictions 0.4 0.6 1.0 --n 100 --armature go2est --config-name genesis_go2est \
    --out "results/acceptance/$v/genesis_go2est_runs.csv"

  echo "== [$v] MuJoCo 验收"
  if ! python -c "import mujoco" 2>/dev/null; then
    pip install -q mujoco matplotlib || echo "装不上 mujoco：把 policies/genesis_go2_$v 拷回笔记本再跑 check_acceptance.py"
  fi
  if python -c "import mujoco" 2>/dev/null; then
    python scripts/check_acceptance.py --name "$v" --policy-dir "policies/genesis_go2_$v" \
      --genesis-runs "results/acceptance/$v/genesis_runs.csv" --workers "$WORKERS" || true
  fi
done

echo "== 完成。需要带回笔记本的："
for v in $VARIANTS; do
  echo "   policies/genesis_go2_$v/  results/acceptance/$v/  logs/train_$v.log"
done
