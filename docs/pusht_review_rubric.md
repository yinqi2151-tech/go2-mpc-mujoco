# PushT 视频复核规则

## 单位与覆盖率

一行对应一次策略运行的一个 episode。先记录 `policy`、`seed`、`episode_ix`、视频路径和是否实际看过视频。LeRobot 当前只保存前 10 个回合的视频；未保存或未观看的回合记为 `not_reviewed`，不能推断失败原因。报告中同时写明“已复核数 / 失败总数”。

## 初始场景标签

在视频 **第一帧** 判断，不使用结果倒推难度：

| 字段 | 可选值 | 判定方式 |
| --- | --- | --- |
| `initial_distance` | `near` / `medium` / `far` / `uncertain` | 可移动 T 与目标 T 的初始中心距离，按画面宽度粗分：< 1/4、1/4–1/2、> 1/2；边界不清填 uncertain |
| `initial_orientation` | `aligned` / `misaligned` / `uncertain` | 初始两个 T 的朝向是否大致平行；不精确测角 |
| `pusher_position` | `near_object` / `far_object` / `uncertain` | 推杆初始位置是否贴近可移动 T |

这些是人工粗标签。需要定量分层时，应改为从环境状态提取距离和角度，并预先固定阈值。

## 过程与失败原因

每个已复核视频只填一个**主要失败原因**，其余现象放在 `observation`；成功回合填 `success`：

| 标签 | 可观察判据 |
| --- | --- |
| `no_contact` | 推杆全程没有有效接触可移动 T |
| `wrong_contact` | 明显接触目标标记或从不利侧推动，且未有效移动可移动 T |
| `overshoot` | 可移动 T 一度接近目标，随后被推离 |
| `insufficient_alignment` | 已有效接触并移动，但位置或朝向始终未达到成功条件 |
| `timeout_after_progress` | 一度明显接近，之后无效动作或重复动作耗尽时间；无法归入上述更具体类别 |
| `other_observed` | 可观察的失败过程不属于以上类别，须写明具体行为 |
| `uncertain` | 视频不足以作出可靠判断 |
| `not_reviewed` | 未观看或没有视频；不能当作一种失败模式 |

先看是否成功和奖励曲线，再看视频。奖励只是辅助定位关键时刻，不能直接生成失败标签。`max_reward` 是全程峰值，不能代表最后一帧。记录具体动作、时刻与可见现象，推测原因单独写入 `hypothesis`，不要把推测写成结论。

## 复核顺序

用 `python scripts/triage_policy_failures.py` 生成待复核清单。优先看同种子两策略奖励差大的回合和新增/丢失的成功回合；之后按种子顺序补全其他已保存视频。每次改动标签时保留人工原始观察。若两人复核意见不一致，记录两个原始判断并复议，避免覆盖。
