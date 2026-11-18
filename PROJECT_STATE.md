# PROJECT_STATE

> 记录时间：2026-11-18
> 本文件是工作区快照，不是自动实时数据。恢复项目时请先执行 `git status` 和 `git rev-parse HEAD`。

## 1. 项目目标与软硬件环境

目标是构建 Scout Mini + Piper6 移动操作机器人的分区导航与移动抓取任务框架，并保留将高层 Skill 选择器替换为 JEV 类轻量概率模型的能力。

当前仓库只包含 ROS 2 软件原型：

- `robot_interfaces`：自定义 Action，目前只有 `NavigateToRegion.action`。
- `robot_nav`：区域导航 Skill Server、Nav2 Action 转发、区域配置、Mock Nav2 Server。
- `robot_task`：任务类型、Selector、Registry、DecisionEngine、Mock Executor、同步 TaskManager、任务级 BT 原型。

外部目录 `/home/jetson/scout_ws`、`/home/jetson/fast_lio_ws`、`/home/jetson/ws_livox` 存在，但不在本仓库内，未审计版本，**不得视为已经集成**。

## 2. 当前架构边界

目标数据流：

```text
自然语言（Grounding 未实现）
  -> TaskGoal
  -> 独立决策层：Selector -> Registry -> DecisionEngine
  -> SkillCandidate(skill_id, arguments, score)
  -> 任务级 BT / 调度器（尚未连接）
  -> SkillExecutor / Adapter（尚无 ROS 实现）
  -> ROS 2 Skill Action Server
  -> 可信结果/观测
  -> RobotState 更新
  -> 下一次决策
```

当前仍是两条独立链路：

1. **纯 Python Mock 链**：`TaskManager.run()` 同步循环；`MockSkillExecutor` 直接返回成功和新 `RobotState`。
2. **ROS 2 导航原型**：`task_bt.py` 静态 Sequence 只执行一次导航，然后进入占位 Success；没有 DecisionNode、Blackboard 或 `RobotState` 回写。

职责约定：

- 决策层不依赖 BT，不执行 ROS Action。
- BT 负责外层生命周期，不写死五个 Skill 的固定顺序。
- Executor/Adapter 只执行单个 Skill，Action accepted/feedback 不等于成功。
- 只有终态成功才提交已确认的 `RobotState`。

## 3. 里程碑

| 任务 | 内容 | 当前状态 | 证据 |
| --- | --- | --- | --- |
| Task 001 | MockSkillExecutor 四类正向测试 | 已提交，回归测试通过 | Commit `1b51e6e` |
| Task 002 | SkillCandidate 参数契约前移到 Registry | 已提交，回归测试通过 | Commit `35f4aa8` |
| Task 003 | 同步 Mock TaskManager 闭环 | 已提交，回归测试通过 | Commit `1f044c5` |
| Task 004 | BT / 决策层 / ROS 2 Action 只读审计 | 只读审计完成，未运行测试 | 审计报告；未改代码 |
| Task 005 | 纯 Python 非阻塞 TaskRunner + 测试 + 本文档 | 已实现，测试通过，待人工审阅；未提交 | 待审阅 Diff |

Task 005 不修改 Task 003 的同步 Mock 基线，也不修改任何 ROS/BT/导航代码。

## 4. 当前工作树快照

- 仓库：`/home/jetson/mobile_robot_ws`
- 分支：`main`
- Remote：`git@github.com:Amooomo/scout-piper6-mobile-robot.git`
- 创建本文档时 HEAD：`1f044c52b71f9106a936bdd6d32537377e8884db`
- 说明：Task 005 开始时 HEAD 为 `35f4aa8`，执行期间用户提交了 Task 003，因此当前 HEAD 为 `1f044c5`。以最终审阅时的实际 Git 输出为准。

创建本文档时的未跟踪项：

```text
?? .marscode/
?? PROJECT_STATE.md
?? src/robot_task/robot_task/task_runner.py
?? src/robot_task/robot_task/test_task_runner.py
```

- 未执行 `git add`、Commit 或 Push。
- `.marscode/` 未触碰。
- 新增文件尚未提交，普通 `git diff` 不会显示它们。

## 5. 验证证据

执行目录：`/home/jetson/mobile_robot_ws/src/robot_task`

| 命令 | 退出码 | PASS 数 |
| --- | --- | --- |
| `python3 -m robot_task.test_task_runner` | 0 | 12 |
| `python3 -m robot_task.test_skill_loop` | 0 | 7 |
| `python3 -m robot_task.test_skill_registry` | 0 | 22 |
| `python3 -m robot_task.test_skill_selector` | 0 | 9 |
| `python3 -m robot_task.test_mock_skill_executor` | 0 | 5 |

`git diff --check` 退出码 0，无输出。

Task 004 为只读审计，没有运行测试。以上只证明纯 Python 层测试通过，不代表真实 ROS 2、Nav2、Scout 或 Piper6 已验证。

## 6. 已确定的架构决策

1. 决策层保持独立、可离线测试，Selector/Registry/DecisionEngine 不进入 BT 节点内部。
2. 任务级 BT 固定“决策 -> 派发 -> 等待 -> 提交状态 -> 再决策”的生命周期，不固定具体 Skill 顺序。
3. SkillExecutor/Adapter 负责单个 Skill 的参数映射和生命周期，不重新评分。
4. Action Goal accepted、feedback 或 BT 叶节点仍 RUNNING 都不能当作任务成功。
5. 只有终态成功结果才能提交 `RobotState` 并计入 `completed_skills`。
6. Task 003 的同步 `TaskManager.run()` 保留为 Mock 回归基线，不塞进 BT tick。
7. Task 005 只实现非阻塞 `start/update/cancel` 契约，不实现并发、超时或真实 ROS 回调。

## 7. 关键未解决问题

- `RobotState` 的真实权威更新者尚未定义；当前只有 Mock Executor 写状态。
- `TaskStatus` 没有 `CANCELED`；Task 005 暂将取消结束映射为 FAILED，并记录 cancel 信息。
- 取消确认、永久 pending、部分物理执行后的状态重同步仍未解决。
- 旧 Goal 的异步结果可能污染新执行；需要 Goal 身份/执行序号隔离。
- 尚无执行超时、重试或重复候选抑制机制。
- `VISUAL_DOCK`、`DETECT_OBJECT`、`MANIPULATE_PICK`、`VERIFY_RESULT` 没有 ROS 2 Action/Service。
- `NAVIGATE_REGION` 默认下游为 `/mock_navigate_to_pose`；真实 Nav2 launch/参数未配置。
- 自然语言 Goal Grounding 未实现。
- JEV 分数校准、推理延迟和 Jetson 资源目标尚未确定。
- BT 顶层动态 DecisionNode、Blackboard、Skill Adapter 尚未实现。

## 8. 下一任务建议

优先在 Task 005 审阅通过后讨论：

1. BT 与 `TaskRunner` 的薄桥接层：BT tick 调用 `update()`，不在 tick 内等待。
2. `NAVIGATE_REGION` 的 ROS 2 异步 Adapter：用 Fake Action/Future 先验证 accepted、feedback、result、cancel。
3. 再讨论其余 Skill 的最小 Action/Service 接口。

具体任务编号和范围须由用户与 ChatGPT 确认，不要直接开始实现。

## 9. 协作规则

- ChatGPT：设计、教学、审阅、拆分窄任务。
- OpenCode：只执行用户授权的 `.md` 窄任务，不擅自扩大范围。
- 用户：阅读代码与 Diff、验收测试、决定 Git Commit/Push。
- 未经明确授权，不自动提交、推送、运行真实机器人或发送 Action Goal。

## 10. 新 Chat 恢复流程

1. 执行 `git rev-parse HEAD` 和 `git status --short`。
2. 阅读本文件。
3. 阅读 Task 004 审计报告与 Task 005 最终报告。
4. 阅读当前代码，重点核对 `task_manager.py`、`task_runner.py`、`test_task_runner.py`、`task_bt.py`。
5. 确认 GitHub 上的 Commit 是否已包含本地新增文件；不要假设本文档中的 SHA 等于最终推送 Commit。
