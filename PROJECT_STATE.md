# PROJECT_STATE

> 记录时间：2026-11-24
> 本文件是人工维护的项目快照，不是自动实时数据。恢复项目时先执行 `git status`、`git rev-parse HEAD`，再以本文件为准核对方向与边界。

## 1. 项目方向（已锁定）

方向：面向 Jetson 端侧部署的**可行性感知移动操作机器人**（Feasibility-Aware Mobile Manipulation）。

核心链路：分区导航 -> 停靠/对位 -> 单物体抓取 -> 结果核验。高层决策以确定性的门控路由为主，不再以 SayCan / JEV 叙事作为当前实现目标。

执行边界：

1. **锁定单一操作后端**。第 1–2 周优先离线排障并验证已有 ACT 后端；设置**第 10 天硬止损门**：若 ACT 仍无法稳定抓取且 Leader Arm 借用窗口落空，立即切换至“单物体视觉定位 + MoveIt2/IK”。
2. **不并行维护双操作后端**。同一时间只维护一条操作链路，避免调试资源与数据采集被摊薄。
3. **门控路由对象**：在单一锁定后端下，基于停靠位姿偏差 $(\Delta x, \Delta y, \Delta \theta)$ 与可行性特征，动态决策：
   - `DIRECT_EXEC`：偏差可接受，直接执行操作；
   - `VISUAL_DOCK`：适配 Scout 差速底盘的横向偏差，执行两段式视觉对位；
   - `SAFE_STOP`：超出可恢复范围，停止并交还人工。
4. **硬件与安全约束**：
   - 所有真机连接与上电操作必须先经过独立安全审查；
   - 关节位置越界必须拒绝并中止（`JOINT_LIMIT_ABORT`），**禁止静默位置裁剪**；
   - 明确区分“速度斜率限幅”与“位置越界拒绝”，两者不可互相替代。

## 2. 当前 Git 基线

- 仓库：`/home/jetson/mobile_robot_ws`
- 分支：`main`，已与 `origin/main` 同步，工作树干净。
- 基线 Commit：`8ed1239253556426087de9b213c4057b29fb90d7`
  - Message：`feat: add bt skill executor, navigate adapter, test files and deviceInfo config`

历史里程碑：

| 阶段 | 内容 | 证据 |
| --- | --- | --- |
| Task 001 | `MockSkillExecutor` 四类正向执行 | `1b51e6e` |
| Task 002 | `SkillRegistry` 参数契约前移与校验 | `35f4aa8` |
| Task 003 | 同步 `TaskManager` Mock 闭环 | `1f044c5` |
| Task 004 | BT / 决策层 / ROS 2 Action 只读审计 | 只读审计报告；未改代码 |
| Task 005 | 非阻塞 `TaskRunner`（`start/update/cancel`） | 提交至 `ffe0d5e` |
| Task 006B | `bt_skill_executor.py` | `8ed1239`，12 tests |
| Task 006C | `navigate_region_action_adapter.py`（含立即 Future 修复） | `8ed1239`，17 tests |
| Task 006D | `navigate_bt_execution.py` | `8ed1239`，15 tests |
| Task 006E | `ros_mock_navigation_runner.py` | `8ed1239`，6 tests；Domain 123 隔离实测通过 |

需保留的基线：

- 保留现有 `TaskRunner` 调度架构（`start/update/cancel` 非阻塞契约），不退回同步循环。
- 保留 **62 项 Mock 回归测试** = `TaskRunner` 12 项 + Task 006B–006E 50 项（12 + 17 + 15 + 6）。
- 已审阅的 006B–006E 纯 Mock 补丁（锁内 I/O、`execution_id` 隔离、单叶 BT 1-tick 延迟等）**暂时冻结**。待真机操作后端与 Visual Dock 跑通后，在全链集成阶段面向真实 Action Server 一次性补齐；不再继续扩展纯 Mock 的单叶 BT 壳子。

当前仓库边界：

- `robot_interfaces`：自定义 Action，目前只有 `NavigateToRegion.action`。
- `robot_nav`：区域导航 Skill Server、Nav2 Action 转发、区域配置、Mock Nav2 Server。
- `robot_task`：任务类型、Selector、Registry、DecisionEngine、Mock Executor、`TaskRunner`、任务级 BT 原型与导航 Mock 适配器。

外部目录 `/home/jetson/scout_ws`、`/home/jetson/fast_lio_ws`、`/home/jetson/ws_livox` 不在本仓库内，未审计版本，**不得视为已经集成**。

## 3. 当前最高优先级阻塞项

Piper6 + ACT 真机抓取成功率为 **0%**。已知现象：

- J2/J3 存在超限与负值；
- 存在零位边界钉死（关节贴近零位后被限制在边界附近，无法正常越过）。

因此下一任务为 **Phase 0：纯离线 ACT 数据集与 checkpoint 诊断**，不连接硬件：

1. 数据集一致性：`meta/info.json`、`meta/stats.json` 与 Parquet/视频帧，统计 episode/帧数、相机键名与分辨率、`observation.state`/`action` 极值、每 episode 第 0 帧关节值，以及 J2/J3 边界钉死与越零帧占比。
2. Checkpoint 契约：`pretrained_model/config.json` 的 `input_features`/`output_features` 必须与数据集在维度、相机键名、HWC/CHW 形状上 100% 匹配，直接暴露 “2 相机 vs 3 相机” 或特征错配。
3. 开环推理评测：加载 `ACTPolicy` 与 `make_pre_post_processors`，复用部署脚本一致的 LeRobot v0.4.4 `predict_action` 调用链，逐帧对比预测动作与数据集真实 `action`。
4. 视觉样本导出：抽取起始帧与中段帧生成带键名标注的 PNG 网格，人工确认 RGB 通道、视角、目标物体正常。

硬约束：诊断阶段禁止打开 CAN、相机或连接 Piper6 真机；数据集与 checkpoint 目录严格只读。

## 4. 当前已知待清理项

1. `.marscode/deviceInfo.json` 误入版本库（随 `8ed1239` 提交），需后续单独清理。
2. `src/robot_task/` 内残留嵌套的 `build/`、`install/`、`log/` 目录，需后续确认后清理。
3. 上述清理不得与功能补丁混在同一次提交中；清理前须确认没有依赖这些路径的脚本。

## 5. 已确定的架构决策

1. 决策层保持独立、可离线测试；Selector / Registry / DecisionEngine 不进入 BT 节点内部。
2. 任务生命周期固定为“决策 -> 派发 -> 等待 -> 提交状态 -> 再决策”，由 `TaskRunner` 直接调度异步 Action Handle，冻结单叶 BT 封装；不固定具体 Skill 顺序。
3. `SkillExecutor` / Adapter 只负责单个 Skill 的参数映射与生命周期，不重新评分。
4. Action Goal accepted、feedback 或 BT 叶节点仍 RUNNING 都不能当作任务成功。
5. 只有终态成功结果才能提交 `RobotState` 并计入 `completed_skills`。
6. 同步 `TaskManager.run()` 保留为 Mock 回归基线，不塞进 BT tick。
7. `TaskRunner` 只实现非阻塞 `start/update/cancel` 契约；并发、超时与真实 ROS 回调留待全链集成。
8. 旧 Goal 的异步结果必须有 `execution_id` 身份隔离，避免污染新执行。
9. 位置越界一律拒绝并中止，速度限幅只用于平滑，不作为越界保护。

## 6. 关键未解决问题

- `RobotState` 的真实权威更新者尚未定义；当前只有 Mock Executor 写状态。
- `TaskStatus` 没有 `CANCELED`；取消结束暂映射为 FAILED，并记录 cancel 信息。
- 取消确认、永久 pending、部分物理执行后的状态重同步仍未解决。
- 尚无执行超时、重试或重复候选抑制机制。
- `VISUAL_DOCK`、`DETECT_OBJECT`、`MANIPULATE_PICK`、`VERIFY_RESULT` 没有真实 ROS 2 Action/Service。
- `NAVIGATE_REGION` 默认下游为 `/mock_navigate_to_pose`；真实 Nav2 launch/参数未配置。
- 自然语言 Goal Grounding 未实现。
- ACT 在真机上的 J2/J3 越界与零位钉死根因尚未定位（Phase 0 诊断目标）。
- 单叶 BT 封装已冻结；真实异步 Action Handle 的 accepted/feedback/result/cancel 生命周期尚未在真实 Action Server 上验证，后续由 `TaskRunner` 直接调度。

## 7. 下一阶段顺序

1. **Phase 0（立即执行）**：纯离线 ACT 数据集与 checkpoint 诊断，产出可复核报告与视觉样本。
2. 依据诊断结果修正数据 / 归一化 / checkpoint 契约，并在离线开环评测中确认 J2/J3 不再越界或钉死。
3. 真机操作后端与 Visual Dock 跑通后，进入全链集成，一次性面向真实 Action Server 补齐 006B–006E 冻结补丁。
4. 第 10 天评估 ACT 是否达到稳定抓取；未达到则执行既定切换方案（单物体视觉定位 + MoveIt2/IK）。

具体任务编号与范围须由用户确认，不要自行扩大范围。

## 8. 验证记录

- `git status`：`main` 与 `origin/main` 同步，工作树干净。
- `git log`：HEAD = `8ed1239`。
- 纯 Python Mock 回归（在 `src/robot_task` 下以 `PYTHONPATH=.` 执行）：
  - `test_task_runner`：12 PASS
  - `test_skill_loop`：7 PASS
  - `test_skill_registry`：22 PASS
  - `test_skill_selector`：9 PASS
  - `test_mock_skill_executor`：5 PASS
- Task 006B–006E 的 50 项测试依赖 `py_trees` 运行环境；本轮未在缺失 `pydot` 的系统 Python 中重跑，测试数量以仓库内测试文件与已审阅记录为准。
- 以上只证明纯 Python / Mock 层，不代表真实 ROS 2、Nav2、Scout、Piper6 或 ACT 真机已验证。

## 9. 协作规则

- ChatGPT：设计、教学、审阅、拆分窄任务。
- OpenCode：只执行用户授权的 `.md` 窄任务，不擅自扩大范围。
- 用户：阅读代码与 Diff、验收测试、决定 Git Commit / Push。
- 未经明确授权，不自动 `git add` / `commit` / `push`，不运行真实机器人或发送 Action Goal。

## 10. 新 Chat 恢复流程

1. 执行 `git rev-parse HEAD` 和 `git status --short`，确认是否仍在 `8ed1239`。
2. 阅读本文件，确认当前处于 Phase 0 离线诊断还是全链集成阶段。
3. 阅读 Phase 0 诊断报告与视觉样本（若已生成），核对 J2/J3 边界钉死与越零结论。
4. 阅读 `task_manager.py`、`task_runner.py`、`test_task_runner.py`、`task_bt.py`，以及 `bt_skill_executor.py`、`navigate_region_action_adapter.py`、`navigate_bt_execution.py`、`ros_mock_navigation_runner.py`。
5. 核对 GitHub 上的 Commit 是否已包含本地新增文件；不要假设本文档中的 SHA 等于最终推送 Commit。
