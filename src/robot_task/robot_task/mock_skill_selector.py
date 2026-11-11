from .skill_types import (
    TaskGoal,
    RobotState,
    SkillID,
    SkillCandidate,
)


class MockSkillSelector:
    """
    模拟 JEV/VLM 的 Skill 候选评分器。

    输入：
        TaskGoal
        RobotState

    输出：
        每个候选 Skill 的相关性分数。

    注意：
        本类不负责判断 Skill 是否可执行。
        可执行性由 SkillRegistry 检查。
    """

    def score_candidates(
        self,
        goal: TaskGoal,
        state: RobotState,
    ) -> list[SkillCandidate]:

        # 默认给所有 Skill 一个较低的相关性分数
        scores = {
            skill_id: 0.05
            for skill_id in SkillID
        }

        # 根据当前任务阶段，模拟模型对下一步 Skill 的判断

        # 阶段 1：还没有到达目标区域
        if (
            goal.target_region_id
            and state.current_region_id != goal.target_region_id
        ):
            scores[SkillID.NAVIGATE_REGION] = 0.90

        # 阶段 2：已经到达目标区域，但尚未完成视觉对接
        elif state.docking_complete is False:
            scores[SkillID.VISUAL_DOCK] = 0.90

        # 阶段 3：视觉对接完成，尚未检测到目标物体
        elif (
            state.docking_complete is True
            and state.object_detected is not True
        ):
            scores[SkillID.DETECT_OBJECT] = 0.90

        # 阶段 4：已经检测到目标，但尚未抓取
        elif (
            state.object_detected is True
            and state.object_grasped is False
        ):
            scores[SkillID.MANIPULATE_PICK] = 0.90

        # 阶段 5：已经抓取，尚未验证结果
        elif (
            state.object_grasped is True
            and state.result_verified is not True
        ):
            scores[SkillID.VERIFY_RESULT] = 0.90

        # 阶段 6：任务结果已经验证
        elif state.result_verified is True:
            scores[SkillID.DONE] = 0.99

        # 人工介入始终作为低分候选项
        scores[SkillID.ASK_HUMAN] = 0.01

        # 为每个候选 Skill 准备参数
        arguments = {
            SkillID.NAVIGATE_REGION: {
                "region_id": goal.target_region_id
            },
            SkillID.VISUAL_DOCK: {
                "region_id": goal.target_region_id
            },
            SkillID.DETECT_OBJECT: {
                "object_id": goal.target_object_id
            },
            SkillID.MANIPULATE_PICK: {
                "object_id": goal.target_object_id
            },
            SkillID.VERIFY_RESULT: {
                "object_id": goal.target_object_id
            },
            SkillID.DONE: {},
            SkillID.ASK_HUMAN: {},
        }

        candidates = []

        for skill_id in SkillID:
            candidate = SkillCandidate(
                skill_id=skill_id,
                score=scores[skill_id],
                arguments=arguments[skill_id],
            )
            candidates.append(candidate)

        return candidates