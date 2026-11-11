
from dataclasses import replace

from robot_task.skill_types import (
    SkillID,
    SkillCandidate,
    RobotState,
    SkillExecutionResult,
)


class MockSkillExecutor:

    def execute(
        self,
        candidate: SkillCandidate,
        state: RobotState,
    ) -> SkillExecutionResult:

        skill_id = candidate.skill_id
        arguments = candidate.arguments

        # 1. 导航到目标区域
        if skill_id == SkillID.NAVIGATE_REGION:

            region_id = arguments.get("region_id")

            if not isinstance(region_id, str) or not region_id.strip():
                return SkillExecutionResult(
                    success=False,
                    message="Missing or invalid region_id",
                    state_after=state,
                )

            new_state = replace(
                state,
                current_region_id=region_id,
                docking_complete=False,
                object_detected=False,
                object_grasped=False,
                result_verified=False,
            )

            return SkillExecutionResult(
                success=True,
                message=f"Arrived at {region_id}",
                state_after=new_state,
            )

        # 2. 视觉精确停靠
        elif skill_id == SkillID.VISUAL_DOCK:

            if state.current_region_id is None:
                return SkillExecutionResult(
                    success=False,
                    message="Current region is unknown",
                    state_after=state,
                )

            new_state = replace(
                state,
                docking_complete=True,
            )

            return SkillExecutionResult(
                success=True,
                message="Visual docking completed",
                state_after=new_state,
            )

        # 3. 检测目标物体
        elif skill_id == SkillID.DETECT_OBJECT:

            object_id = arguments.get("object_id")

            if not isinstance(object_id, str) or not object_id.strip():
                return SkillExecutionResult(
                    success=False,
                    message="Missing or invalid object_id",
                    state_after=state,
                )

            if state.docking_complete is not True:
                return SkillExecutionResult(
                    success=False,
                    message="Robot has not completed docking",
                    state_after=state,
                )

            new_state = replace(
                state,
                object_detected=True,
            )

            return SkillExecutionResult(
                success=True,
                message=f"Detected object: {object_id}",
                state_after=new_state,
            )

        # 4. 抓取目标物体
        elif skill_id == SkillID.MANIPULATE_PICK:

            object_id = arguments.get("object_id")

            if not isinstance(object_id, str) or not object_id.strip():
                return SkillExecutionResult(
                    success=False,
                    message="Missing or invalid object_id",
                    state_after=state,
                )

            if state.object_detected is not True:
                return SkillExecutionResult(
                    success=False,
                    message="Target object has not been detected",
                    state_after=state,
                )

            new_state = replace(
                state,
                object_grasped=True,
            )

            return SkillExecutionResult(
                success=True,
                message=f"Picked object: {object_id}",
                state_after=new_state,
            )

        # 5. 验证任务结果
        elif skill_id == SkillID.VERIFY_RESULT:

            if state.object_grasped is not True:
                return SkillExecutionResult(
                    success=False,
                    message="No object has been grasped",
                    state_after=state,
                )

            new_state = replace(
                state,
                result_verified=True,
            )

            return SkillExecutionResult(
                success=True,
                message="Task result verified",
                state_after=new_state,
            )

        # DONE 和 ASK_HUMAN 由 TaskManager 处理
        elif skill_id in (
            SkillID.DONE,
            SkillID.ASK_HUMAN,
        ):
            raise ValueError(
                f"{skill_id.value} is a task-level decision, "
                "not an executable skill"
            )

        else:
            raise ValueError(
                f"Unsupported skill: {skill_id}"
            )
