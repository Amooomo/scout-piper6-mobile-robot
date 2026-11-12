from .skill_types import (
    TaskGoal,
    RobotState,
    SkillID,
    SkillCandidate,
)

from .skill_registry import SkillRegistry


def main():

    registry = SkillRegistry()

    goal = TaskGoal(
        instruction="Go to table A and pick the target object",
        target_region_id="experiment_table_A",
        target_object_id="target_object"
    )

    test_cases = [

        (
            "Navigation",
            RobotState(
                current_region_id=None
            ),
            SkillID.NAVIGATE_REGION
        ),

        (
            "Visual docking",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=False
            ),
            SkillID.VISUAL_DOCK
        ),

        (
            "Object detection",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False
            ),
            SkillID.DETECT_OBJECT
        ),

        (
            "Manipulation",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False
            ),
            SkillID.MANIPULATE_PICK
        ),

        (
            "Verification",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=True,
                result_verified=False
            ),
            SkillID.VERIFY_RESULT
        ),

        (
            "Done",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=True,
                result_verified=True
            ),
            SkillID.DONE
        )
    ]

    for name, state, skill_id in test_cases:

        if skill_id == SkillID.NAVIGATE_REGION:
            arguments = {
                "region_id": goal.target_region_id
            }
        elif skill_id in (
            SkillID.DETECT_OBJECT,
            SkillID.MANIPULATE_PICK,
        ):
            arguments = {
                "object_id": goal.target_object_id
            }
        else:
            arguments = {}

        candidate = SkillCandidate(
            skill_id=skill_id,
            score=0.50,
            arguments=arguments,
        )

        allowed, reason = registry.check(
            candidate,
            goal,
            state
        )

        assert allowed, (
            f"{name} failed: {reason}"
        )

        print(
            f"[PASS] {name}: {reason}"
        )

    # 额外测试：未完成停靠时，禁止机械臂操作
    unsafe_state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=None,
        object_detected=True,
        object_grasped=False
    )

    unsafe_candidate = SkillCandidate(
        skill_id=SkillID.MANIPULATE_PICK,
        score=0.50,
        arguments={
            "object_id": goal.target_object_id
        },
    )

    allowed, reason = registry.check(
        unsafe_candidate,
        goal,
        unsafe_state
    )

    assert not allowed

    print(
        f"[PASS] Unsafe manipulation rejected: {reason}"
    )

    # 参数缺失、类型错误、空白或与 TaskGoal 不一致时均应拒绝
    invalid_cases = [
        (
            "Navigation without region_id",
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.90,
                arguments={},
            ),
            goal,
            RobotState(current_region_id=None),
        ),
        (
            "Navigation with empty region_id",
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.90,
                arguments={"region_id": "   "},
            ),
            goal,
            RobotState(current_region_id=None),
        ),
        (
            "Navigation with non-string region_id",
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.90,
                arguments={"region_id": 42},
            ),
            goal,
            RobotState(current_region_id=None),
        ),
        (
            "Navigation with mismatched region_id",
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.90,
                arguments={"region_id": "another_table"},
            ),
            goal,
            RobotState(current_region_id=None),
        ),
        (
            "Navigation without target region in goal",
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.90,
                arguments={"region_id": "experiment_table_A"},
            ),
            TaskGoal(instruction="No target region"),
            RobotState(current_region_id=None),
        ),
        (
            "Detection without object_id",
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.90,
                arguments={},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
        ),
        (
            "Detection with empty object_id",
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.90,
                arguments={"object_id": "   "},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
        ),
        (
            "Detection with non-string object_id",
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.90,
                arguments={"object_id": 123},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
        ),
        (
            "Detection with mismatched object_id",
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.90,
                arguments={"object_id": "another_object"},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
        ),
        (
            "Detection without target object in goal",
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.90,
                arguments={"object_id": "target_object"},
            ),
            TaskGoal(
                instruction="No target object",
                target_region_id="experiment_table_A",
            ),
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
        ),
        (
            "Manipulation without object_id",
            SkillCandidate(
                skill_id=SkillID.MANIPULATE_PICK,
                score=0.90,
                arguments={},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
        ),
        (
            "Manipulation with empty object_id",
            SkillCandidate(
                skill_id=SkillID.MANIPULATE_PICK,
                score=0.90,
                arguments={"object_id": "   "},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
        ),
        (
            "Manipulation with non-string object_id",
            SkillCandidate(
                skill_id=SkillID.MANIPULATE_PICK,
                score=0.90,
                arguments={"object_id": None},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
        ),
        (
            "Manipulation with mismatched object_id",
            SkillCandidate(
                skill_id=SkillID.MANIPULATE_PICK,
                score=0.90,
                arguments={"object_id": "another_object"},
            ),
            goal,
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
        ),
        (
            "Manipulation without target object in goal",
            SkillCandidate(
                skill_id=SkillID.MANIPULATE_PICK,
                score=0.90,
                arguments={"object_id": "target_object"},
            ),
            TaskGoal(
                instruction="No target object",
                target_region_id="experiment_table_A",
            ),
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
        ),
    ]

    for name, candidate, case_goal, case_state in invalid_cases:

        allowed, reason = registry.check(
            candidate,
            case_goal,
            case_state,
        )

        assert not allowed, (
            f"{name} should be rejected"
        )

        print(
            f"[PASS] {name} rejected: {reason}"
        )


if __name__ == "__main__":
    main()