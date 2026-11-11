from .skill_types import (
    TaskGoal,
    RobotState,
    SkillID
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

        allowed, reason = registry.check(
            skill_id,
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

    allowed, reason = registry.check(
        SkillID.MANIPULATE_PICK,
        goal,
        unsafe_state
    )

    assert not allowed

    print(
        f"[PASS] Unsafe manipulation rejected: {reason}"
    )


if __name__ == "__main__":
    main()