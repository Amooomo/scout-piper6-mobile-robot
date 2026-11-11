
from robot_task.skill_types import (
    SkillID,
    SkillCandidate,
    RobotState,
)

from robot_task.mock_skill_executor import MockSkillExecutor


def main():

    executor = MockSkillExecutor()

    state = RobotState(
        current_region_id="start",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )

    candidate = SkillCandidate(
        skill_id=SkillID.NAVIGATE_REGION,
        score=0.90,
        arguments={
            "region_id": "experiment_table_A"
        },
    )

    result = executor.execute(candidate, state)

    assert result.success
    assert result.state_after.current_region_id == "experiment_table_A"
    assert result.state_after.docking_complete is False

    # 确认原来的 RobotState 没有被修改
    assert state.current_region_id == "start"

    print("[PASS] Navigation execution")
    print("Before:", state)
    print("After:", result.state_after)
    print("Message:", result.message)


if __name__ == "__main__":
    main()
