
from robot_task.skill_types import (
    SkillID,
    SkillCandidate,
    RobotState,
)

from robot_task.mock_skill_executor import MockSkillExecutor


def test_visual_dock(executor):

    state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )

    candidate = SkillCandidate(
        skill_id=SkillID.VISUAL_DOCK,
        score=0.90,
        arguments={},
    )

    result = executor.execute(candidate, state)

    assert result.success is True
    assert result.state_after.docking_complete is True

    # 其他状态保持不变
    assert result.state_after.current_region_id == "experiment_table_A"
    assert result.state_after.object_detected is False
    assert result.state_after.object_grasped is False
    assert result.state_after.result_verified is False

    # 确认原来的 RobotState 没有被修改
    assert state.docking_complete is False

    print("[PASS] Visual dock execution")


def test_detect_object(executor):

    state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=True,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )

    candidate = SkillCandidate(
        skill_id=SkillID.DETECT_OBJECT,
        score=0.90,
        arguments={
            "object_id": "red_cube"
        },
    )

    result = executor.execute(candidate, state)

    assert result.success is True
    assert result.state_after.object_detected is True

    # 其他状态保持不变
    assert result.state_after.current_region_id == "experiment_table_A"
    assert result.state_after.docking_complete is True
    assert result.state_after.object_grasped is False
    assert result.state_after.result_verified is False

    # 确认原来的 RobotState 没有被修改
    assert state.object_detected is False

    print("[PASS] Object detection execution")


def test_manipulate_pick(executor):

    state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=True,
        object_detected=True,
        object_grasped=False,
        result_verified=False,
    )

    candidate = SkillCandidate(
        skill_id=SkillID.MANIPULATE_PICK,
        score=0.90,
        arguments={
            "object_id": "red_cube"
        },
    )

    result = executor.execute(candidate, state)

    assert result.success is True
    assert result.state_after.object_grasped is True

    # 其他状态保持不变
    assert result.state_after.current_region_id == "experiment_table_A"
    assert result.state_after.docking_complete is True
    assert result.state_after.object_detected is True
    assert result.state_after.result_verified is False

    # 确认原来的 RobotState 没有被修改
    assert state.object_grasped is False

    print("[PASS] Manipulation pick execution")


def test_verify_result(executor):

    state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=True,
        object_detected=True,
        object_grasped=True,
        result_verified=False,
    )

    candidate = SkillCandidate(
        skill_id=SkillID.VERIFY_RESULT,
        score=0.90,
        arguments={},
    )

    result = executor.execute(candidate, state)

    assert result.success is True
    assert result.state_after.result_verified is True

    # 其他状态保持不变
    assert result.state_after.current_region_id == "experiment_table_A"
    assert result.state_after.docking_complete is True
    assert result.state_after.object_detected is True
    assert result.state_after.object_grasped is True

    # 确认原来的 RobotState 没有被修改
    assert state.result_verified is False

    print("[PASS] Result verification execution")


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

    test_visual_dock(executor)
    test_detect_object(executor)
    test_manipulate_pick(executor)
    test_verify_result(executor)


if __name__ == "__main__":
    main()
