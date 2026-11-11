from .skill_types import (
    TaskGoal,
    RobotState,
    SkillID,
    SkillCandidate,
)

from .skill_registry import SkillRegistry
from .mock_skill_selector import MockSkillSelector
from .skill_decision_engine import SkillDecisionEngine


def main():

    goal = TaskGoal(
        instruction="Go to table A and pick the target object",
        target_region_id="experiment_table_A",
        target_object_id="target_object",
    )

    selector = MockSkillSelector()

    registry = SkillRegistry()

    engine = SkillDecisionEngine(registry)

    # 测试不同任务阶段
    test_cases = [
        (
            "Navigation",
            RobotState(
                current_region_id=None,
            ),
            SkillID.NAVIGATE_REGION,
        ),

        (
            "Visual docking",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=False,
            ),
            SkillID.VISUAL_DOCK,
        ),

        (
            "Object detection",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=False,
            ),
            SkillID.DETECT_OBJECT,
        ),

        (
            "Manipulation",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=False,
            ),
            SkillID.MANIPULATE_PICK,
        ),

        (
            "Verification",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=True,
                result_verified=False,
            ),
            SkillID.VERIFY_RESULT,
        ),

        (
            "Done",
            RobotState(
                current_region_id="experiment_table_A",
                docking_complete=True,
                object_detected=True,
                object_grasped=True,
                result_verified=True,
            ),
            SkillID.DONE,
        ),
    ]

    print("\n========== Skill Selection Tests ==========\n")

    for name, state, expected_skill in test_cases:

        candidates = selector.score_candidates(
            goal,
            state,
        )

        decision = engine.decide(
            candidates,
            goal,
            state,
        )

        assert decision.selected is not None

        actual_skill = decision.selected.skill_id

        assert actual_skill == expected_skill, (
            f"{name}: expected {expected_skill}, "
            f"got {actual_skill}"
        )

        print(
            f"[PASS] {name}: "
            f"{actual_skill.value}, "
            f"score={decision.selected.score:.2f}"
        )

    # 测试：状态不明确时，不能执行视觉对接
    uncertain_state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=None,
        object_detected=False,
        object_grasped=False,
    )

    candidates = selector.score_candidates(
        goal,
        uncertain_state,
    )

    decision = engine.decide(
        candidates,
        goal,
        uncertain_state,
    )

    assert decision.selected is not None
    assert decision.selected.skill_id == SkillID.ASK_HUMAN

    print(
        "[PASS] Uncertain state fallback: "
        f"{decision.selected.skill_id.value}"
    )

    # 测试：高分不代表一定能够执行
    custom_candidates = [
        SkillCandidate(
            skill_id=SkillID.NAVIGATE_REGION,
            score=0.99,
            arguments={
                "region_id": "experiment_table_A"
            },
        ),
        SkillCandidate(
            skill_id=SkillID.VISUAL_DOCK,
            score=0.70,
            arguments={
                "region_id": "experiment_table_A"
            },
        ),
        SkillCandidate(
            skill_id=SkillID.ASK_HUMAN,
            score=0.01,
            arguments={},
        ),
    ]

    docking_state = RobotState(
        current_region_id="experiment_table_A",
        docking_complete=False,
    )

    decision = engine.decide(
        custom_candidates,
        goal,
        docking_state,
    )

    assert decision.selected is not None
    assert decision.selected.skill_id == SkillID.VISUAL_DOCK

    assert (
        SkillID.NAVIGATE_REGION.value
        in decision.rejected_candidates
    )

    print(
        "[PASS] Feasibility overrides score: "
        f"{decision.selected.skill_id.value}"
    )

    print("\n========== All Tests Passed ==========\n")


if __name__ == "__main__":
    main()