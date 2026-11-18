from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillDecision,
    SkillExecutionResult,
    SkillID,
    TaskGoal,
    TaskStatus,
)

from .mock_skill_executor import MockSkillExecutor
from .mock_skill_selector import MockSkillSelector
from .skill_decision_engine import SkillDecisionEngine
from .skill_registry import SkillRegistry
from .task_manager import TaskManager


class EmptySelector:

    def score_candidates(self, goal, state):
        return []


class FixedSelector:

    def __init__(self, candidates):
        self.candidates = candidates

    def score_candidates(self, goal, state):
        return list(self.candidates)


class FixedDecisionEngine:

    def __init__(self, selected):
        self.selected = selected

    def decide(self, candidates, goal, state):
        return SkillDecision(selected=self.selected)


class RecordingExecutor:

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def execute(self, candidate, state):
        self.calls.append((candidate, state))

        if self.result is not None:
            return self.result

        return SkillExecutionResult(
            success=True,
            message="Stub execution succeeded",
            state_after=state,
        )


def make_goal():
    return TaskGoal(
        instruction="Pick the target object",
        target_region_id="experiment_table_A",
        target_object_id="target_object",
    )


def make_start_state():
    return RobotState(
        current_region_id="start",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )


def make_real_manager(max_steps=10):
    registry = SkillRegistry()

    return TaskManager(
        selector=MockSkillSelector(),
        decision_engine=SkillDecisionEngine(registry),
        executor=MockSkillExecutor(),
        max_steps=max_steps,
    )


def test_success_path():

    goal = make_goal()
    initial_state = make_start_state()
    manager = make_real_manager()

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.SUCCEEDED

    assert (
        result.final_state.current_region_id
        == goal.target_region_id
    )
    assert result.final_state.docking_complete is True
    assert result.final_state.object_detected is True
    assert result.final_state.object_grasped is True
    assert result.final_state.result_verified is True

    assert result.progress.completed_skills == [
        SkillID.NAVIGATE_REGION,
        SkillID.VISUAL_DOCK,
        SkillID.DETECT_OBJECT,
        SkillID.MANIPULATE_PICK,
        SkillID.VERIFY_RESULT,
    ]
    assert result.progress.step_count == 6
    assert len(result.execution_results) == 5

    assert all(
        execution.success
        for execution in result.execution_results
    )

    # 初始 RobotState 没有被修改
    assert initial_state == make_start_state()

    print("[PASS] Full success loop")
    print("  Skills:", result.progress.completed_skills)
    print("  Steps:", result.progress.step_count)
    print("  Final state:", result.final_state)


def test_no_feasible_candidates():

    goal = make_goal()
    initial_state = make_start_state()
    executor = RecordingExecutor()

    manager = TaskManager(
        selector=EmptySelector(),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=5,
    )

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.FAILED
    assert result.progress.step_count == 1
    assert result.progress.completed_skills == []
    assert result.execution_results == ()
    assert executor.calls == []
    assert result.final_state == initial_state

    print("[PASS] No feasible candidates")
    print("  Message:", result.progress.last_message)


def test_ask_human():

    goal = make_goal()
    initial_state = make_start_state()
    executor = RecordingExecutor()

    candidates = [
        SkillCandidate(
            skill_id=SkillID.ASK_HUMAN,
            score=0.90,
            arguments={},
        ),
    ]

    manager = TaskManager(
        selector=FixedSelector(candidates),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=5,
    )

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.NEEDS_HUMAN
    assert result.progress.current_skill == SkillID.ASK_HUMAN
    assert result.progress.step_count == 1
    assert result.progress.completed_skills == []
    assert result.execution_results == ()
    assert executor.calls == []
    assert result.final_state == initial_state

    print("[PASS] ASK_HUMAN terminates without execution")
    print("  Message:", result.progress.last_message)


def test_executor_failure():

    goal = make_goal()
    initial_state = make_start_state()

    failed_state = RobotState(
        current_region_id="untrusted_region",
        docking_complete=True,
        object_detected=True,
        object_grasped=True,
        result_verified=True,
    )

    failed_result = SkillExecutionResult(
        success=False,
        message="Simulated execution failure",
        state_after=failed_state,
    )

    executor = RecordingExecutor(result=failed_result)

    candidates = [
        SkillCandidate(
            skill_id=SkillID.NAVIGATE_REGION,
            score=0.90,
            arguments={
                "region_id": goal.target_region_id
            },
        ),
    ]

    manager = TaskManager(
        selector=FixedSelector(candidates),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=5,
    )

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.FAILED
    assert result.progress.step_count == 1
    assert result.progress.completed_skills == []
    assert result.execution_results == (failed_result,)
    assert len(executor.calls) == 1

    # 失败结果中的 state_after 不被接受
    assert result.final_state == initial_state
    assert initial_state == make_start_state()

    print("[PASS] Executor failure stops the loop")
    print("  Message:", result.progress.last_message)


def test_max_steps():

    goal = make_goal()
    initial_state = make_start_state()
    manager = make_real_manager(max_steps=3)

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.FAILED
    assert result.progress.step_count == 3
    assert result.progress.completed_skills == [
        SkillID.NAVIGATE_REGION,
        SkillID.VISUAL_DOCK,
    ]
    assert len(result.execution_results) == 2

    assert (
        "maximum decision steps"
        in result.progress.last_message
    )

    print("[PASS] Maximum step limit stops the loop")
    print("  Message:", result.progress.last_message)


def test_unverified_done_is_rejected():

    goal = make_goal()
    initial_state = make_start_state()
    executor = RecordingExecutor()

    selected = SkillCandidate(
        skill_id=SkillID.DONE,
        score=0.99,
        arguments={},
    )

    manager = TaskManager(
        selector=EmptySelector(),
        decision_engine=FixedDecisionEngine(selected),
        executor=executor,
        max_steps=5,
    )

    result = manager.run(goal, initial_state)

    assert result.progress.status == TaskStatus.FAILED
    assert result.progress.current_skill == SkillID.DONE
    assert result.progress.step_count == 1
    assert result.progress.completed_skills == []
    assert result.execution_results == ()
    assert executor.calls == []
    assert result.final_state == initial_state

    print("[PASS] Unverified DONE is rejected")
    print("  Message:", result.progress.last_message)


def test_invalid_max_steps():

    for invalid_value in (0, -1, 1.5, True):

        try:
            TaskManager(
                selector=MockSkillSelector(),
                decision_engine=SkillDecisionEngine(
                    SkillRegistry()
                ),
                executor=MockSkillExecutor(),
                max_steps=invalid_value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"max_steps={invalid_value!r} should be rejected"
            )

    print("[PASS] Invalid max_steps rejected")


def main():

    print("\n========== Skill Loop Tests ==========\n")

    test_success_path()
    test_no_feasible_candidates()
    test_ask_human()
    test_executor_failure()
    test_max_steps()
    test_unverified_done_is_rejected()
    test_invalid_max_steps()

    print("\n========== All Tests Passed ==========\n")


if __name__ == "__main__":
    main()
