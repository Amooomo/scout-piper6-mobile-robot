from .mock_skill_executor import MockSkillExecutor
from .mock_skill_selector import MockSkillSelector
from .skill_decision_engine import SkillDecisionEngine
from .skill_registry import SkillRegistry
from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillDecision,
    SkillExecutionResult,
    SkillID,
    TaskGoal,
    TaskStatus,
)
from .task_runner import TaskRunner


class FakeHandle:

    def __init__(self, result, pending_polls=0):
        self.result = result
        self.pending_polls = pending_polls
        self.poll_calls = 0
        self.cancel_calls = 0

    def poll(self):
        self.poll_calls += 1

        if self.pending_polls > 0:
            self.pending_polls -= 1
            return None

        return self.result

    def cancel(self):
        self.cancel_calls += 1


class FakeAsyncExecutor:

    def __init__(self, result_factory=None, pending_polls=0):
        self.result_factory = result_factory
        self.pending_polls = pending_polls
        self.start_calls = []
        self.handles = []
        self._sync_executor = (
            MockSkillExecutor()
            if result_factory is None
            else None
        )

    def start(self, candidate, state):
        self.start_calls.append((candidate, state))

        if self.result_factory is None:
            result = self._sync_executor.execute(
                candidate,
                state,
            )
        else:
            result = self.result_factory(candidate, state)

        handle = FakeHandle(result, self.pending_polls)
        self.handles.append(handle)
        return handle


class CountingSelector:

    def __init__(self):
        self.inner = MockSkillSelector()
        self.calls = 0

    def score_candidates(self, goal, state):
        self.calls += 1
        return self.inner.score_candidates(goal, state)


class CountingDecisionEngine:

    def __init__(self):
        self.inner = SkillDecisionEngine(SkillRegistry())
        self.calls = 0

    def decide(self, candidates, goal, state):
        self.calls += 1
        return self.inner.decide(candidates, goal, state)


class EmptySelector:

    def score_candidates(self, goal, state):
        return []


class FixedSelector:

    def __init__(self, candidates):
        self.candidates = candidates

    def score_candidates(self, goal, state):
        return list(self.candidates)


class FixedDecisionEngine:

    def __init__(self, decision):
        self.decision = decision

    def decide(self, candidates, goal, state):
        return self.decision


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


def make_verified_state(goal):
    return RobotState(
        current_region_id=goal.target_region_id,
        docking_complete=True,
        object_detected=True,
        object_grasped=True,
        result_verified=True,
    )


def make_runner(executor, max_steps=10):
    return TaskRunner(
        selector=MockSkillSelector(),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=max_steps,
    )


def run_until_finished(runner, limit=100):
    for _ in range(limit):
        progress = runner.update()

        if progress.status != TaskStatus.RUNNING:
            return progress

    raise AssertionError("TaskRunner did not finish")


def test_success_path():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=2)
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    progress = run_until_finished(runner)

    assert progress.status == TaskStatus.SUCCEEDED
    assert progress.completed_skills == [
        SkillID.NAVIGATE_REGION,
        SkillID.VISUAL_DOCK,
        SkillID.DETECT_OBJECT,
        SkillID.MANIPULATE_PICK,
        SkillID.VERIFY_RESULT,
    ]
    assert progress.step_count == 6
    assert len(runner.execution_results) == 5
    assert len(executor.start_calls) == 5

    state = runner.current_state
    assert state.current_region_id == goal.target_region_id
    assert state.docking_complete is True
    assert state.object_detected is True
    assert state.object_grasped is True
    assert state.result_verified is True

    # 初始 RobotState 未被修改
    assert initial_state == make_start_state()

    print("[PASS] Async success loop")
    print("  Skills:", progress.completed_skills)
    print("  Steps:", progress.step_count)


def test_pending_update_does_not_redispatch():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=3)
    selector = CountingSelector()
    engine = CountingDecisionEngine()

    runner = TaskRunner(
        selector=selector,
        decision_engine=engine,
        executor=executor,
        max_steps=10,
    )

    runner.start(goal, initial_state)

    progress = runner.update()
    assert progress.step_count == 1
    assert selector.calls == 1
    assert engine.calls == 1
    assert len(executor.start_calls) == 1
    assert len(executor.handles) == 1

    handle = executor.handles[0]
    assert runner.current_state == initial_state
    assert progress.completed_skills == []
    assert runner.execution_results == ()

    for _ in range(3):
        progress = runner.update()

        assert progress.status == TaskStatus.RUNNING
        assert progress.step_count == 1
        assert selector.calls == 1
        assert engine.calls == 1
        assert len(executor.start_calls) == 1
        assert runner.current_state == initial_state
        assert progress.completed_skills == []

    # 本次轮询才拿到终态并提交状态
    progress = runner.update()

    assert progress.status == TaskStatus.RUNNING
    assert progress.step_count == 1
    assert progress.completed_skills == [
        SkillID.NAVIGATE_REGION
    ]
    assert runner.current_state.current_region_id == (
        goal.target_region_id
    )
    assert handle.poll_calls == 4

    print("[PASS] Pending updates do not re-dispatch")


def test_arguments_passthrough():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=0)
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    run_until_finished(runner)

    started = {
        candidate.skill_id: candidate
        for candidate, _ in executor.start_calls
    }

    assert started[SkillID.NAVIGATE_REGION].arguments == {
        "region_id": goal.target_region_id
    }
    assert started[SkillID.DETECT_OBJECT].arguments == {
        "object_id": goal.target_object_id
    }
    assert started[SkillID.MANIPULATE_PICK].arguments == {
        "object_id": goal.target_object_id
    }

    # 执行完成后候选参数保持不变
    assert started[SkillID.NAVIGATE_REGION].arguments == {
        "region_id": goal.target_region_id
    }
    assert started[SkillID.DETECT_OBJECT].arguments == {
        "object_id": goal.target_object_id
    }

    print("[PASS] Candidate arguments pass through unchanged")


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

    executor = FakeAsyncExecutor(
        result_factory=lambda candidate, state: failed_result,
        pending_polls=0,
    )
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    progress = run_until_finished(runner)

    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 1
    assert progress.completed_skills == []
    assert runner.execution_results == (failed_result,)
    assert runner.current_state == initial_state
    assert len(executor.start_calls) == 1

    print("[PASS] Execution failure does not commit state")


def test_cancel_while_pending():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=100)
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    progress = runner.update()

    assert progress.status == TaskStatus.RUNNING
    assert len(executor.handles) == 1

    handle = executor.handles[0]
    runner.cancel()

    assert handle.cancel_calls == 1
    assert "cancel_requested" in progress.last_message

    for _ in range(2):
        progress = runner.update()

        assert progress.status == TaskStatus.RUNNING
        assert progress.step_count == 1
        assert len(executor.start_calls) == 1
        assert runner.current_state == initial_state
        assert progress.completed_skills == []

    # 重复 cancel 不重复请求
    runner.cancel()
    assert handle.cancel_calls == 1

    print("[PASS] Cancel while pending waits without re-dispatch")


def test_cancel_then_success_result():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=1)
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    progress = runner.update()

    handle = executor.handles[0]
    runner.cancel()
    assert handle.cancel_calls == 1

    progress = runner.update()
    assert progress.status == TaskStatus.RUNNING

    # 底层恰好返回 success=True，但不能提交为任务成功
    progress = runner.update()

    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 1
    assert progress.completed_skills == []
    assert runner.current_state == initial_state
    assert len(runner.execution_results) == 1
    assert runner.execution_results[0].success is True
    assert "cancel_requested" in progress.last_message
    assert "cancel_confirmed_or_settled" in progress.last_message

    print("[PASS] Success after cancel is not accepted")


def test_terminal_decisions_without_executor():

    goal = make_goal()
    initial_state = make_start_state()

    # 无可行候选
    executor = FakeAsyncExecutor()
    runner = TaskRunner(
        selector=EmptySelector(),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=5,
    )

    runner.start(goal, initial_state)
    progress = runner.update()

    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 1
    assert executor.start_calls == []

    # ASK_HUMAN
    ask_candidate = SkillCandidate(
        skill_id=SkillID.ASK_HUMAN,
        score=0.90,
        arguments={},
    )
    executor = FakeAsyncExecutor()
    runner = TaskRunner(
        selector=FixedSelector([ask_candidate]),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=5,
    )

    runner.start(goal, initial_state)
    progress = runner.update()

    assert progress.status == TaskStatus.NEEDS_HUMAN
    assert progress.current_skill == SkillID.ASK_HUMAN
    assert progress.step_count == 1
    assert executor.start_calls == []

    # 未验证 DONE
    done_candidate = SkillCandidate(
        skill_id=SkillID.DONE,
        score=0.99,
        arguments={},
    )
    executor = FakeAsyncExecutor()
    runner = TaskRunner(
        selector=EmptySelector(),
        decision_engine=FixedDecisionEngine(
            SkillDecision(selected=done_candidate)
        ),
        executor=executor,
        max_steps=5,
    )

    runner.start(goal, initial_state)
    progress = runner.update()

    assert progress.status == TaskStatus.FAILED
    assert progress.current_skill == SkillID.DONE
    assert progress.step_count == 1
    assert executor.start_calls == []
    assert "not verified" in progress.last_message

    print("[PASS] No candidate / ASK_HUMAN / unverified DONE")


def test_max_steps_stops_before_ordinary_skill():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor(pending_polls=0)
    runner = make_runner(executor, max_steps=3)

    runner.start(goal, initial_state)
    progress = run_until_finished(runner)

    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 3
    assert progress.completed_skills == [
        SkillID.NAVIGATE_REGION,
        SkillID.VISUAL_DOCK,
    ]
    assert len(runner.execution_results) == 2
    assert len(executor.start_calls) == 2
    assert "maximum decision steps" in progress.last_message

    print("[PASS] max_steps boundary does not start ordinary skill")


def test_max_steps_allows_terminal_decisions():

    goal = make_goal()

    # 边界轮选中 DONE，允许成功
    executor = FakeAsyncExecutor()
    runner = make_runner(executor, max_steps=1)
    runner.start(goal, make_verified_state(goal))
    progress = runner.update()

    assert progress.status == TaskStatus.SUCCEEDED
    assert progress.step_count == 1
    assert executor.start_calls == []

    # 边界轮选中 ASK_HUMAN，允许终止
    ask_candidate = SkillCandidate(
        skill_id=SkillID.ASK_HUMAN,
        score=0.90,
        arguments={},
    )
    executor = FakeAsyncExecutor()
    runner = TaskRunner(
        selector=FixedSelector([ask_candidate]),
        decision_engine=SkillDecisionEngine(SkillRegistry()),
        executor=executor,
        max_steps=1,
    )

    runner.start(goal, make_start_state())
    progress = runner.update()

    assert progress.status == TaskStatus.NEEDS_HUMAN
    assert progress.step_count == 1
    assert executor.start_calls == []

    print("[PASS] max_steps boundary allows DONE / ASK_HUMAN")


def test_cancel_before_dispatch():

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor()
    runner = make_runner(executor)

    runner.start(goal, initial_state)
    runner.cancel()

    progress = runner.update()

    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 0
    assert executor.start_calls == []
    assert runner.current_state == initial_state
    assert "cancel_requested before dispatch" in (
        progress.last_message
    )

    # 终态 update 幂等
    progress = runner.update()
    assert progress.status == TaskStatus.FAILED
    assert progress.step_count == 0

    print("[PASS] Cancel before dispatch does not start a skill")


def test_terminal_update_is_idempotent():

    goal = make_goal()
    executor = FakeAsyncExecutor()
    runner = make_runner(executor, max_steps=1)
    runner.start(goal, make_verified_state(goal))

    progress = runner.update()
    assert progress.status == TaskStatus.SUCCEEDED

    completed = list(progress.completed_skills)
    step_count = progress.step_count

    for _ in range(3):
        progress = runner.update()

        assert progress.status == TaskStatus.SUCCEEDED
        assert progress.step_count == step_count
        assert progress.completed_skills == completed
        assert executor.start_calls == []

    runner.cancel()
    assert progress.status == TaskStatus.SUCCEEDED

    print("[PASS] Terminal updates are idempotent")


def test_invalid_start_order_and_max_steps():

    for invalid_value in (0, -1, 1.5, True):

        try:
            TaskRunner(
                selector=MockSkillSelector(),
                decision_engine=SkillDecisionEngine(
                    SkillRegistry()
                ),
                executor=FakeAsyncExecutor(),
                max_steps=invalid_value,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"max_steps={invalid_value!r} should be rejected"
            )

    goal = make_goal()
    initial_state = make_start_state()
    executor = FakeAsyncExecutor()
    runner = make_runner(executor, max_steps=5)

    try:
        runner.update()
    except RuntimeError:
        pass
    else:
        raise AssertionError("update() before start() should fail")

    try:
        runner.cancel()
    except RuntimeError:
        pass
    else:
        raise AssertionError("cancel() before start() should fail")

    runner.start(goal, initial_state)

    try:
        runner.start(goal, initial_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("start() twice should fail")

    assert executor.start_calls == []

    print("[PASS] Invalid max_steps and start order rejected")


def main():

    print("\n========== Async TaskRunner Tests ==========\n")

    test_success_path()
    test_pending_update_does_not_redispatch()
    test_arguments_passthrough()
    test_executor_failure()
    test_cancel_while_pending()
    test_cancel_then_success_result()
    test_terminal_decisions_without_executor()
    test_max_steps_stops_before_ordinary_skill()
    test_max_steps_allows_terminal_decisions()
    test_cancel_before_dispatch()
    test_terminal_update_is_idempotent()
    test_invalid_start_order_and_max_steps()

    print("\n========== All Tests Passed ==========\n")


if __name__ == "__main__":
    main()
