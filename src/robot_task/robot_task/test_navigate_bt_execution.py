import unittest
from types import SimpleNamespace

from .bt_skill_executor import BtSkillExecutor
from .navigate_bt_execution import (
    NavigateBtExecutionFactory,
)
from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillDecision,
    SkillID,
    TaskGoal,
    TaskStatus,
)
from .task_runner import TaskRunner

from .test_navigate_region_action_adapter import (
    FakeActionClient,
    FakeGoalHandle,
    FakeGoalStatus,
    fake_goal_builder,
    make_feedback,
    make_result_response,
)


class ScriptedSelector:

    def __init__(self):
        self.calls = 0

    def score_candidates(self, goal, state):
        self.calls += 1
        return []


class ScriptedDecisionEngine:

    def __init__(self, selected):
        self.selected = list(selected)
        self.calls = 0

    def decide(self, candidates, goal, state):
        self.calls += 1

        selected = (
            self.selected.pop(0)
            if self.selected
            else None
        )

        return SkillDecision(selected=selected)


class RecordingFactory:

    def __init__(self, inner):
        self.inner = inner
        self.executions = []

    def __call__(self, candidate, state):
        execution = self.inner(candidate, state)
        self.executions.append(execution)
        return execution


class CountingConfirm:

    def __init__(self, result_state):
        self.result_state = result_state
        self.calls = 0

    def __call__(self, candidate, state_before, outcome):
        self.calls += 1
        return self.result_state


def make_goal():
    return TaskGoal(
        instruction="Navigate to the target region",
        target_region_id="experiment_table_A",
        target_object_id="target_object",
    )


def make_nav_candidate(region_id="experiment_table_A"):
    return SkillCandidate(
        skill_id=SkillID.NAVIGATE_REGION,
        score=0.90,
        arguments={"region_id": region_id},
    )


def make_start_state():
    return RobotState(
        current_region_id="start",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )


def make_arrived_state():
    return RobotState(
        current_region_id="experiment_table_A",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )


def make_factory(client, confirm_state=None):
    return NavigateBtExecutionFactory(
        action_client_factory=lambda: client,
        goal_builder=fake_goal_builder,
        goal_status=FakeGoalStatus,
        confirm_state=confirm_state,
    )


def make_runner(decisions, factory, max_steps=10):
    selector = ScriptedSelector()
    engine = ScriptedDecisionEngine(decisions)
    executor = BtSkillExecutor(factory)

    runner = TaskRunner(
        selector=selector,
        decision_engine=engine,
        executor=executor,
        max_steps=max_steps,
    )

    return runner, selector, engine, executor


def start_runner(runner):
    runner.start(make_goal(), make_start_state())


def dispatch_and_send_goal(runner):
    runner.update()
    return runner.update()


class NavigateBtExecutionTest(unittest.TestCase):

    def test_d01_candidate_to_fake_goal(self):
        client = FakeActionClient()
        factory = RecordingFactory(make_factory(client))
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        runner.update()

        self.assertEqual(client.send_calls, [])
        self.assertEqual(len(factory.executions), 1)

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(len(client.send_calls), 1)
        self.assertEqual(
            client.send_calls[0].region_id,
            "experiment_table_A",
        )
        self.assertFalse(hasattr(client.send_calls[0], "score"))
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])

    def test_d02_pending_ticks_do_not_redispatch(self):
        client = FakeActionClient()
        factory = RecordingFactory(make_factory(client))
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        leaf = factory.executions[0].root
        self.assertEqual(leaf.ticks, 1)

        for _ in range(3):
            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.RUNNING)
            self.assertEqual(selector.calls, 1)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(len(factory.executions), 1)
            self.assertEqual(len(client.send_calls), 1)
            self.assertEqual(runner.current_state, make_start_state())
            self.assertEqual(progress.completed_skills, [])

        self.assertEqual(leaf.ticks, 4)

    def test_d03_accepted_feedback_are_not_terminal(self):
        client = FakeActionClient()
        factory = RecordingFactory(make_factory(client))
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        self.assertEqual(runner.progress.status, TaskStatus.RUNNING)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(
            runner.progress.completed_skills,
            [],
        )

        client.feedback_callbacks[0](
            make_feedback("NAVIGATING", 1.5)
        )
        client.feedback_callbacks[0](
            make_feedback("NAVIGATING", 0.5)
        )

        progress = runner.update()

        leaf = factory.executions[0].root
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])
        self.assertEqual(leaf.ticks, 2)
        self.assertEqual(leaf.last_feedback_phase, "NAVIGATING")
        self.assertEqual(leaf.last_distance_remaining, 0.5)
        self.assertEqual(handle.get_result_calls, 1)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(len(factory.executions), 1)
        self.assertEqual(len(client.send_calls), 1)

    def test_d04_success_requires_confirmation_and_commits_once(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = RecordingFactory(make_factory(client, confirm))
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate(), SkillCandidate(
                skill_id=SkillID.ASK_HUMAN,
                score=0.01,
                arguments={},
            )],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(confirm.calls, 1)
        self.assertEqual(
            runner.current_state,
            make_arrived_state(),
        )
        self.assertEqual(
            progress.completed_skills,
            [SkillID.NAVIGATE_REGION],
        )
        self.assertEqual(factory.executions[0].root.ticks, 2)
        self.assertEqual(len(client.send_calls), 1)

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.NEEDS_HUMAN)
        self.assertEqual(selector.calls, 2)
        self.assertEqual(engine.calls, 2)
        self.assertEqual(len(factory.executions), 1)

    def test_d05_success_without_confirm_provider_fails(self):
        client = FakeActionClient()
        factory = make_factory(client, confirm_state=None)
        runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])
        self.assertIn(
            "no state confirmation provider",
            progress.last_message,
        )

    def test_d05b_confirm_returns_none_fails_safely(self):
        client = FakeActionClient()
        confirm = CountingConfirm(None)
        factory = make_factory(client, confirm)
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(confirm.calls, 1)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])
        self.assertIn(
            "valid RobotState",
            progress.last_message,
        )
        self.assertEqual(len(client.send_calls), 1)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)

        # 终态后 update 幂等，不产生下一次决策
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(confirm.calls, 1)
        self.assertEqual(len(client.send_calls), 1)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)

    def test_d06_result_false_does_not_call_confirm(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = make_factory(client, confirm)
        runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=False,
                message="Result false",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(confirm.calls, 0)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])

    def test_d07_rejected_aborted_canceled_unknown(self):
        def run_case(status_setup, expected_message):
            client = FakeActionClient()
            confirm = CountingConfirm(make_arrived_state())
            factory = make_factory(client, confirm)
            runner, _, _, _ = make_runner(
                [make_nav_candidate()],
                factory,
            )

            start_runner(runner)
            dispatch_and_send_goal(runner)

            status_setup(client)

            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.FAILED)
            self.assertEqual(confirm.calls, 0)
            self.assertEqual(
                runner.current_state,
                make_start_state(),
            )
            self.assertEqual(progress.completed_skills, [])
            self.assertIn(
                expected_message,
                progress.last_message,
            )

        def rejected(client):
            client.goal_futures[0].set_result(
                FakeGoalHandle(accepted=False)
            )

        def accepted_with_result(status, message, include_success=True):
            def setup(client):
                handle = FakeGoalHandle(accepted=True)
                client.goal_futures[0].set_result(handle)
                handle.result_future.set_result(
                    make_result_response(
                        status,
                        success=False,
                        message=message,
                        include_success=include_success,
                    )
                )
            return setup

        run_case(rejected, "rejected")
        run_case(
            accepted_with_result(
                FakeGoalStatus.STATUS_ABORTED,
                "aborted",
            ),
            "aborted",
        )
        run_case(
            accepted_with_result(
                FakeGoalStatus.STATUS_CANCELED,
                "canceled",
            ),
            "canceled",
        )
        run_case(
            accepted_with_result(
                FakeGoalStatus.STATUS_SUCCEEDED,
                "missing",
                include_success=False,
            ),
            "UNKNOWN",
        )

    def test_d08_cancel_after_start_waits_then_exits(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = RecordingFactory(make_factory(client, confirm))
        runner, selector, engine, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        leaf = factory.executions[0].root
        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        runner.cancel()

        self.assertEqual(handle.cancel_calls, 1)
        self.assertEqual(leaf.ticks, 1)

        progress = runner.update()
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(leaf.ticks, 1)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)

        handle.cancel_future.set_result(
            SimpleNamespace(goals_canceling=True)
        )
        progress = runner.update()
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(leaf.ticks, 1)

        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_CANCELED,
                success=False,
                message="canceled",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(confirm.calls, 0)
        self.assertEqual(leaf.ticks, 1)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])

    def test_d09_cancel_before_first_tick_sends_no_goal(self):
        client = FakeActionClient()
        factory = RecordingFactory(make_factory(client))
        runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        runner.update()

        leaf = factory.executions[0].root
        self.assertEqual(leaf.ticks, 0)
        self.assertEqual(client.send_calls, [])

        runner.cancel()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(client.send_calls, [])
        self.assertEqual(leaf.ticks, 0)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertIn(
            "Cancelled before navigation goal dispatch",
            progress.last_message,
        )

    def test_d10_late_success_after_cancel_not_committed(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = make_factory(client, confirm)
        runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        runner.cancel()

        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Late success",
            )
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(confirm.calls, 0)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])
        self.assertIn("cancel_requested", progress.last_message)
        self.assertIn("not committed", progress.last_message)

    def test_d11_terminal_poll_and_cancel_idempotent(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = make_factory(client, confirm)
        runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(runner)
        dispatch_and_send_goal(runner)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        progress = runner.update()
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(len(client.send_calls), 1)

        direct_client = FakeActionClient()
        direct_confirm = CountingConfirm(make_arrived_state())
        direct_factory = RecordingFactory(
            make_factory(direct_client, direct_confirm)
        )
        executor = BtSkillExecutor(direct_factory)
        direct_handle = executor.start(
            make_nav_candidate(),
            make_start_state(),
        )

        self.assertIsNone(direct_handle.poll())
        direct_handle_goal = FakeGoalHandle(accepted=True)
        direct_client.goal_futures[0].set_result(
            direct_handle_goal
        )
        direct_handle_goal.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        first = direct_handle.poll()
        second = direct_handle.poll()
        direct_leaf = direct_factory.executions[0].root

        self.assertIs(first, second)
        self.assertTrue(first.success)
        self.assertEqual(len(direct_client.send_calls), 1)
        self.assertEqual(direct_leaf.ticks, 2)
        self.assertEqual(direct_confirm.calls, 1)

        direct_handle.cancel()
        direct_handle.cancel()
        self.assertIs(first, direct_handle.poll())
        self.assertEqual(direct_leaf.ticks, 2)
        self.assertEqual(len(direct_client.send_calls), 1)
        self.assertEqual(direct_confirm.calls, 1)

    def test_d12_invalid_skill_or_region_sends_no_goal(self):
        invalid_candidates = [
            SkillCandidate(
                skill_id=SkillID.DETECT_OBJECT,
                score=0.50,
                arguments={"object_id": "target_object"},
            ),
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.50,
                arguments={},
            ),
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.50,
                arguments={"region_id": ""},
            ),
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.50,
                arguments={"region_id": "   "},
            ),
        ]

        for candidate in invalid_candidates:
            client = FakeActionClient()
            factory = make_factory(client)
            runner, _, _, _ = make_runner([candidate], factory)

            start_runner(runner)
            runner.update()
            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.FAILED)
            self.assertEqual(client.send_calls, [])
            self.assertEqual(
                runner.current_state,
                make_start_state(),
            )
            self.assertEqual(progress.completed_skills, [])

    def test_d13_confirm_errors_are_conservative(self):
        def raising_confirm(candidate, state_before, outcome):
            raise RuntimeError("confirm boom")

        def invalid_confirm(candidate, state_before, outcome):
            return "not a RobotState"

        for confirm in (raising_confirm, invalid_confirm):
            client = FakeActionClient()
            factory = make_factory(client, confirm)
            runner, _, _, _ = make_runner(
                [make_nav_candidate()],
                factory,
            )

            start_runner(runner)
            dispatch_and_send_goal(runner)

            handle = FakeGoalHandle(accepted=True)
            client.goal_futures[0].set_result(handle)
            handle.result_future.set_result(
                make_result_response(
                    FakeGoalStatus.STATUS_SUCCEEDED,
                    success=True,
                    message="Arrived",
                )
            )

            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.FAILED)
            self.assertEqual(
                runner.current_state,
                make_start_state(),
            )
            self.assertEqual(progress.completed_skills, [])
            self.assertIn(
                "State confirmation",
                progress.last_message,
            )

    def test_d14_old_session_does_not_pollute_new_runner(self):
        client = FakeActionClient()
        confirm = CountingConfirm(make_arrived_state())
        factory = make_factory(client, confirm)

        old_runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )
        new_runner, _, _, _ = make_runner(
            [make_nav_candidate()],
            factory,
        )

        start_runner(old_runner)
        dispatch_and_send_goal(old_runner)

        start_runner(new_runner)
        dispatch_and_send_goal(new_runner)

        self.assertEqual(len(client.send_calls), 2)

        old_handle = FakeGoalHandle(accepted=True)
        new_handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(old_handle)
        client.goal_futures[1].set_result(new_handle)

        client.feedback_callbacks[0](
            make_feedback("OLD", 9.0)
        )

        old_handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="old result",
            )
        )

        old_progress = old_runner.update()
        self.assertEqual(old_progress.status, TaskStatus.RUNNING)
        self.assertEqual(
            old_runner.current_state,
            make_arrived_state(),
        )
        self.assertEqual(confirm.calls, 1)

        new_progress = new_runner.update()
        self.assertEqual(new_progress.status, TaskStatus.RUNNING)
        self.assertEqual(
            new_runner.current_state,
            make_start_state(),
        )
        self.assertEqual(new_progress.completed_skills, [])

        new_handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_ABORTED,
                success=False,
                message="new result",
            )
        )

        new_progress = new_runner.update()
        self.assertEqual(new_progress.status, TaskStatus.FAILED)
        self.assertEqual(
            new_runner.current_state,
            make_start_state(),
        )
        self.assertIn("new result", new_progress.last_message)
        self.assertEqual(confirm.calls, 1)


if __name__ == "__main__":
    unittest.main()
