import unittest

import py_trees

from .bt_skill_executor import (
    BtSkillExecution,
    BtSkillExecutor,
)
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


class ResultHolder:

    def __init__(self, result=None):
        self.result = result

    def get(self):
        return self.result

    def set(self, result):
        self.result = result


class ScriptedSkillNode(py_trees.behaviour.Behaviour):

    def __init__(
        self,
        candidate,
        state,
        script,
        raise_on_tick=False,
    ):
        super().__init__(name="ScriptedSkill")

        self.candidate = candidate
        self.state = state
        self.script = list(script)
        self.index = 0
        self.raise_on_tick = raise_on_tick

        self.ticks = 0
        self.cancel_requests = 0
        self.terminate_calls = []

    def update(self):
        self.ticks += 1

        if self.raise_on_tick:
            raise RuntimeError("scripted tick failure")

        if self.index < len(self.script):
            status = self.script[self.index]
            self.index += 1
            return status

        if self.script:
            return self.script[-1]

        return py_trees.common.Status.RUNNING

    def request_cancel(self):
        self.cancel_requests += 1

    def terminate(self, new_status):
        self.terminate_calls.append(new_status)
        super().terminate(new_status)


class ScriptedBtFactory:

    def __init__(
        self,
        script,
        result=None,
        raise_on_tick=False,
    ):
        self.script = list(script)
        self.result = result
        self.raise_on_tick = raise_on_tick

        self.calls = []
        self.nodes = []
        self.holders = []

    def __call__(self, candidate, state):
        self.calls.append((candidate, state))

        node = ScriptedSkillNode(
            candidate,
            state,
            self.script,
            raise_on_tick=self.raise_on_tick,
        )

        root = py_trees.composites.Sequence(
            name="SingleSkillExecution",
            memory=True,
            children=[node],
        )

        holder = ResultHolder(self.result)
        self.nodes.append(node)
        self.holders.append(holder)

        return BtSkillExecution(
            root=root,
            result_provider=holder.get,
            cancel_request=node.request_cancel,
        )


class RaisingFactory:

    def __init__(self):
        self.calls = []

    def __call__(self, candidate, state):
        self.calls.append((candidate, state))
        raise RuntimeError("factory failure")


class ScriptedSelector:

    def __init__(self, candidates=None):
        self.candidates = list(candidates or [])
        self.calls = 0

    def score_candidates(self, goal, state):
        self.calls += 1
        return list(self.candidates)


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


def make_goal():
    return TaskGoal(
        instruction="Navigate to the target region",
        target_region_id="experiment_table_A",
        target_object_id="target_object",
    )


def make_nav_candidate():
    return SkillCandidate(
        skill_id=SkillID.NAVIGATE_REGION,
        score=0.90,
        arguments={"region_id": "experiment_table_A"},
    )


def make_done_candidate():
    return SkillCandidate(
        skill_id=SkillID.DONE,
        score=0.99,
        arguments={},
    )


def make_ask_candidate():
    return SkillCandidate(
        skill_id=SkillID.ASK_HUMAN,
        score=0.01,
        arguments={},
    )


def make_start_state():
    return RobotState(
        current_region_id="start",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )


def make_verified_state():
    return RobotState(
        current_region_id="experiment_table_A",
        docking_complete=True,
        object_detected=True,
        object_grasped=True,
        result_verified=True,
    )


def make_arrived_state():
    return RobotState(
        current_region_id="experiment_table_A",
        docking_complete=False,
        object_detected=False,
        object_grasped=False,
        result_verified=False,
    )


def make_success_result(message="Fake navigation success"):
    return SkillExecutionResult(
        success=True,
        message=message,
        state_after=make_arrived_state(),
    )


def make_components(decisions, factory, max_steps=10):
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


class BtSkillExecutorTest(unittest.TestCase):

    def test_b01_external_runner_dispatches_once(self):
        factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
        )
        runner, selector, engine, _ = make_components(
            [make_nav_candidate()],
            factory,
        )

        runner.start(make_goal(), make_start_state())
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(len(factory.calls), 1)
        self.assertEqual(len(factory.nodes), 1)

    def test_b02_candidate_passthrough_and_copy(self):
        candidate = make_nav_candidate()
        factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
        )
        runner, _, _, _ = make_components([candidate], factory)
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()

        received_candidate, received_state = factory.calls[0]

        self.assertIsNot(received_candidate, candidate)
        self.assertIsNot(
            received_candidate.arguments,
            candidate.arguments,
        )
        self.assertEqual(
            received_candidate.skill_id,
            SkillID.NAVIGATE_REGION,
        )
        self.assertEqual(
            received_candidate.arguments,
            {"region_id": "experiment_table_A"},
        )
        self.assertEqual(received_state, initial_state)
        self.assertEqual(
            candidate.arguments,
            {"region_id": "experiment_table_A"},
        )

    def test_b03_multiple_running_ticks(self):
        factory = ScriptedBtFactory(
            script=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.SUCCESS,
            ],
            result=make_success_result(),
        )
        runner, selector, engine, _ = make_components(
            [make_nav_candidate()],
            factory,
        )

        runner.start(make_goal(), make_start_state())
        progress = runner.update()

        for _ in range(3):
            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.RUNNING)
            self.assertEqual(selector.calls, 1)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(len(factory.calls), 1)
            self.assertEqual(runner.current_state, make_start_state())
            self.assertEqual(progress.completed_skills, [])

        self.assertEqual(factory.nodes[0].ticks, 3)

    def test_b04_success_only_commits_after_terminal(self):
        factory = ScriptedBtFactory(
            script=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.SUCCESS,
            ],
            result=make_success_result(),
        )
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            factory,
        )

        runner.start(make_goal(), make_start_state())
        runner.update()

        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(runner.progress.completed_skills, [])

        progress = runner.update()
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(runner.current_state, make_start_state())
        self.assertEqual(progress.completed_skills, [])

        progress = runner.update()
        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(runner.current_state, make_arrived_state())
        self.assertEqual(
            progress.completed_skills,
            [SkillID.NAVIGATE_REGION],
        )
        self.assertEqual(factory.nodes[0].ticks, 2)

    def test_b05_terminal_update_does_not_decide_again(self):
        factory = ScriptedBtFactory(
            script=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.SUCCESS,
            ],
            result=make_success_result(),
        )
        runner, selector, engine, _ = make_components(
            [make_nav_candidate(), make_ask_candidate()],
            factory,
        )

        runner.start(make_goal(), make_start_state())
        runner.update()
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(
            progress.completed_skills,
            [SkillID.NAVIGATE_REGION],
        )

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.NEEDS_HUMAN)
        self.assertEqual(selector.calls, 2)
        self.assertEqual(engine.calls, 2)
        self.assertEqual(len(factory.calls), 1)

    def test_b06_bt_failure_does_not_commit_state(self):
        failed_result = SkillExecutionResult(
            success=False,
            message="Fake navigation failure",
            state_after=make_verified_state(),
        )
        factory = ScriptedBtFactory(
            script=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.FAILURE,
            ],
            result=failed_result,
        )
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            factory,
        )
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(runner.current_state, initial_state)

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])
        self.assertEqual(
            runner.execution_results,
            (failed_result,),
        )

    def test_b07_success_without_result_evidence_fails_safely(self):
        factory = ScriptedBtFactory(
            script=[
                py_trees.common.Status.RUNNING,
                py_trees.common.Status.SUCCESS,
            ],
            result=None,
        )
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            factory,
        )
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])
        self.assertIn(
            "without explicit result evidence",
            progress.last_message,
        )

    def test_b08_cancel_requests_once_and_stays_pending(self):
        factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
        )
        runner, selector, engine, _ = make_components(
            [make_nav_candidate()],
            factory,
        )
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()

        node = factory.nodes[0]
        runner.cancel()

        self.assertEqual(node.cancel_requests, 1)

        for _ in range(3):
            progress = runner.update()

            self.assertEqual(progress.status, TaskStatus.RUNNING)
            self.assertEqual(selector.calls, 1)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(len(factory.calls), 1)
            self.assertEqual(runner.current_state, initial_state)
            self.assertEqual(progress.completed_skills, [])

        runner.cancel()
        self.assertEqual(node.cancel_requests, 1)
        self.assertEqual(node.ticks, 0)

    def test_b09_late_success_after_cancel_not_committed(self):
        factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
        )
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            factory,
        )
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()
        runner.cancel()

        factory.holders[0].set(make_success_result())
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])
        self.assertEqual(len(runner.execution_results), 1)
        self.assertTrue(runner.execution_results[0].success)
        self.assertIn("cancel_requested", progress.last_message)

    def test_b10_repeated_poll_and_cancel_are_idempotent(self):
        factory = ScriptedBtFactory(
            script=[py_trees.common.Status.SUCCESS],
            result=make_success_result(),
        )
        executor = BtSkillExecutor(factory)
        handle = executor.start(
            make_nav_candidate(),
            make_start_state(),
        )

        first = handle.poll()
        second = handle.poll()

        self.assertIs(first, second)
        self.assertTrue(first.success)
        self.assertEqual(factory.nodes[0].ticks, 1)

        handle.cancel()
        handle.cancel()
        self.assertEqual(factory.nodes[0].cancel_requests, 0)
        self.assertEqual(factory.nodes[0].ticks, 1)

        pending_factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
        )
        pending_executor = BtSkillExecutor(pending_factory)
        pending_handle = pending_executor.start(
            make_nav_candidate(),
            make_start_state(),
        )

        pending_handle.cancel()
        pending_handle.cancel()

        self.assertEqual(
            pending_factory.nodes[0].cancel_requests,
            1,
        )

    def test_b11_done_and_ask_do_not_use_bt(self):
        done_factory = ScriptedBtFactory(
            script=[py_trees.common.Status.SUCCESS],
            result=make_success_result(),
        )
        done_runner, _, _, _ = make_components(
            [make_done_candidate()],
            done_factory,
        )

        done_runner.start(make_goal(), make_verified_state())
        progress = done_runner.update()

        self.assertEqual(progress.status, TaskStatus.SUCCEEDED)
        self.assertEqual(done_factory.calls, [])

        ask_factory = ScriptedBtFactory(
            script=[py_trees.common.Status.SUCCESS],
            result=make_success_result(),
        )
        ask_runner, _, _, _ = make_components(
            [make_ask_candidate()],
            ask_factory,
        )

        ask_runner.start(make_goal(), make_start_state())
        progress = ask_runner.update()

        self.assertEqual(progress.status, TaskStatus.NEEDS_HUMAN)
        self.assertEqual(ask_factory.calls, [])

    def test_b12_bt_errors_and_invalid_status_do_not_fake_success(self):
        # factory 抛异常
        factory = RaisingFactory()
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            factory,
        )
        initial_state = make_start_state()

        runner.start(make_goal(), initial_state)
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])
        self.assertEqual(len(factory.calls), 1)

        # tick 抛异常
        raising_factory = ScriptedBtFactory(
            script=[py_trees.common.Status.RUNNING],
            result=None,
            raise_on_tick=True,
        )
        runner, _, _, _ = make_components(
            [make_nav_candidate()],
            raising_factory,
        )

        runner.start(make_goal(), initial_state)
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.FAILED)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])
        self.assertIn("BT tick failed", progress.last_message)

        # INVALID 不等于取消终态，也不产生伪成功
        invalid_factory = ScriptedBtFactory(
            script=[py_trees.common.Status.INVALID],
            result=None,
        )
        runner, selector, engine, _ = make_components(
            [make_nav_candidate()],
            invalid_factory,
        )

        runner.start(make_goal(), initial_state)
        runner.update()
        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(selector.calls, 1)
        self.assertEqual(engine.calls, 1)
        self.assertEqual(runner.current_state, initial_state)
        self.assertEqual(progress.completed_skills, [])


if __name__ == "__main__":
    unittest.main()
