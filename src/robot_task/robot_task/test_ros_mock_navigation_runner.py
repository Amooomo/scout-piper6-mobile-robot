import unittest

from .bt_skill_executor import BtSkillExecutor
from .mock_skill_selector import MockSkillSelector
from .navigate_bt_execution import NavigateBtExecutionFactory
from .ros_mock_navigation_runner import (
    MOCK_NAVIGATION_SKILL_COMPLETED,
    MockNavigationRunController,
)
from .skill_decision_engine import SkillDecisionEngine
from .skill_registry import SkillRegistry
from .skill_types import (
    RobotState,
    SkillID,
    TaskGoal,
    TaskProgress,
    TaskStatus,
)
from .task_runner import TaskRunner

from .test_navigate_region_action_adapter import (
    FakeActionClient,
    FakeGoalStatus,
    fake_goal_builder,
)


class FakeRunner:

    def __init__(self):
        self.start_calls = 0
        self.update_calls = 0
        self.cancel_calls = 0
        self.steps = []

        self.progress = TaskProgress()
        self.current_state = make_start_state()

    def start(self, goal, initial_state):
        self.start_calls += 1
        self.goal = goal
        self.current_state = initial_state

    def update(self):
        self.update_calls += 1

        if self.steps:
            progress, state = self.steps.pop(0)
            self.progress = progress
            self.current_state = state

        return self.progress

    def cancel(self):
        self.cancel_calls += 1


def make_goal():
    return TaskGoal(
        instruction="Navigate to experiment_table_A",
        target_region_id="experiment_table_A",
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


def make_progress(
    status=TaskStatus.RUNNING,
    current_skill=SkillID.NAVIGATE_REGION,
    completed=None,
    message="",
):
    return TaskProgress(
        status=status,
        current_skill=current_skill,
        completed_skills=list(completed or []),
        last_message=message,
    )


class RosMockNavigationRunnerTest(unittest.TestCase):

    def test_e01_real_mock_selector_selects_navigate(self):
        client = FakeActionClient()
        factory = NavigateBtExecutionFactory(
            action_client_factory=lambda: client,
            goal_builder=fake_goal_builder,
            goal_status=FakeGoalStatus,
            confirm_state=None,
        )
        runner = TaskRunner(
            selector=MockSkillSelector(),
            decision_engine=SkillDecisionEngine(
                SkillRegistry()
            ),
            executor=BtSkillExecutor(factory),
            max_steps=10,
        )

        runner.start(make_goal(), make_start_state())

        progress = runner.update()

        self.assertEqual(
            progress.current_skill,
            SkillID.NAVIGATE_REGION,
        )
        self.assertEqual(client.send_calls, [])

        progress = runner.update()

        self.assertEqual(progress.status, TaskStatus.RUNNING)
        self.assertEqual(len(client.send_calls), 1)
        self.assertEqual(
            client.send_calls[0].region_id,
            "experiment_table_A",
        )
        self.assertFalse(hasattr(client.send_calls[0], "score"))

    def test_e02_e04_controller_one_shot_completion(self):
        runner = FakeRunner()
        runner.steps = [
            (
                make_progress(current_skill=SkillID.NAVIGATE_REGION),
                make_start_state(),
            ),
            (
                make_progress(
                    current_skill=SkillID.NAVIGATE_REGION,
                    completed=[SkillID.NAVIGATE_REGION],
                ),
                make_arrived_state(),
            ),
        ]

        controller = MockNavigationRunController(
            runner=runner,
            goal=make_goal(),
            initial_state=make_start_state(),
            server_ready=lambda: True,
            startup_timeout_sec=10.0,
            execution_timeout_sec=30.0,
            cancel_grace_sec=3.0,
        )

        self.assertIsNone(controller.tick(0.0))
        self.assertEqual(runner.start_calls, 1)
        self.assertEqual(runner.update_calls, 0)

        self.assertIsNone(controller.tick(1.0))
        self.assertEqual(runner.update_calls, 1)

        result = controller.tick(2.0)

        self.assertEqual(
            result.status,
            MOCK_NAVIGATION_SKILL_COMPLETED,
        )
        self.assertEqual(
            result.completed_skills,
            (SkillID.NAVIGATE_REGION,),
        )
        self.assertEqual(runner.start_calls, 1)
        self.assertEqual(runner.update_calls, 2)

        # 终态后不再更新/重决策
        result_again = controller.tick(3.0)

        self.assertIs(result, result_again)
        self.assertEqual(runner.update_calls, 2)
        self.assertEqual(runner.start_calls, 1)
        self.assertEqual(runner.cancel_calls, 0)

    def test_e05_server_timeout_fails_closed(self):
        runner = FakeRunner()

        controller = MockNavigationRunController(
            runner=runner,
            goal=make_goal(),
            initial_state=make_start_state(),
            server_ready=lambda: False,
            startup_timeout_sec=5.0,
        )

        self.assertIsNone(controller.tick(0.0))
        result = controller.tick(5.0)

        self.assertEqual(result.status, "SERVER_TIMEOUT")
        self.assertEqual(runner.start_calls, 0)
        self.assertEqual(runner.update_calls, 0)
        self.assertEqual(result.completed_skills, ())

    def test_e06_failure_stops_without_success(self):
        runner = FakeRunner()
        runner.steps = [
            (
                make_progress(
                    status=TaskStatus.FAILED,
                    current_skill=SkillID.NAVIGATE_REGION,
                    message="mock action failed",
                ),
                make_start_state(),
            ),
        ]

        controller = MockNavigationRunController(
            runner=runner,
            goal=make_goal(),
            initial_state=make_start_state(),
            server_ready=lambda: True,
        )

        controller.tick(0.0)
        result = controller.tick(1.0)

        self.assertEqual(result.status, "FAILED")
        self.assertIn("mock action failed", result.message)
        self.assertEqual(result.completed_skills, ())
        self.assertEqual(runner.update_calls, 1)

        controller.tick(2.0)
        self.assertEqual(runner.update_calls, 1)

    def test_e08_timeout_cancel_and_inconclusive_exit(self):
        runner = FakeRunner()
        runner.steps = [
            (
                make_progress(
                    current_skill=SkillID.NAVIGATE_REGION,
                ),
                make_start_state(),
            ),
            (
                make_progress(
                    current_skill=SkillID.NAVIGATE_REGION,
                ),
                make_start_state(),
            ),
            (
                make_progress(
                    current_skill=SkillID.NAVIGATE_REGION,
                ),
                make_start_state(),
            ),
        ]

        controller = MockNavigationRunController(
            runner=runner,
            goal=make_goal(),
            initial_state=make_start_state(),
            server_ready=lambda: True,
            execution_timeout_sec=5.0,
            cancel_grace_sec=2.0,
        )

        controller.tick(0.0)
        controller.tick(6.0)

        self.assertEqual(runner.cancel_calls, 1)
        self.assertIsNone(controller.result)

        result = controller.tick(8.0)

        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertEqual(result.completed_skills, ())
        self.assertEqual(runner.cancel_calls, 1)

        controller.tick(9.0)
        self.assertEqual(runner.cancel_calls, 1)

    def test_e09_unexpected_skill_fails_closed(self):
        runner = FakeRunner()
        runner.steps = [
            (
                make_progress(
                    current_skill=SkillID.VISUAL_DOCK,
                ),
                make_start_state(),
            ),
        ]

        controller = MockNavigationRunController(
            runner=runner,
            goal=make_goal(),
            initial_state=make_start_state(),
            server_ready=lambda: True,
        )

        controller.tick(0.0)
        result = controller.tick(1.0)

        self.assertEqual(result.status, "FAIL_CLOSED")
        self.assertIn("unexpected Skill", result.message)
        self.assertEqual(result.completed_skills, ())


if __name__ == "__main__":
    unittest.main()
