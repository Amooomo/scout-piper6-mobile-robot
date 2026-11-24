import time
from dataclasses import dataclass, replace

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from .bt_skill_executor import BtSkillExecutor
from .mock_skill_selector import MockSkillSelector
from .navigate_bt_execution import NavigateBtExecutionFactory
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


MOCK_NAVIGATION_SKILL_COMPLETED = (
    "MOCK_NAVIGATION_SKILL_COMPLETED"
)


@dataclass(frozen=True)
class MockNavigationRunResult:
    status: str
    message: str
    final_state: RobotState
    completed_skills: tuple
    events: tuple


class MockNavigationRunController:
    """Pure-Python gate for one MOCK_ONLY navigation Skill.

    It never selects the next Skill after the first navigation result.
    """

    def __init__(
        self,
        runner,
        goal,
        initial_state,
        server_ready,
        clock=time.monotonic,
        startup_timeout_sec=10.0,
        execution_timeout_sec=30.0,
        cancel_grace_sec=3.0,
    ):
        self._runner = runner
        self._goal = goal
        self._initial_state = initial_state
        self._server_ready = server_ready
        self._clock = clock
        self._startup_timeout_sec = startup_timeout_sec
        self._execution_timeout_sec = execution_timeout_sec
        self._cancel_grace_sec = cancel_grace_sec

        self._reference_time = None
        self._runner_started = False
        self._execution_started = None
        self._cancel_started = None
        self._finished = False
        self._result = None
        self._events = []

    @property
    def finished(self):
        return self._finished

    @property
    def result(self):
        return self._result

    @property
    def events(self):
        return tuple(self._events)

    def tick(self, now=None):
        if self._finished:
            return self._result

        if now is None:
            now = self._clock()

        if self._reference_time is None:
            self._reference_time = now

        if not self._runner_started:
            return self._start_when_ready(now)

        return self._advance_running(now)

    def _start_when_ready(self, now):
        try:
            ready = bool(self._server_ready())
        except Exception as exc:
            return self._finish(
                "SERVER_CHECK_ERROR",
                f"Action server readiness check failed: {exc}",
            )

        if ready:
            try:
                self._runner.start(
                    self._goal,
                    self._initial_state,
                )
            except Exception as exc:
                return self._finish(
                    "START_FAILED",
                    f"TaskRunner.start failed: {exc}",
                )

            self._runner_started = True
            self._execution_started = now
            self._events.append(
                "MOCK_NAVIGATION_RUNNER_STARTED"
            )
            return None

        if (
            now - self._reference_time
            >= self._startup_timeout_sec
        ):
            return self._finish(
                "SERVER_TIMEOUT",
                "NavigateToRegion Action Server not ready",
            )

        return None

    def _advance_running(self, now):
        if (
            self._cancel_started is None
            and now - self._execution_started
            >= self._execution_timeout_sec
        ):
            self._cancel_started = now
            self._events.append(
                "MOCK_NAVIGATION_TIMEOUT_CANCEL_REQUESTED"
            )

            try:
                self._runner.cancel()
            except Exception as exc:
                self._events.append(
                    f"CANCEL_REQUEST_ERROR: {exc}"
                )

        if (
            self._cancel_started is not None
            and now - self._cancel_started
            >= self._cancel_grace_sec
        ):
            return self._finish(
                "INCONCLUSIVE",
                "cancel requested but terminal result "
                "was not confirmed",
            )

        try:
            progress = self._runner.update()
        except Exception as exc:
            return self._finish(
                "RUNNER_ERROR",
                f"TaskRunner.update failed: {exc}",
            )

        return self._inspect_progress(progress)

    def _inspect_progress(self, progress):
        completed = list(progress.completed_skills)

        if completed:
            if len(completed) != 1:
                return self._finish(
                    "FAIL_CLOSED",
                    "unexpected completed Skill count: "
                    f"{completed}",
                )

            if completed[0] != SkillID.NAVIGATE_REGION:
                return self._finish(
                    "FAIL_CLOSED",
                    "unexpected completed Skill: "
                    f"{completed[0]}",
                )

            state = self._runner.current_state

            if state is None or (
                state.current_region_id
                != self._goal.target_region_id
            ):
                return self._finish(
                    "MOCK_ONLY_STATE_MISMATCH",
                    "MOCK_ONLY confirmation did not produce "
                    "the target region",
                )

            return self._finish(
                MOCK_NAVIGATION_SKILL_COMPLETED,
                "one MOCK_ONLY NAVIGATE_REGION Skill completed",
            )

        if progress.status == TaskStatus.NEEDS_HUMAN:
            return self._finish(
                "NEEDS_HUMAN",
                progress.last_message,
            )

        if progress.status == TaskStatus.FAILED:
            return self._finish(
                "FAILED",
                progress.last_message,
            )

        if progress.status == TaskStatus.SUCCEEDED:
            return self._finish(
                "FAIL_CLOSED",
                "task-level SUCCEEDED before one nav result",
            )

        if (
            progress.current_skill is not None
            and progress.current_skill != SkillID.NAVIGATE_REGION
        ):
            return self._finish(
                "FAIL_CLOSED",
                "selector chose unexpected Skill: "
                f"{progress.current_skill}",
            )

        return None

    def _finish(self, status, message):
        if self._finished:
            return self._result

        self._finished = True
        self._events.append(f"MOCK_NAVIGATION_FINISHED: {status}")

        state = self._runner.current_state
        if state is None:
            state = self._initial_state

        completed = []
        progress = getattr(self._runner, "progress", None)
        if progress is not None:
            completed = list(progress.completed_skills)

        self._result = MockNavigationRunResult(
            status=status,
            message=message,
            final_state=state,
            completed_skills=tuple(completed),
            events=tuple(self._events),
        )
        return self._result


class RosMockNavigationRunner(Node):

    def __init__(
        self,
        action_type,
        goal_status,
    ):
        super().__init__('ros_mock_navigation_runner')

        self.declare_parameter(
            'target_region_id',
            'experiment_table_A',
        )
        self.declare_parameter(
            'action_name',
            '/robot_nav/navigate_to_region',
        )
        self.declare_parameter(
            'tick_period_sec',
            0.1,
        )
        self.declare_parameter(
            'startup_timeout_sec',
            10.0,
        )
        self.declare_parameter(
            'execution_timeout_sec',
            30.0,
        )
        self.declare_parameter(
            'cancel_grace_sec',
            3.0,
        )

        self._target_region_id = self.get_parameter(
            'target_region_id'
        ).value
        action_name = self.get_parameter(
            'action_name'
        ).value
        tick_period = float(self.get_parameter(
            'tick_period_sec'
        ).value)
        startup_timeout = float(self.get_parameter(
            'startup_timeout_sec'
        ).value)
        execution_timeout = float(self.get_parameter(
            'execution_timeout_sec'
        ).value)
        cancel_grace = float(self.get_parameter(
            'cancel_grace_sec'
        ).value)

        self._action_type = action_type
        self._action_client = ActionClient(
            self,
            action_type,
            action_name,
        )

        self._goal = TaskGoal(
            instruction=(
                "Navigate to "
                f"{self._target_region_id}"
            ),
            target_region_id=self._target_region_id,
        )
        self._initial_state = RobotState(
            current_region_id="start",
            docking_complete=False,
            object_detected=False,
            object_grasped=False,
            result_verified=False,
        )

        factory = NavigateBtExecutionFactory(
            action_client_factory=lambda: self._action_client,
            goal_builder=self._build_goal,
            goal_status=goal_status,
            confirm_state=self._confirm_mock_state,
        )

        runner = TaskRunner(
            selector=MockSkillSelector(),
            decision_engine=SkillDecisionEngine(
                SkillRegistry()
            ),
            executor=BtSkillExecutor(factory),
            max_steps=10,
        )

        self._controller = MockNavigationRunController(
            runner=runner,
            goal=self._goal,
            initial_state=self._initial_state,
            server_ready=self._action_client.server_is_ready,
            startup_timeout_sec=startup_timeout,
            execution_timeout_sec=execution_timeout,
            cancel_grace_sec=cancel_grace,
        )

        self._run_result = None
        self._timer = self.create_timer(
            tick_period,
            self._tick,
        )

        self.get_logger().warn(
            "MOCK_ONLY: one NAVIGATE_REGION Skill; "
            "state confirmation is simulated and not "
            "localization evidence"
        )
        self.get_logger().info(
            f"Waiting for {action_name}; target="
            f"{self._target_region_id}"
        )

    @property
    def finished(self):
        return self._run_result is not None

    @property
    def run_result(self):
        return self._run_result

    def _build_goal(self, region_id):
        goal = self._action_type.Goal()
        goal.region_id = region_id
        return goal

    def _confirm_mock_state(
        self,
        candidate,
        state_before,
        outcome,
    ):
        if outcome.region_id != self._target_region_id:
            self.get_logger().error(
                "MOCK_ONLY confirmation rejected: "
                f"{outcome.region_id} != "
                f"{self._target_region_id}"
            )
            return None

        self.get_logger().warn(
            "MOCK_ONLY: accepting simulated navigation "
            f"state for {outcome.region_id}; this is not "
            "real localization evidence"
        )
        return replace(
            state_before,
            current_region_id=outcome.region_id,
        )

    def _tick(self):
        if self._run_result is not None:
            return

        result = self._controller.tick()

        if result is None:
            return

        self._run_result = result
        self._timer.cancel()

        log = self.get_logger()

        if result.status == MOCK_NAVIGATION_SKILL_COMPLETED:
            log.info(
                f"{result.status}: {result.message}; "
                f"completed={result.completed_skills}; "
                f"state={result.final_state}"
            )
        else:
            log.error(
                f"{result.status}: {result.message}; "
                f"completed={result.completed_skills}; "
                f"state={result.final_state}"
            )

        log.info("MOCK_ONLY run events: " + " | ".join(result.events))


def main(args=None):
    rclpy.init(args=args)

    from action_msgs.msg import GoalStatus
    from robot_interfaces.action import NavigateToRegion

    node = RosMockNavigationRunner(
        action_type=NavigateToRegion,
        goal_status=GoalStatus,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)

    exit_code = 1

    try:
        while rclpy.ok() and not node.finished:
            executor.spin_once(timeout_sec=0.1)

        if (
            node.run_result is not None
            and node.run_result.status
            == MOCK_NAVIGATION_SKILL_COMPLETED
        ):
            exit_code = 0

    except KeyboardInterrupt:
        exit_code = 130

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
