from typing import Protocol

from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillExecutionResult,
    SkillID,
    TaskGoal,
    TaskProgress,
    TaskStatus,
)


class SkillExecutionHandle(Protocol):

    def poll(self) -> SkillExecutionResult | None:
        """Return None while running, or the terminal result."""

    def cancel(self) -> None:
        """Request cancellation without assuming it has completed."""


class AsyncSkillExecutor(Protocol):

    def start(
        self,
        candidate: SkillCandidate,
        state: RobotState,
    ) -> SkillExecutionHandle:
        """Start one skill and return immediately without waiting."""


class TaskRunner:

    def __init__(
        self,
        selector,
        decision_engine,
        executor,
        max_steps=10,
    ):

        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise ValueError(
                f"max_steps must be a positive integer: {max_steps}"
            )

        self.selector = selector
        self.decision_engine = decision_engine
        self.executor = executor
        self.max_steps = max_steps

        self._started = False
        self._goal = None
        self._current_state = None
        self._progress = TaskProgress()
        self._execution_results = []
        self._handle = None
        self._active_candidate = None
        self._cancel_requested = False
        self._cancel_sent = False

    @property
    def progress(self) -> TaskProgress:
        return self._progress

    @property
    def current_state(self) -> RobotState:
        return self._current_state

    @property
    def execution_results(
        self,
    ) -> tuple[SkillExecutionResult, ...]:
        return tuple(self._execution_results)

    def start(
        self,
        goal: TaskGoal,
        initial_state: RobotState,
    ) -> None:

        if self._started:
            raise RuntimeError("TaskRunner has already been started")

        self._started = True
        self._goal = goal
        self._current_state = initial_state
        self._progress = TaskProgress()
        self._execution_results = []
        self._handle = None
        self._active_candidate = None
        self._cancel_requested = False
        self._cancel_sent = False

    def update(self) -> TaskProgress:

        if not self._started:
            raise RuntimeError(
                "TaskRunner.start() must be called before update()"
            )

        if self._progress.status != TaskStatus.RUNNING:
            return self._progress

        if self._cancel_requested and self._handle is None:
            self._finish_cancelled_before_dispatch()
            return self._progress

        if self._handle is not None:
            self._update_active_handle()
            return self._progress

        self._decide_and_dispatch()
        return self._progress

    def cancel(self) -> None:

        if not self._started:
            raise RuntimeError(
                "TaskRunner.start() must be called before cancel()"
            )

        if self._progress.status != TaskStatus.RUNNING:
            return

        if self._cancel_requested:
            return

        self._cancel_requested = True

        if self._handle is None:
            self._finish_cancelled_before_dispatch()
            return

        if not self._cancel_sent:
            self._cancel_sent = True
            self._handle.cancel()

        self._progress.last_message = (
            "cancel_requested; awaiting terminal result"
        )

    def _decide_and_dispatch(self) -> None:

        candidates = self.selector.score_candidates(
            self._goal,
            self._current_state,
        )

        decision = self.decision_engine.decide(
            candidates,
            self._goal,
            self._current_state,
        )

        self._progress.step_count += 1

        selected = decision.selected

        if selected is None:
            self._progress.status = TaskStatus.FAILED
            self._progress.last_message = (
                "No feasible skill candidates"
            )
            return

        skill_id = selected.skill_id
        self._progress.current_skill = skill_id

        if skill_id == SkillID.DONE:
            if self._current_state.result_verified is True:
                self._progress.status = TaskStatus.SUCCEEDED
                self._progress.last_message = "Task completed"
            else:
                self._progress.status = TaskStatus.FAILED
                self._progress.last_message = (
                    "DONE selected but task result is not verified"
                )
            return

        if skill_id == SkillID.ASK_HUMAN:
            self._progress.status = TaskStatus.NEEDS_HUMAN
            self._progress.last_message = (
                "Human intervention requested"
            )
            return

        if self._progress.step_count >= self.max_steps:
            self._progress.status = TaskStatus.FAILED
            self._progress.last_message = (
                "Reached maximum decision steps "
                f"({self.max_steps}) without completion"
            )
            return

        self._active_candidate = selected
        self._handle = self.executor.start(
            selected,
            self._current_state,
        )

    def _update_active_handle(self) -> None:

        candidate = self._active_candidate
        result = self._handle.poll()

        if result is None:
            return

        self._handle = None
        self._active_candidate = None
        self._execution_results.append(result)

        if self._cancel_requested:
            self._progress.status = TaskStatus.FAILED
            self._progress.last_message = (
                "cancel_requested; cancel_confirmed_or_settled: "
                f"{result.message}"
            )
            return

        self._progress.last_message = result.message

        if result.success is not True:
            self._progress.status = TaskStatus.FAILED
            return

        self._current_state = result.state_after
        self._progress.completed_skills.append(
            candidate.skill_id
        )

    def _finish_cancelled_before_dispatch(self) -> None:

        self._progress.status = TaskStatus.FAILED
        self._progress.last_message = (
            "cancel_requested before dispatch; "
            "no skill was started"
        )
