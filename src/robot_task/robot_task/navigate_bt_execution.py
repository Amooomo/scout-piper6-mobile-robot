import py_trees

from .bt_skill_executor import BtSkillExecution
from .navigate_region_action_adapter import (
    NavigateRegionActionSession,
)
from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillExecutionResult,
    SkillID,
)


class NavigateRegionLeaf(py_trees.behaviour.Behaviour):

    def __init__(self, execution_state):
        super().__init__(name="NavigateRegion")

        self.execution_state = execution_state
        self.ticks = 0

    @property
    def last_feedback_phase(self):
        return self.execution_state.last_feedback_phase

    @property
    def last_distance_remaining(self):
        return self.execution_state.last_distance_remaining

    def update(self):
        self.ticks += 1

        result = self.execution_state.advance()

        if result is None:
            return py_trees.common.Status.RUNNING

        if result.success is True:
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class NavigateBtExecutionFactory:

    def __init__(
        self,
        action_client_factory,
        goal_builder,
        goal_status,
        confirm_state=None,
    ):
        self._action_client_factory = action_client_factory
        self._goal_builder = goal_builder
        self._goal_status = goal_status
        self._confirm_state = confirm_state

    def __call__(
        self,
        candidate: SkillCandidate,
        state_before: RobotState,
    ) -> BtSkillExecution:

        if candidate.skill_id != SkillID.NAVIGATE_REGION:
            raise ValueError(
                "NavigateBtExecutionFactory requires "
                "NAVIGATE_REGION"
            )

        region_id = candidate.arguments.get("region_id")

        if not isinstance(region_id, str) or not region_id.strip():
            raise ValueError(
                "Missing or invalid region_id"
            )

        action_client = self._action_client_factory()

        session = NavigateRegionActionSession(
            action_client=action_client,
            candidate=candidate,
            goal_builder=self._goal_builder,
            goal_status=self._goal_status,
        )

        execution_state = _NavigateBtExecutionState(
            candidate=candidate,
            state_before=state_before,
            session=session,
            confirm_state=self._confirm_state,
        )

        leaf = NavigateRegionLeaf(execution_state)

        return BtSkillExecution(
            root=leaf,
            result_provider=execution_state.poll,
            cancel_request=execution_state.cancel,
        )


class _NavigateBtExecutionState:

    def __init__(
        self,
        candidate,
        state_before,
        session,
        confirm_state,
    ):
        self._candidate = candidate
        self._state_before = state_before
        self._session = session
        self._confirm_state = confirm_state

        self._started = False
        self._cancel_requested = False
        self._terminal_result = None

    @property
    def last_feedback_phase(self):
        return self._session.last_feedback_phase

    @property
    def last_distance_remaining(self):
        return self._session.last_distance_remaining

    def advance(self):
        if self._terminal_result is not None:
            return self._terminal_result

        if not self._started:
            self._started = True

            try:
                self._session.start()
            except Exception as exc:
                return self._set_failure(
                    "Navigation session start failed: "
                    f"{exc}"
                )

            return self._terminal_result

        return self.poll()

    def poll(self):
        if self._terminal_result is not None:
            return self._terminal_result

        if not self._started:
            return None

        outcome = self._session.poll()

        if outcome is None:
            return None

        return self._resolve_outcome(outcome)

    def cancel(self):
        if self._terminal_result is not None:
            return

        if self._cancel_requested:
            return

        self._cancel_requested = True

        if not self._started:
            self._set_failure(
                "Cancelled before navigation goal dispatch"
            )
            return

        self._session.cancel()

    def _resolve_outcome(self, outcome):

        if (
            outcome.terminal_kind == "SUCCEEDED"
            and outcome.result_success is True
        ):
            if (
                outcome.cancel_requested
                or self._cancel_requested
            ):
                return self._set_failure(
                    "cancel_requested; action later succeeded; "
                    f"state not committed: {outcome.message}"
                )

            if self._confirm_state is None:
                return self._set_failure(
                    "Navigation execution succeeded but no "
                    "state confirmation provider is available"
                )

            try:
                confirmed_state = self._confirm_state(
                    self._candidate,
                    self._state_before,
                    outcome,
                )
            except Exception as exc:
                return self._set_failure(
                    f"State confirmation failed: {exc}"
                )

            if not isinstance(confirmed_state, RobotState):
                return self._set_failure(
                    "State confirmation did not return "
                    "a valid RobotState"
                )

            self._terminal_result = SkillExecutionResult(
                success=True,
                message=outcome.message,
                state_after=confirmed_state,
            )
            return self._terminal_result

        return self._set_failure(
            f"Navigation {outcome.terminal_kind}: "
            f"{outcome.message}"
        )

    def _set_failure(self, message):
        self._terminal_result = SkillExecutionResult(
            success=False,
            message=message,
            state_after=self._state_before,
        )
        return self._terminal_result
