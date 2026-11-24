from dataclasses import dataclass
from functools import wraps
from threading import RLock

from .skill_types import (
    SkillCandidate,
    SkillID,
)


def _synchronized(method):

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def _default_goal_builder(region_id):
    from robot_interfaces.action import NavigateToRegion

    goal = NavigateToRegion.Goal()
    goal.region_id = region_id
    return goal


@dataclass(frozen=True)
class NavigateActionOutcome:
    region_id: str
    terminal_kind: str
    goal_status: int | None
    result_success: bool | None
    message: str
    cancel_requested: bool


class NavigateRegionActionSession:

    def __init__(
        self,
        action_client,
        candidate: SkillCandidate,
        goal_builder=None,
        goal_status=None,
    ):
        self._action_client = action_client
        self._candidate = candidate
        self._goal_builder = goal_builder or _default_goal_builder
        self._goal_status = goal_status
        self._lock = RLock()

        self._state = "CREATED"
        self._started = False
        self._goal = None
        self._goal_handle = None
        self._result_future = None
        self._outcome = None

        self._cancel_requested = False
        self._cancel_sent = False
        self._cancel_error = None
        self._cancel_accepted = None

        self._last_feedback_phase = None
        self._last_distance_remaining = None

    @property
    def state(self):
        return self._state

    @property
    def region_id(self):
        return self._candidate.arguments.get("region_id")

    @property
    def cancel_requested(self):
        return self._cancel_requested

    @property
    def cancel_sent(self):
        return self._cancel_sent

    @property
    def cancel_error(self):
        return self._cancel_error

    @property
    def cancel_accepted(self):
        return self._cancel_accepted

    @property
    def last_feedback_phase(self):
        return self._last_feedback_phase

    @property
    def last_distance_remaining(self):
        return self._last_distance_remaining

    @_synchronized
    def start(self) -> None:

        if self._started:
            raise RuntimeError(
                "NavigateRegionActionSession already started"
            )

        self._validate_candidate()
        self._started = True
        self._state = "GOAL_REQUESTED"

        try:
            self._goal = self._goal_builder(self.region_id)

            if self._goal is None:
                raise ValueError(
                    "goal_builder returned no goal"
                )

            goal_future = self._action_client.send_goal_async(
                self._goal,
                feedback_callback=self._on_feedback,
            )

            if goal_future is None:
                raise ValueError(
                    "send_goal_async returned no future"
                )

            goal_future.add_done_callback(
                self._on_goal_response
            )

        except Exception as exc:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message=f"Failed to send navigation goal: {exc}",
            )
            return

    @_synchronized
    def poll(self) -> NavigateActionOutcome | None:
        return self._outcome

    @_synchronized
    def cancel(self) -> None:

        if self._outcome is not None:
            return

        if self._cancel_requested:
            return

        self._cancel_requested = True

        if self._goal_handle is not None:
            self._send_cancel_once()

    def _validate_candidate(self) -> None:

        if self._candidate.skill_id != SkillID.NAVIGATE_REGION:
            raise ValueError(
                "NavigateRegionActionSession requires "
                "NAVIGATE_REGION"
            )

        region_id = self.region_id

        if not isinstance(region_id, str) or not region_id.strip():
            raise ValueError(
                "Missing or invalid region_id"
            )

    @_synchronized
    def _on_goal_response(self, future) -> None:

        if self._outcome is not None:
            return

        if self._goal_handle is not None:
            return

        try:
            goal_handle = future.result()
        except Exception as exc:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message=f"Goal response error: {exc}",
            )
            return

        if goal_handle is None:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message="Goal response was empty",
            )
            return

        if getattr(goal_handle, "accepted", False) is not True:
            self._set_terminal(
                terminal_kind="FAILED",
                goal_status=None,
                result_success=None,
                message="Navigation goal rejected",
            )
            return

        self._goal_handle = goal_handle
        self._state = "GOAL_ACCEPTED"

        if self._cancel_requested:
            self._send_cancel_once()

        try:
            self._result_future = goal_handle.get_result_async()

            if self._result_future is None:
                raise ValueError(
                    "get_result_async returned no future"
                )

            self._result_future.add_done_callback(
                self._on_result_response
            )

        except Exception as exc:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message=f"Result request error: {exc}",
            )

    @_synchronized
    def _on_feedback(self, feedback_msg) -> None:

        if self._outcome is not None:
            return

        try:
            feedback = feedback_msg.feedback
            self._last_feedback_phase = getattr(
                feedback,
                "phase",
                None,
            )
            self._last_distance_remaining = getattr(
                feedback,
                "distance_remaining",
                None,
            )
        except Exception:
            pass

    @_synchronized
    def _on_result_response(self, future) -> None:

        if self._outcome is not None:
            return

        try:
            response = future.result()
        except Exception as exc:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message=f"Result response error: {exc}",
            )
            return

        status = getattr(response, "status", None)
        result = getattr(response, "result", None)

        if status is None:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=None,
                result_success=None,
                message="Result response missing status",
            )
            return

        try:
            goal_status = self._load_goal_status()
        except Exception as exc:
            self._set_terminal(
                terminal_kind="UNKNOWN",
                goal_status=status,
                result_success=None,
                message=f"Cannot interpret GoalStatus: {exc}",
            )
            return

        result_success = (
            getattr(result, "success", None)
            if result is not None
            else None
        )
        result_message = (
            getattr(result, "message", "")
            if result is not None
            else ""
        )

        if status == goal_status.STATUS_SUCCEEDED:
            if result_success is True:
                terminal_kind = "SUCCEEDED"
                message = result_message or "Navigation succeeded"
            elif result_success is False:
                terminal_kind = "FAILED"
                message = result_message or "Navigation failed"
            else:
                terminal_kind = "UNKNOWN"
                message = (
                    "Navigation result missing result.success"
                )

        elif status == goal_status.STATUS_ABORTED:
            terminal_kind = "FAILED"
            message = result_message or "Navigation aborted"

        elif status == goal_status.STATUS_CANCELED:
            terminal_kind = "CANCELED"
            message = result_message or "Navigation canceled"

        else:
            terminal_kind = "UNKNOWN"
            message = f"Unexpected GoalStatus: {status}"

        self._set_terminal(
            terminal_kind=terminal_kind,
            goal_status=status,
            result_success=result_success,
            message=message,
        )

    def _send_cancel_once(self) -> None:

        if self._cancel_sent:
            return

        self._cancel_sent = True

        try:
            cancel_future = (
                self._goal_handle.cancel_goal_async()
            )

            if cancel_future is None:
                raise ValueError(
                    "cancel_goal_async returned no future"
                )

            cancel_future.add_done_callback(
                self._on_cancel_response
            )

        except Exception as exc:
            self._cancel_error = f"Cancel request error: {exc}"

    @_synchronized
    def _on_cancel_response(self, future) -> None:

        try:
            response = future.result()
            self._cancel_accepted = bool(
                getattr(response, "goals_canceling", False)
            )
        except Exception as exc:
            self._cancel_error = f"Cancel response error: {exc}"

    def _load_goal_status(self):

        if self._goal_status is None:
            from action_msgs.msg import GoalStatus

            self._goal_status = GoalStatus

        return self._goal_status

    def _set_terminal(
        self,
        terminal_kind,
        goal_status,
        result_success,
        message,
    ) -> None:

        if self._outcome is not None:
            return

        self._state = "FINAL_RESULT"
        self._outcome = NavigateActionOutcome(
            region_id=self.region_id,
            terminal_kind=terminal_kind,
            goal_status=goal_status,
            result_success=result_success,
            message=message,
            cancel_requested=self._cancel_requested,
        )
