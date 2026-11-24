from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import py_trees

from .skill_types import (
    RobotState,
    SkillCandidate,
    SkillExecutionResult,
)


@dataclass(frozen=True)
class BtSkillExecution:
    root: py_trees.behaviour.Behaviour
    result_provider: (
        Callable[[], SkillExecutionResult | None] | None
    ) = None
    cancel_request: Callable[[], None] | None = None


class BtSkillExecutor:

    def __init__(self, bt_factory):
        self._bt_factory = bt_factory

    def start(
        self,
        candidate: SkillCandidate,
        state: RobotState,
    ):
        try:
            execution_candidate = SkillCandidate(
                skill_id=candidate.skill_id,
                score=candidate.score,
                arguments=deepcopy(candidate.arguments),
            )

            execution = self._bt_factory(
                execution_candidate,
                state,
            )

            if not isinstance(execution, BtSkillExecution):
                raise TypeError(
                    "bt_factory must return BtSkillExecution"
                )

            if not isinstance(
                execution.root,
                py_trees.behaviour.Behaviour,
            ):
                raise TypeError(
                    "BtSkillExecution.root must be a Behaviour"
                )

            if (
                execution.result_provider is not None
                and not callable(execution.result_provider)
            ):
                raise TypeError(
                    "result_provider must be callable"
                )

            if (
                execution.cancel_request is not None
                and not callable(execution.cancel_request)
            ):
                raise TypeError(
                    "cancel_request must be callable"
                )

            return BtSkillExecutionHandle(execution, state)

        except Exception as exc:
            return _FailedBtSkillExecutionHandle(
                SkillExecutionResult(
                    success=False,
                    message=f"BT start failed: {exc}",
                    state_after=state,
                )
            )


class BtSkillExecutionHandle:

    def __init__(self, execution, state):
        self._execution = execution
        self._state = state
        self._result = None
        self._cancel_sent = False

    def poll(self) -> SkillExecutionResult | None:

        if self._result is not None:
            return self._result

        if self._cancel_sent:
            result = self._read_explicit_result()

            if result is None:
                return None

            return self._finish(result)

        try:
            self._execution.root.tick_once()
            status = self._execution.root.status

        except Exception as exc:
            return self._finish(
                self._failure(f"BT tick failed: {exc}")
            )

        if status == py_trees.common.Status.RUNNING:
            return None

        if status == py_trees.common.Status.SUCCESS:
            result = self._read_explicit_result()

            if result is None:
                return self._finish(
                    self._failure(
                        "BT reported SUCCESS without "
                        "explicit result evidence"
                    )
                )

            return self._finish(result)

        if status == py_trees.common.Status.FAILURE:
            result = self._read_explicit_result()

            if result is None or result.success is not False:
                return self._finish(
                    self._failure("BT reported FAILURE")
                )

            return self._finish(result)

        return None

    def cancel(self) -> None:

        if self._result is not None:
            return

        if self._cancel_sent:
            return

        self._cancel_sent = True

        if self._execution.cancel_request is not None:
            try:
                self._execution.cancel_request()
            except Exception:
                pass

        try:
            self._execution.root.stop(
                py_trees.common.Status.INVALID
            )
        except Exception:
            pass

    def _read_explicit_result(self):

        if self._execution.result_provider is None:
            return None

        try:
            return self._execution.result_provider()
        except Exception as exc:
            return self._failure(
                f"BT result provider failed: {exc}"
            )

    def _failure(self, message):
        return SkillExecutionResult(
            success=False,
            message=message,
            state_after=self._state,
        )

    def _finish(self, result):
        self._result = result
        return result


class _FailedBtSkillExecutionHandle:

    def __init__(self, result):
        self._result = result

    def poll(self):
        return self._result

    def cancel(self):
        pass
