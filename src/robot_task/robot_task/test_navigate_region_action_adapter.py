import unittest
from types import SimpleNamespace

from .navigate_region_action_adapter import (
    NavigateRegionActionSession,
)
from .skill_types import (
    SkillCandidate,
    SkillID,
)


class FakeGoalStatus:
    STATUS_UNKNOWN = 0
    STATUS_ACCEPTED = 1
    STATUS_EXECUTING = 2
    STATUS_CANCELING = 3
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


class FakeFuture:

    def __init__(self):
        self._done = False
        self._result = None
        self._exception = None
        self._callbacks = []

    def add_done_callback(self, callback):
        if self._done:
            callback(self)
        else:
            self._callbacks.append(callback)

    def set_result(self, result):
        if self._done:
            raise RuntimeError("FakeFuture already completed")

        self._result = result
        self._done = True

        for callback in list(self._callbacks):
            callback(self)

    def set_exception(self, exception):
        if self._done:
            raise RuntimeError("FakeFuture already completed")

        self._exception = exception
        self._done = True

        for callback in list(self._callbacks):
            callback(self)

    def result(self):
        if not self._done:
            raise RuntimeError("FakeFuture is not done")

        if self._exception is not None:
            raise self._exception

        return self._result


class FakeGoalHandle:

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result_future = FakeFuture()
        self.cancel_future = FakeFuture()
        self.get_result_calls = 0
        self.cancel_calls = 0

    def get_result_async(self):
        self.get_result_calls += 1
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_future


class FakeActionClient:

    def __init__(self):
        self.send_exception = None
        self.send_calls = []
        self.goal_futures = []
        self.feedback_callbacks = []

    def send_goal_async(self, goal, feedback_callback=None):
        if self.send_exception is not None:
            raise self.send_exception

        future = FakeFuture()
        self.send_calls.append(goal)
        self.goal_futures.append(future)
        self.feedback_callbacks.append(feedback_callback)
        return future


class ImmediateActionClient:

    def __init__(self, goal_result):
        self.goal_result = goal_result
        self.send_calls = []
        self.feedback_callbacks = []

    def send_goal_async(self, goal, feedback_callback=None):
        self.send_calls.append(goal)
        self.feedback_callbacks.append(feedback_callback)

        future = FakeFuture()
        future.set_result(self.goal_result)
        return future


class FakeGoal:

    def __init__(self):
        self.region_id = None


def fake_goal_builder(region_id):
    goal = FakeGoal()
    goal.region_id = region_id
    return goal


def make_nav_candidate(region_id="experiment_table_A"):
    return SkillCandidate(
        skill_id=SkillID.NAVIGATE_REGION,
        score=0.90,
        arguments={"region_id": region_id},
    )


def make_session(
    candidate=None,
    client=None,
    goal_builder=fake_goal_builder,
):
    return NavigateRegionActionSession(
        action_client=client or FakeActionClient(),
        candidate=candidate or make_nav_candidate(),
        goal_builder=goal_builder,
        goal_status=FakeGoalStatus,
    )


def make_result_response(
    status,
    success=None,
    message="",
    include_result=True,
    include_success=True,
):
    if not include_result:
        return SimpleNamespace(status=status)

    result = SimpleNamespace(message=message)

    if include_success:
        result.success = success

    return SimpleNamespace(status=status, result=result)


def make_feedback(phase, distance_remaining):
    return SimpleNamespace(
        feedback=SimpleNamespace(
            phase=phase,
            distance_remaining=distance_remaining,
        )
    )


class NavigateRegionActionSessionTest(unittest.TestCase):

    def test_c01_region_id_mapping_and_single_send(self):
        client = FakeActionClient()
        candidate = make_nav_candidate()
        session = make_session(candidate, client)

        session.start()

        self.assertEqual(len(client.send_calls), 1)
        goal = client.send_calls[0]
        self.assertEqual(goal.region_id, "experiment_table_A")
        self.assertFalse(hasattr(goal, "score"))
        self.assertEqual(session.region_id, "experiment_table_A")
        self.assertIsNone(session.poll())

        with self.assertRaises(RuntimeError):
            session.start()

        self.assertEqual(len(client.send_calls), 1)

    def test_c02_invalid_candidates_do_not_send(self):
        cases = [
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
            SkillCandidate(
                skill_id=SkillID.NAVIGATE_REGION,
                score=0.50,
                arguments={"region_id": 42},
            ),
        ]

        for candidate in cases:
            client = FakeActionClient()
            session = make_session(candidate, client)

            with self.assertRaises(ValueError):
                session.start()

            self.assertEqual(client.send_calls, [])

    def test_c03_goal_request_pending(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        for _ in range(3):
            self.assertIsNone(session.poll())

        self.assertEqual(len(client.send_calls), 1)
        self.assertEqual(session.state, "GOAL_REQUESTED")

    def test_c04_accepted_and_feedback_are_not_terminal(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        self.assertEqual(handle.get_result_calls, 1)
        self.assertIsNone(session.poll())

        client.feedback_callbacks[0](
            make_feedback("NAVIGATING", 1.5)
        )
        client.feedback_callbacks[0](
            make_feedback("NAVIGATING", 0.5)
        )

        self.assertIsNone(session.poll())
        self.assertEqual(session.state, "GOAL_ACCEPTED")
        self.assertEqual(
            session.last_feedback_phase,
            "NAVIGATING",
        )
        self.assertEqual(session.last_distance_remaining, 0.5)

    def test_c05_goal_rejected_is_terminal_failure(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=False)
        client.goal_futures[0].set_result(handle)

        outcome = session.poll()

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.terminal_kind, "FAILED")
        self.assertIsNone(outcome.result_success)
        self.assertIsNone(outcome.goal_status)
        self.assertEqual(handle.get_result_calls, 0)
        self.assertEqual(session.state, "FINAL_RESULT")

    def test_c06_succeeded_requires_status_and_result_success(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Arrived",
            )
        )

        outcome = session.poll()

        self.assertEqual(outcome.terminal_kind, "SUCCEEDED")
        self.assertEqual(
            outcome.goal_status,
            FakeGoalStatus.STATUS_SUCCEEDED,
        )
        self.assertIs(outcome.result_success, True)
        self.assertEqual(outcome.message, "Arrived")
        self.assertIs(outcome, session.poll())

    def test_c07_status_succeeded_with_result_false_fails(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=False,
                message="Navigation result false",
            )
        )

        outcome = session.poll()

        self.assertEqual(outcome.terminal_kind, "FAILED")
        self.assertIs(outcome.result_success, False)
        self.assertEqual(outcome.message, "Navigation result false")

    def test_c08_aborted_and_canceled_are_not_success(self):
        for status, expected_kind in (
            (FakeGoalStatus.STATUS_ABORTED, "FAILED"),
            (FakeGoalStatus.STATUS_CANCELED, "CANCELED"),
        ):
            client = FakeActionClient()
            session = make_session(client=client)
            session.start()

            handle = FakeGoalHandle(accepted=True)
            client.goal_futures[0].set_result(handle)
            handle.result_future.set_result(
                make_result_response(
                    status,
                    success=False,
                    message=f"status {status}",
                )
            )

            outcome = session.poll()

            self.assertEqual(
                outcome.terminal_kind,
                expected_kind,
            )
            self.assertNotEqual(
                outcome.terminal_kind,
                "SUCCEEDED",
            )

    def test_c09_cancel_accepted_is_not_terminal(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        session.cancel()

        self.assertTrue(session.cancel_requested)
        self.assertTrue(session.cancel_sent)
        self.assertEqual(handle.cancel_calls, 1)
        self.assertIsNone(session.poll())

        handle.cancel_future.set_result(
            SimpleNamespace(goals_canceling=True)
        )

        self.assertTrue(session.cancel_accepted)
        self.assertIsNone(session.poll())
        self.assertEqual(session.state, "GOAL_ACCEPTED")

        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_CANCELED,
                success=False,
                message="Navigation canceled",
            )
        )

        outcome = session.poll()
        self.assertEqual(outcome.terminal_kind, "CANCELED")
        self.assertTrue(outcome.cancel_requested)

    def test_c10_cancel_before_goal_response(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        session.cancel()
        self.assertTrue(session.cancel_requested)
        self.assertFalse(session.cancel_sent)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        self.assertEqual(handle.cancel_calls, 1)
        self.assertTrue(session.cancel_sent)
        self.assertIsNone(session.poll())

        # Goal rejected before accepted: no cancel attempt
        rejected_client = FakeActionClient()
        rejected_session = make_session(
            client=rejected_client
        )
        rejected_session.start()
        rejected_session.cancel()

        rejected_handle = FakeGoalHandle(accepted=False)
        rejected_client.goal_futures[0].set_result(
            rejected_handle
        )

        outcome = rejected_session.poll()
        self.assertEqual(outcome.terminal_kind, "FAILED")
        self.assertTrue(outcome.cancel_requested)
        self.assertEqual(rejected_handle.cancel_calls, 0)

    def test_c11_late_success_after_cancel_keeps_fact(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        session.cancel()
        self.assertEqual(handle.cancel_calls, 1)

        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Late success",
            )
        )

        outcome = session.poll()

        self.assertEqual(outcome.terminal_kind, "SUCCEEDED")
        self.assertIs(outcome.result_success, True)
        self.assertTrue(outcome.cancel_requested)
        self.assertTrue(session.cancel_requested)
        self.assertFalse(hasattr(outcome, "state_after"))

    def test_c12_old_session_callbacks_do_not_pollute_new_session(self):
        client = FakeActionClient()

        old_session = make_session(client=client)
        new_session = make_session(client=client)

        old_session.start()
        new_session.start()

        old_handle = FakeGoalHandle(accepted=True)
        new_handle = FakeGoalHandle(accepted=True)

        client.goal_futures[0].set_result(old_handle)
        client.goal_futures[1].set_result(new_handle)

        client.feedback_callbacks[0](
            make_feedback("OLD", 9.0)
        )

        self.assertEqual(
            old_session.last_feedback_phase,
            "OLD",
        )
        self.assertIsNone(new_session.last_feedback_phase)

        old_handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="old result",
            )
        )

        old_outcome = old_session.poll()
        self.assertEqual(old_outcome.terminal_kind, "SUCCEEDED")
        self.assertEqual(old_outcome.message, "old result")
        self.assertIsNone(new_session.poll())

        new_handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_ABORTED,
                success=False,
                message="new result",
            )
        )

        new_outcome = new_session.poll()
        self.assertEqual(new_outcome.terminal_kind, "FAILED")
        self.assertEqual(new_outcome.message, "new result")

    def test_c13_repeated_start_cancel_poll_are_idempotent(self):
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()

        with self.assertRaises(RuntimeError):
            session.start()

        self.assertEqual(len(client.send_calls), 1)

        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)

        session.cancel()
        session.cancel()

        self.assertEqual(handle.cancel_calls, 1)

        client.feedback_callbacks[0](
            make_feedback("NAVIGATING", 2.0)
        )
        self.assertEqual(
            session.last_feedback_phase,
            "NAVIGATING",
        )

        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="done",
            )
        )

        first = session.poll()
        second = session.poll()

        self.assertIs(first, second)
        self.assertEqual(first.terminal_kind, "SUCCEEDED")
        self.assertEqual(session.state, "FINAL_RESULT")

        # 终态后的 feedback 不覆盖 final fact
        client.feedback_callbacks[0](
            make_feedback("LATE", 0.1)
        )
        self.assertEqual(session.last_feedback_phase, "NAVIGATING")
        self.assertIs(first, session.poll())
        self.assertEqual(len(client.send_calls), 1)

    def test_c14_future_errors_and_missing_fields_are_conservative(self):
        # Goal request Future 异常
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()
        client.goal_futures[0].set_exception(
            RuntimeError("goal boom")
        )

        outcome = session.poll()
        self.assertEqual(outcome.terminal_kind, "UNKNOWN")
        self.assertIn("goal boom", outcome.message)
        self.assertFalse(outcome.result_success)

        # Result Future 异常
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()
        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_exception(
            RuntimeError("result boom")
        )

        outcome = session.poll()
        self.assertEqual(outcome.terminal_kind, "UNKNOWN")
        self.assertIn("result boom", outcome.message)

        # Cancel Future 异常：不当作终态，不伪造取消成功
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()
        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        session.cancel()
        handle.cancel_future.set_exception(
            RuntimeError("cancel boom")
        )

        self.assertIsNone(session.poll())
        self.assertIn("cancel boom", session.cancel_error)

        # Result 缺少 result.success 字段
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()
        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                include_success=False,
            )
        )

        outcome = session.poll()
        self.assertEqual(outcome.terminal_kind, "UNKNOWN")
        self.assertIn(
            "missing result.success",
            outcome.message,
        )

        # Result 缺少 result 对象
        client = FakeActionClient()
        session = make_session(client=client)
        session.start()
        handle = FakeGoalHandle(accepted=True)
        client.goal_futures[0].set_result(handle)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                include_result=False,
            )
        )

        outcome = session.poll()
        self.assertEqual(outcome.terminal_kind, "UNKNOWN")

    def test_c15_already_completed_goal_future_accepted(self):
        handle = FakeGoalHandle(accepted=True)
        client = ImmediateActionClient(handle)

        session = NavigateRegionActionSession(
            action_client=client,
            candidate=make_nav_candidate(),
            goal_builder=fake_goal_builder,
            goal_status=FakeGoalStatus,
        )

        session.start()

        self.assertEqual(session.state, "GOAL_ACCEPTED")
        self.assertEqual(handle.get_result_calls, 1)
        self.assertIsNone(session.poll())
        self.assertEqual(len(client.send_calls), 1)

    def test_c16_already_completed_goal_future_rejected(self):
        handle = FakeGoalHandle(accepted=False)
        client = ImmediateActionClient(handle)

        session = NavigateRegionActionSession(
            action_client=client,
            candidate=make_nav_candidate(),
            goal_builder=fake_goal_builder,
            goal_status=FakeGoalStatus,
        )

        session.start()

        outcome = session.poll()

        self.assertEqual(session.state, "FINAL_RESULT")
        self.assertEqual(outcome.terminal_kind, "FAILED")
        self.assertEqual(handle.get_result_calls, 0)
        self.assertEqual(len(client.send_calls), 1)

    def test_c17_already_completed_goal_and_result_futures(self):
        handle = FakeGoalHandle(accepted=True)
        handle.result_future.set_result(
            make_result_response(
                FakeGoalStatus.STATUS_SUCCEEDED,
                success=True,
                message="Immediate result",
            )
        )
        client = ImmediateActionClient(handle)

        session = NavigateRegionActionSession(
            action_client=client,
            candidate=make_nav_candidate(),
            goal_builder=fake_goal_builder,
            goal_status=FakeGoalStatus,
        )

        session.start()

        outcome = session.poll()

        self.assertEqual(session.state, "FINAL_RESULT")
        self.assertEqual(outcome.terminal_kind, "SUCCEEDED")
        self.assertEqual(outcome.message, "Immediate result")
        self.assertEqual(handle.get_result_calls, 1)
        self.assertEqual(len(client.send_calls), 1)


if __name__ == "__main__":
    unittest.main()
