from dataclasses import dataclass

from .skill_types import (
    RobotState,
    SkillExecutionResult,
    SkillID,
    TaskGoal,
    TaskProgress,
    TaskStatus,
)


@dataclass(frozen=True)
class TaskRunResult:
    progress: TaskProgress
    final_state: RobotState
    execution_results: tuple[SkillExecutionResult, ...]


class TaskManager:

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

    def run(
        self,
        goal: TaskGoal,
        initial_state: RobotState,
    ) -> TaskRunResult:

        progress = TaskProgress()
        state = initial_state
        execution_results = []

        while True:

            candidates = self.selector.score_candidates(
                goal,
                state,
            )

            decision = self.decision_engine.decide(
                candidates,
                goal,
                state,
            )

            progress.step_count += 1

            selected = decision.selected

            if selected is None:
                progress.status = TaskStatus.FAILED
                progress.last_message = (
                    "No feasible skill candidates"
                )
                break

            skill_id = selected.skill_id
            progress.current_skill = skill_id

            if skill_id == SkillID.DONE:
                if state.result_verified is True:
                    progress.status = TaskStatus.SUCCEEDED
                    progress.last_message = "Task completed"
                else:
                    progress.status = TaskStatus.FAILED
                    progress.last_message = (
                        "DONE selected but task result "
                        "is not verified"
                    )
                break

            if skill_id == SkillID.ASK_HUMAN:
                progress.status = TaskStatus.NEEDS_HUMAN
                progress.last_message = (
                    "Human intervention requested"
                )
                break

            if progress.step_count >= self.max_steps:
                progress.status = TaskStatus.FAILED
                progress.last_message = (
                    "Reached maximum decision steps "
                    f"({self.max_steps}) without completion"
                )
                break

            result = self.executor.execute(
                selected,
                state,
            )

            execution_results.append(result)
            progress.last_message = result.message

            if result.success is not True:
                progress.status = TaskStatus.FAILED
                break

            state = result.state_after
            progress.completed_skills.append(skill_id)

        return TaskRunResult(
            progress=progress,
            final_state=state,
            execution_results=tuple(execution_results),
        )
