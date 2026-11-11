from dataclasses import dataclass
from typing import Callable

from .skill_types import TaskGoal, RobotState, SkillID


Precondition = Callable[
    [TaskGoal, RobotState],
    tuple[bool, str]
]


@dataclass(frozen=True)
class SkillSpec:
    skill_id: SkillID
    precondition: Precondition


class SkillRegistry:

    def __init__(self):

        self._skills = {
            SkillID.NAVIGATE_REGION: SkillSpec(
                SkillID.NAVIGATE_REGION,
                self._can_navigate
            ),

            SkillID.VISUAL_DOCK: SkillSpec(
                SkillID.VISUAL_DOCK,
                self._can_visual_dock
            ),

            SkillID.DETECT_OBJECT: SkillSpec(
                SkillID.DETECT_OBJECT,
                self._can_detect_object
            ),

            SkillID.MANIPULATE_PICK: SkillSpec(
                SkillID.MANIPULATE_PICK,
                self._can_manipulate
            ),

            SkillID.VERIFY_RESULT: SkillSpec(
                SkillID.VERIFY_RESULT,
                self._can_verify
            ),

            SkillID.DONE: SkillSpec(
                SkillID.DONE,
                self._can_finish
            ),

            SkillID.ASK_HUMAN: SkillSpec(
                SkillID.ASK_HUMAN,
                self._always_available
            ),
        }

    def check(
        self,
        skill_id: SkillID,
        goal: TaskGoal,
        state: RobotState
    ) -> tuple[bool, str]:

        spec = self._skills.get(skill_id)

        if spec is None:
            return False, "Unknown skill"

        return spec.precondition(goal, state)

    @staticmethod
    def _can_navigate(goal, state):

        if not goal.target_region_id:
            return False, "Target region is missing"

        if state.current_region_id == goal.target_region_id:
            return False, "Already at target region"

        return True, "Navigation target is available"

    @staticmethod
    def _can_visual_dock(goal, state):

        if not goal.target_region_id:
            return False, "Target region is missing"

        if state.current_region_id != goal.target_region_id:
            return False, "Robot has not reached target region"

        if state.docking_complete is not False:
            return False, "Docking state is not confirmed incomplete"

        return True, "Ready for visual docking"

    @staticmethod
    def _can_detect_object(goal, state):

        if state.docking_complete is not True:
            return False, "Visual docking is not complete"

        if state.object_detected is True:
            return False, "Object already detected"

        return True, "Ready for object detection"

    @staticmethod
    def _can_manipulate(goal, state):

        if state.docking_complete is not True:
            return False, "Visual docking is not complete"

        if state.object_detected is not True:
            return False, "Target object has not been detected"

        if state.object_grasped is not False:
            return False, "Grasp state is not confirmed"

        return True, "Ready for manipulation"

    @staticmethod
    def _can_verify(goal, state):

        if state.object_grasped is not True:
            return False, "Object has not been grasped"

        if state.result_verified is True:
            return False, "Result already verified"

        return True, "Ready for verification"

    @staticmethod
    def _can_finish(goal, state):

        if state.result_verified is not True:
            return False, "Task result is not verified"

        return True, "Task completed"

    @staticmethod
    def _always_available(goal, state):

        return True, "Fallback: human intervention"