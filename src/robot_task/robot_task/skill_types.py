from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


class SkillID(str, Enum):
    NAVIGATE_REGION = "navigate_region"
    VISUAL_DOCK = "visual_dock"
    DETECT_OBJECT = "detect_object"
    MANIPULATE_PICK = "manipulate_pick"
    VERIFY_RESULT = "verify_result"
    DONE = "done"
    ASK_HUMAN = "ask_human"


@dataclass(frozen=True)
class TaskGoal:
    instruction: str
    target_region_id: str | None = None
    target_object_id: str | None = None


@dataclass(frozen=True)
class RobotState:
    current_region_id: str | None = None
    docking_complete: bool | None = None
    object_detected: bool | None = None
    object_grasped: bool | None = None
    result_verified: bool | None = None


@dataclass(frozen=True)
class SkillCandidate:
    skill_id: SkillID
    score: float
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"Invalid score: {self.score}"
            )


@dataclass(frozen=True)
class SkillDecision:
    selected: SkillCandidate | None
    feasible_candidates: tuple[SkillCandidate, ...] = ()
    rejected_candidates: dict[str, str] = field(default_factory=dict)


class TaskStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_HUMAN = "needs_human"


@dataclass
class SkillExecutionResult:
    success: bool
    message: str
    state_after: RobotState


@dataclass
class TaskProgress:
    status: TaskStatus = TaskStatus.RUNNING
    current_skill: SkillID | None = None
    completed_skills: list[SkillID] = field(default_factory=list)
    step_count: int = 0
    last_message: str = ""