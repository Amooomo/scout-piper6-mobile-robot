from .skill_types import (
    TaskGoal,
    RobotState,
    SkillCandidate,
    SkillDecision,
)

from .skill_registry import SkillRegistry


class SkillDecisionEngine:
    """
    将模型给出的候选分数与 SkillRegistry 的可行性检查结合。

    只允许从可执行的 Skill 中选择分数最高的一个。
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def decide(
        self,
        candidates: list[SkillCandidate],
        goal: TaskGoal,
        state: RobotState,
    ) -> SkillDecision:

        feasible_candidates = []
        rejected_candidates = {}

        # 逐个检查候选 Skill
        for candidate in candidates:

            allowed, reason = self.registry.check(
                candidate.skill_id,
                goal,
                state,
            )

            if allowed:
                feasible_candidates.append(candidate)

            else:
                rejected_candidates[
                    candidate.skill_id.value
                ] = reason

        # 按照分数从高到低排序
        feasible_candidates.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        # 选择最高分的可执行 Skill
        selected = (
            feasible_candidates[0]
            if feasible_candidates
            else None
        )

        return SkillDecision(
            selected=selected,
            feasible_candidates=tuple(feasible_candidates),
            rejected_candidates=rejected_candidates,
        )