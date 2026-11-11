
# Scout Mini + Piper6 Mobile Manipulation

A ROS 2-based modular mobile manipulation framework.

## Project Overview

This project integrates Scout Mini mobile robot
navigation with a high-level task and skill framework,
with future support for Piper6 manipulation.

The current research direction is inspired by SayCan:
select one feasible skill, execute it, update robot
state, and select the next skill.

## Main ROS 2 Packages

- robot_interfaces: Custom ROS 2 interfaces and Actions
- robot_nav: Region-based navigation skill and Nav2 integration
- robot_task: Task-level behavior trees and skill selection

## Current Development Status

- Region-based NavigateSkill: Mock tested
- ROS 2 Action cancellation: Mock tested
- Task-level behavior tree: Basic implementation completed
- SkillRegistry: Implemented and tested
- MockSkillSelector: Implemented and tested
- SkillDecisionEngine: Implemented and tested
- MockSkillExecutor: Navigation unit test passed

## Next Development Goal

Implement the complete mock task loop:

TaskGoal
→ Skill Selection
→ Feasibility Check
→ Skill Execution
→ RobotState Update
→ Next Skill Selection

## External Workspaces

The following workspaces are maintained separately:

- scout_ws
- fast_lio_ws
- ws_livox

Their dependency versions, configuration, and setup
instructions will be documented later.

## Notes

Mock tests do not imply that real-robot navigation
or manipulation has passed safety validation.
