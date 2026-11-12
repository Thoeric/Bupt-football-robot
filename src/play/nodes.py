"""Core PLAY subtree leaves: dynamic role assignment, fallback, and role utilities.

Shared PLAY leaves include :class:`AssignRoles`, :class:`IsRole`, and
:class:`WaitForBall`. Role utility leaves and builders include
:class:`MoveToTarget`, :class:`KickAction`, :class:`IsKickWanted`, and
:func:`build_attack_subtree`.

All data is read from the blackboard. Commands are written to
``cmd_key(player_id)`` and finally collected by :class:`CommitTeamCommands`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import py_trees

from ..soccer_framework import PlayContext, Pose2D, RobotCommand
from ..behavior_tree.blackboard import BlackboardKeys, BlackboardClient, cmd_key
from ..behavior_tree.nodes.conditions import IsInKickRange
from ..tactics.targeting import restart as restart_targeting
from .playbook import Playbook, RoleAssignment

if TYPE_CHECKING:
    from ..runtime import SoccerKit


def _read_play_context(
    blackboard: BlackboardClient,
) -> PlayContext | None:
    """Read and narrow the blackboard values needed by PLAY decisions."""

    context = blackboard.read(BlackboardKeys.PLAY_CONTEXT)
    if not isinstance(context, PlayContext):
        return None
    return context


# ----------------------------------------------------------------------
# PLAY dynamic role assignment, written to blackboard
# ----------------------------------------------------------------------


class AssignRoles(py_trees.behaviour.Behaviour):
    """Call :meth:`Playbook.assign_roles` each tick and write the result to ``/team/roles``.

    This is the PLAY subtree's core strategy node. Every :class:`IsRole` condition
    leaf branches from its :class:`RoleAssignment`, so overriding ``Playbook.assign_roles``
    changes the whole team's PLAY role assignment.

    SafetyGuards have already filtered missing GameController and ball data before
    this leaf runs in PLAYING.
    """

    def __init__(self, playbook: Playbook):
        super().__init__("AssignRoles")
        self._playbook = playbook
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            self.blackboard.write(BlackboardKeys.ROLES, RoleAssignment())
            return py_trees.common.Status.SUCCESS
        assignment = self._playbook.assign_roles(context)
        self.blackboard.write(BlackboardKeys.ROLES, assignment)
        return py_trees.common.Status.SUCCESS


# ----------------------------------------------------------------------
# Common condition leaf: IsRole
# ----------------------------------------------------------------------


class IsRole(py_trees.behaviour.Behaviour):
    """Read ``/team/roles`` and test whether a player currently has the ``expected`` role."""

    def __init__(self, player_id: int, expected: str):
        super().__init__(f"IsRole<{expected}>({player_id})")
        self._player_id = player_id
        self._expected = expected
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        assignment = self.blackboard.read(BlackboardKeys.ROLES)
        if not isinstance(assignment, RoleAssignment):
            return py_trees.common.Status.FAILURE
        if assignment.role_of(self._player_id) != self._expected:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ----------------------------------------------------------------------
# Player Selector fallback leaf
# ----------------------------------------------------------------------


class WaitForBall(py_trees.behaviour.Behaviour):
    """Fallback command when no role is assigned; behavior comes from :meth:`Playbook.waiting_command`.

    As the final Player Selector fallback it must always write a command. Even if
    context is absent, it writes a ReadySlot-tagged stop so PlaybookCore does not
    fail into ``StopAll("unsupported state")``.
    """

    def __init__(self, kit: "SoccerKit", playbook: Playbook, player_id: int):
        super().__init__(f"WaitForBall({player_id})")
        self._kit = kit
        self._playbook = playbook
        self._player_id = player_id
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        self._kit.kicker.clear_player(self._player_id)
        context = _read_play_context(self.blackboard)
        if context is None:
            slot = self._kit.config.ready_slot_for_player(self._player_id)
            command = RobotCommand.stop(f"{slot.value} waiting for ball")
        else:
            command = self._playbook.waiting_command(self._player_id, context)
        self.blackboard.write(cmd_key(self._player_id), command)
        return py_trees.common.Status.SUCCESS


# ----------------------------------------------------------------------
# KickAction / IsKickWanted / Role utility leaves
# ----------------------------------------------------------------------

# Callable type aliases used by utility nodes
TargetFn = Callable[[PlayContext], Pose2D]
WantsKickFn = Callable[[PlayContext], bool]
ReasonFn = Callable[[], str]
KickReasonFn = Callable[[Pose2D], str]
KickPowerFn = Callable[[PlayContext], float]


def _default_move_reason(player_id: int) -> str:
    return f"role {player_id} move"


def _default_kick_reason(player_id: int) -> str:
    return f"role {player_id} kick"


@dataclass(frozen=True)
class AttackSubtreeConfig:
    """Parameters for the standard attack subtree builder."""

    target_fn: TargetFn
    kick_target_fn: TargetFn
    wants_kick_fn: WantsKickFn | None = None
    reason_fn: ReasonFn | None = None
    kick_reason_fn: KickReasonFn | None = None
    hold_vyaw: float = 0.0
    # May be a fixed per-role value or a context callback for restart-specific power.
    kick_power: float | KickPowerFn | None = None
    # Forward roles use a controlled walk-turn profile only inside the final
    # 1.2 m around the ball; normal support movement stays fully agile.
    final_ball_approach: bool = False


class MoveToTarget(py_trees.behaviour.Behaviour):
    """Call ``target_fn`` each tick to compute a target and issue a move command.

    Kick hysteresis is cleared before every move to avoid leaking last tick's
    kick state into this tick's movement. If ``target_fn`` raises ``ValueError``,
    this leaf returns FAILURE and the parent Selector falls back.
    """

    def __init__(
        self,
        kit: "SoccerKit",
        player_id: int,
        target_fn: TargetFn,
        *,
        reason_fn: ReasonFn | None = None,
        hold_vyaw: float = 0.0,
        final_ball_approach: bool = False,
    ):
        super().__init__(f"MoveToTarget({player_id})")
        self._kit = kit
        self._player_id = player_id
        self._target_fn = target_fn
        self._reason_fn: ReasonFn = reason_fn or (
            lambda: _default_move_reason(player_id)
        )
        self._hold_vyaw = hold_vyaw
        self._final_ball_approach = final_ball_approach
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            return py_trees.common.Status.FAILURE
        try:
            target = self._target_fn(context)
        except ValueError:
            return py_trees.common.Status.FAILURE

        kit = self._kit
        player_id = self._player_id
        kit.kicker.clear_player(player_id)
        command = kit.motion.move_to_target(
            player_id,
            context,
            target,
            self._reason_fn(),
            hold_vyaw=self._hold_vyaw,
            final_ball_approach=self._final_ball_approach,
        )
        self.blackboard.write(cmd_key(player_id), command)
        return py_trees.common.Status.SUCCESS


class KickAction(py_trees.behaviour.Behaviour):
    """Call ``kick_target_fn`` each tick to compute the aim point and issue a kick command.

    Returns FAILURE when ``kick_target_fn`` raises ``ValueError``.
    """

    def __init__(
        self,
        kit: "SoccerKit",
        player_id: int,
        kick_target_fn: TargetFn,
        *,
        reason_fn: KickReasonFn | None = None,
        kick_power: float | KickPowerFn | None = None,
    ):
        super().__init__(f"KickAction({player_id})")
        self._kit = kit
        self._player_id = player_id
        self._kick_target_fn = kick_target_fn
        self._reason_fn = reason_fn
        self._kick_power = kick_power
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            return py_trees.common.Status.FAILURE
        try:
            kt = self._kick_target_fn(context)
        except ValueError:
            return py_trees.common.Status.FAILURE

        ball = context.known_ball
        kick_theta = math.atan2(kt.y - ball.y, kt.x - ball.x)
        kit = self._kit
        player_id = self._player_id
        reason = (
            self._reason_fn(kt)
            if self._reason_fn is not None
            else _default_kick_reason(player_id)
        )
        power = self._kick_power(context) if callable(self._kick_power) else self._kick_power
        command = kit.motion.kick_command(
            player_id,
            context,
            kick_theta,
            reason,
            power=power,
        )
        self.blackboard.write(cmd_key(player_id), command)
        return py_trees.common.Status.SUCCESS


class IsKickWanted(py_trees.behaviour.Behaviour):
    """Condition leaf that calls ``wants_kick_fn`` to decide whether this tick wants a kick."""

    def __init__(
        self,
        kit: "SoccerKit",
        player_id: int,
        wants_kick_fn: WantsKickFn,
    ):
        super().__init__(f"IsKickWanted({player_id})")
        self._kit = kit
        self._player_id = player_id
        self._wants_kick_fn = wants_kick_fn
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            return py_trees.common.Status.FAILURE
        try:
            if self._wants_kick_fn(context):
                return py_trees.common.Status.SUCCESS
        except ValueError:
            pass
        return py_trees.common.Status.FAILURE


# ----------------------------------------------------------------------
# Set Play positioning nodes
# ----------------------------------------------------------------------


class IsSetPlayActive(py_trees.behaviour.Behaviour):
    """SUCCESS during any set play, EXCEPT for the designated kicker in own set plays.

    The designated kicker skips this guard so they fall through to their role
    subtree where :meth:`wants_to_kick` returns True and normal kick logic runs.
    """

    def __init__(
        self,
        kit: "SoccerKit",
        playbook: Playbook,
        player_id: int,
    ):
        super().__init__(f"IsSetPlayActive({player_id})")
        self._kit = kit
        self._playbook = playbook
        self._player_id = player_id
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            return py_trees.common.Status.FAILURE
        if not self._playbook.is_set_play_active(context):
            return py_trees.common.Status.FAILURE
        # Own set play: let the designated kicker skip this guard.
        if self._playbook.is_own_set_play(context):
            role = self._my_role()
            if role and self._playbook.is_set_play_kicker(role, context):
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS

    def _my_role(self) -> str | None:
        assignment = self.blackboard.read(BlackboardKeys.ROLES)
        if isinstance(assignment, RoleAssignment):
            return assignment.role_of(self._player_id)
        return None


class GoSetPlayTarget(py_trees.behaviour.Behaviour):
    """Move to the set-play position from the playbook's positioning table."""

    def __init__(
        self,
        kit: "SoccerKit",
        playbook: Playbook,
        player_id: int,
    ):
        super().__init__(f"GoSetPlayTarget({player_id})")
        self._kit = kit
        self._playbook = playbook
        self._player_id = player_id
        self.blackboard = BlackboardClient(name=self.name)

    def update(self) -> py_trees.common.Status:
        context = _read_play_context(self.blackboard)
        if context is None:
            return py_trees.common.Status.FAILURE
        role = self._my_role()
        if role is None:
            return py_trees.common.Status.FAILURE
        target = self._playbook.get_set_play_target(role, context)
        kit = self._kit
        player_id = self._player_id
        kit.kicker.clear_player(player_id)
        opponent_restart = not self._playbook.is_own_set_play(context)
        if opponent_restart:
            robot = context.teammates.get(player_id)
            if robot is not None and robot.pose is not None:
                distance_to_ball = math.hypot(
                    robot.pose.x - context.known_ball.x,
                    robot.pose.y - context.known_ball.y,
                )
                if distance_to_ball < 1.80:
                    command = kit.motion.escape_restart_ball(
                        player_id,
                        context,
                        f"set play {role}: emergency escape",
                    )
                    self.blackboard.write(cmd_key(player_id), command)
                    return py_trees.common.Status.SUCCESS
                target = restart_targeting.safe_restart_position(
                    kit.config,
                    kit.field,
                    robot.pose,
                    context.known_ball,
                    context.known_game,
                    min_distance=max(
                        2.2,
                        kit.config.strategy.opponent_restart_avoid_distance_m,
                    ),
                )
                if math.hypot(
                    target.x - robot.pose.x,
                    target.y - robot.pose.y,
                ) <= 0.05:
                    command = RobotCommand.stop(
                        f"set play {role}: legal and still",
                    )
                    self.blackboard.write(cmd_key(player_id), command)
                    return py_trees.common.Status.SUCCESS
        command = kit.motion.move_to_target(
            player_id,
            context,
            target,
            f"set play {role}",
            arrive_distance=0.25,
            avoid_opponents=opponent_restart,
        )
        self.blackboard.write(cmd_key(player_id), command)
        return py_trees.common.Status.SUCCESS

    def _my_role(self) -> str | None:
        assignment = self.blackboard.read(BlackboardKeys.ROLES)
        if isinstance(assignment, RoleAssignment):
            return assignment.role_of(self._player_id)
        return None


# ----------------------------------------------------------------------
# Default assembly factory: build_attack_subtree
# ----------------------------------------------------------------------


def build_attack_subtree(
    kit: "SoccerKit",
    player_id: int,
    config: AttackSubtreeConfig,
) -> py_trees.behaviour.Behaviour:
    """Assemble the standard attack subtree: conditional kick, otherwise move.

    Shape::

    Selector(Attack)
    |-- Sequence(KickBranch)
    |   |-- IsInKickRange
    |   `-- KickAction
    `-- MoveToTarget

    When ``config.wants_kick_fn`` is ``None``, :class:`IsKickWanted` is omitted,
    equivalent to "always willing to kick". SafetyGuards ensure visible ball data
    before this PLAY subtree runs.
    """

    kick_children: list[py_trees.behaviour.Behaviour] = []
    if config.wants_kick_fn is not None:
        kick_children.append(IsKickWanted(kit, player_id, config.wants_kick_fn))
    kick_children.extend(
        [
            IsInKickRange(player_id, kit.kicker),
            KickAction(
                kit,
                player_id,
                config.kick_target_fn,
                reason_fn=config.kick_reason_fn,
                kick_power=config.kick_power,
            ),
        ],
    )

    return py_trees.composites.Selector(
        name=f"Attack({player_id})",
        memory=False,
        children=[
            py_trees.composites.Sequence(
                name=f"KickBranch({player_id})",
                memory=False,
                children=kick_children,
            ),
            MoveToTarget(
                kit,
                player_id,
                config.target_fn,
                reason_fn=config.reason_fn,
                hold_vyaw=config.hold_vyaw,
                final_ball_approach=config.final_ball_approach,
            ),
        ],
    )
